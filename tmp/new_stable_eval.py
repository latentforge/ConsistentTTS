"""Gen-Gen COS and WER evaluation for the single-pass learnable VQ query.

The harness mirrors the testing section of experiment_qwen_v3.ipynb: one batched
synthesize call through Method2Strategy.generate_batch_group_all, seeded at 42 with the
surrounding RNG state restored afterwards.
"""

import argparse
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", default="vq", choices=["vq", "commit", "random", "plain"])
    parser.add_argument("--ckpt", default="ckpt/stable_query_vq_k32_s1.pt")
    parser.add_argument("--tag", default="stable_vq_s1")
    parser.add_argument("--anchor-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", default="ckpt/Qwen3-TTS-12Hz-1.7B-VoiceDesign-HF")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import gc
    import random
    from enum import Enum

    import numpy as np
    import soundfile as sf
    import torch
    from transformers.models.qwen3_tts.generation_qwen3_tts import Qwen3TTSGenerationMixin

    from spk_incon.datasets import DatasetType, create_dataset
    from spk_incon.metrics.presets import GenerationMethod, SynthesisConfig
    from spk_incon.metrics.strategies import create_strategy
    from spk_incon.models.stable_qwen3_tts import (
        StableQwen3TTSConfig,
        StableQwen3TTSForConditionalGeneration,
    )
    from spk_incon.utils.evaluate import EvaluationPipeline
    from voicestudio.models.qwen3_tts import Qwen3TTSProcessor

    import torchaudio

    def _load_with_soundfile(path, *unused_args, **unused_kwargs):
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T), int(rate)

    def _save_with_soundfile(uri, src, sample_rate, **unused_kwargs):
        audio = src.detach().cpu().numpy()
        sf.write(str(uri), audio.T if audio.ndim > 1 else audio, int(sample_rate))

    # torchaudio routes through torchcodec, whose ffmpeg shared objects are absent here.
    torchaudio.load = _load_with_soundfile
    torchaudio.save = _save_with_soundfile

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    started = time.perf_counter()

    def log(message):
        print(f"[{(time.perf_counter() - started) / 60:6.1f} min] {message}", flush=True)

    log(f"mode={args.mode} tag={args.tag} ckpt={args.ckpt}")

    config = StableQwen3TTSConfig.from_pretrained(args.model, num_query_tokens=32)
    if args.anchor_frames is not None:
        config.anchor_num_frames = args.anchor_frames
    model = (
        StableQwen3TTSForConditionalGeneration.from_pretrained(
            args.model, config=config, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to(device)
        .eval()
    )
    processor = Qwen3TTSProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    processor.audio_tokenizer.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if args.mode in ("vq", "commit"):
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        with torch.no_grad():
            model.query.data = checkpoint["param"].to(model.query.device, model.query.dtype)
        model._query_ready = True
        nearest = model.get_projected_text_vocab()[
            torch.tensor(checkpoint["token_ids"], device=model.query.device)
        ]
        gap = float((model.query.float() - nearest).norm(dim=-1).mean())
        if args.mode == "commit":
            with torch.no_grad():
                model.query.data = nearest.to(model.query.dtype)
        log(f"{args.mode} query, gap between continuous and nearest tokens {gap:.3f}")
    elif args.mode == "random":
        model.init_query(generator=torch.Generator(device=model.query.device).manual_seed(args.seed))
    log(f"anchor_num_frames={config.anchor_num_frames}")

    sample_rate = int(processor.feature_extractor.sampling_rate)
    upsample_rate = processor.audio_tokenizer.get_decode_upsample_rate()
    talker_kwargs = dict(
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
        repetition_penalty=1.05,
    )

    class TestModel:
        @classmethod
        def seed_everything(cls, seed: int = 42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        @classmethod
        @torch.no_grad()
        def synthesize(
            cls,
            text: str,
            output_path: Path,
            reference_audio: Path | None = None,
            style_prompt: str | None = None,
            speaker_id: str | None = None,
        ) -> bool:
            rng_state = {
                "random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            cls.seed_everything(42)

            texts = list(text) if isinstance(text, (tuple, list)) else [text]
            paths = list(output_path) if isinstance(output_path, (tuple, list)) else [output_path]
            prompts = list(style_prompt) if isinstance(style_prompt, (tuple, list)) else [style_prompt] * len(texts)
            Path(paths[0]).parent.mkdir(parents=True, exist_ok=True)

            waveforms = []
            for start in range(0, len(texts), args.batch):
                chunk = slice(start, start + args.batch)
                input_ids, instruct_ids = [], []
                for content, persona in zip(texts[chunk], prompts[chunk]):
                    input_ids.append(
                        tokenizer(processor._build_synthesis_text(content), return_tensors="pt").input_ids.to(device)
                    )
                    instruct_ids.append(
                        tokenizer(processor._build_instruct_text(persona or ""), return_tensors="pt").input_ids.to(device)
                    )

                if args.mode == "plain":
                    codes, _ = Qwen3TTSGenerationMixin.generate(
                        model,
                        input_ids=input_ids,
                        instruct_ids=instruct_ids,
                        languages=["auto"] * len(input_ids),
                        speakers=[None] * len(input_ids),
                        **talker_kwargs,
                    )
                    decoded = [w.squeeze().float().cpu() for w in processor.batch_decode(codes)]
                else:
                    codes, anchor_frames = model.generate(
                        input_ids=input_ids, instruct_ids=instruct_ids, **talker_kwargs
                    )
                    decoded = [w.squeeze().float().cpu() for w in processor.batch_decode(codes)]
                    decoded = model.trim_anchor(decoded, upsample_rate, anchor_frames)
                waveforms.extend(decoded)
                log(f"  synthesized {min(start + args.batch, len(texts))}/{len(texts)}")

            for path, waveform in zip(paths, waveforms):
                sf.write(str(path), waveform.numpy(), sample_rate)

            random.setstate(rng_state["random"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if rng_state["cuda"]:
                torch.cuda.set_rng_state_all(rng_state["cuda"])

            gc.collect()
            torch.cuda.empty_cache()
            return True

    class ModelType(Enum):
        TEST = model.__class__.__name__

    output_dir = Path("results/Qwen") / f"stable_{args.tag}"
    test_config = SynthesisConfig()
    test_dataset_type = DatasetType.LIBRITTS
    test_dataset_config = test_config.get_dataset_config(test_dataset_type.value)
    test_dataset = create_dataset(test_dataset_type, test_dataset_config, root_dir="./data")

    import io
    import types

    import datasets

    test_dataset.dataset = test_dataset.dataset.cast_column("audio", datasets.Audio(decode=False))

    def get_sample_via_soundfile(self, index):
        sample = self.dataset[index]
        style_prompt = sample["style_prompt"].split(";")[0] if sample["style_prompt"] else None
        audio, rate = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio_path = self.temp_audio_dir / f"temp_audio_{index}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), audio, int(rate))
        return sample["content_prompt"], audio_path, style_prompt, str(sample["spk_id"])

    # The Audio feature decodes through torchcodec, whose ffmpeg shared objects are absent here.
    test_dataset.get_sample = types.MethodType(get_sample_via_soundfile, test_dataset)

    test_model_type = ModelType.TEST
    test_model = TestModel()
    evaluator = EvaluationPipeline(base_dir=output_dir, html=True, verbose=False)
    test_config.generation.output_dir = output_dir

    if args.eval_only:
        log("reusing existing synthesis")
    else:
        log("generating")
        strategy = create_strategy(GenerationMethod.METHOD2, test_config, test_dataset, test_model)
        strategy.generate_batch_group_all(test_dataset_type.value, test_model_type.value)

    log("evaluating")
    exp2_eval_result = evaluator.evaluate_dataset_model(
        dataset_type=test_dataset_type, model_type=test_model_type, methods=[GenerationMethod.METHOD2]
    )
    evaluator.save_results_to_csv(exp2_eval_result, test_dataset_type, test_model_type)

    summary_dir = Path("results/_summary/stable")
    summary_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        method.value: {k: float(v) for k, v in values.items()} for method, values in exp2_eval_result.items()
    }
    import json

    with open(summary_dir / f"{args.tag}.json", "w") as handle:
        json.dump(
            {"tag": args.tag, "mode": args.mode, "ckpt": args.ckpt,
             "anchor_num_frames": config.anchor_num_frames, "metrics": stats},
            handle,
            indent=2,
        )
    for method, values in stats.items():
        log(f"{method}: " + "  ".join(f"{k}={v:.4f}" for k, v in values.items()))
    log(f"wrote {summary_dir / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
