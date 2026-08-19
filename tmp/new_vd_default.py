"""Plain Qwen3-TTS VoiceDesign baseline on method2 (Gen-Gen), NO anchor / NO VQ.
Each utterance is generated directly from the persona (default voice-design generation).
Writes results/Qwen/..._vd_default/ so the stock evaluator can score it."""
import os, json, argparse, shutil, random
from pathlib import Path
import numpy as np, torch

MODEL_CLASS_NAME = "Qwen3TTSForConditionalGeneration"
COMP = [
 "At least, no friend came forwards immediately, and mrs Thornton is not one, I fancy, to wait till tardy kindness comes to find her out.",
 "And the poor men around him-they were poor because they were vicious-out of the pale of his sympathies because they had not his iron nature, and the capabilities that it gives him for being rich.'",
 "The modes of treatment may be ranged under three heads: (one) To eliminate the poison; (two) to antagonize its action; (three) to avert the tendency to death.",
 "Visiting register offices, seeing all manner of unlikely people, and very few in the least likely, absorbed Margaret's time and thoughts for several days.",
 "But though she received caresses and fond words back again, in such profusion as would have gladdened her formerly, yet she felt that there was a secret withheld from her, and she believed it bore serious reference to her mother's health.",
 "And the poor men around him-they were poor because they were vicious-out of the pale of his sympathies because they had not his iron nature, and the capabilities that it gives him for being rich.",
 "mr Bell said they absolutely lived upon water porridge for years-how, he did not know; but long after the creditors had given up hope of any payment of old mr Thornton's debts (if, indeed, they ever had hoped at all about it, after his suicide,) this young man returned to Milton, and went quietly round to each creditor, paying him the first instalment of the money owing to him.",
 "In using the elastic stomach tube, some fluid should be introduced into the stomach before attempting to empty it, or a portion of the mucous membrane may be sucked into the aperture.",
 "'Margaret!' said mr Hale, as he returned from showing his guest downstairs; 'I could not help watching your face with some anxiety, when mr Thornton made his confession of having been a shop boy.",
]

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="vd_default")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import transformers; transformers.logging.set_verbosity_error()
    from voicestudio._qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel
    vd = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    gk = dict(do_sample=True, top_k=50, temperature=0.9)

    from spk_incon.metrics.presets import DatasetType, GenerationMethod, SynthesisConfig
    from spk_incon.metrics.strategies import create_strategy
    from spk_incon.datasets import create_dataset
    cfg = SynthesisConfig()
    ds = create_dataset(DatasetType.LIBRITTS, cfg.get_dataset_config("libritts"), root_dir="./data")
    class _D:
        def synthesize(self, **k): return True
    strat = create_strategy(GenerationMethod.METHOD2, cfg, ds, _D())
    idxs = strat.select_unique_speakers(cfg.generation.method2_ref_samples)
    syn_per = cfg.generation.method2_syn_per_ref

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    out = Path(f"results/{model_id}_{args.tag}")
    syn_base = out / "syn" / "libritts" / MODEL_CLASS_NAME / "method2"
    ref_base = out / "ref" / "libritts" / "method2"
    syn_base.mkdir(parents=True, exist_ok=True); ref_base.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    for ri, si in enumerate(idxs):
        transcript, audio_path, style, spk = ds.get_sample(si)
        persona = style or "A person speaks."
        try: shutil.copy(audio_path, ref_base / f"ref_{ri:03d}.wav")
        except Exception: pass
        set_dir = syn_base / f"set_{ri:03d}"; set_dir.mkdir(parents=True, exist_ok=True)
        texts = [transcript] + [COMP[k] for k in range(syn_per - 1)]
        seed(args.seed)
        wavs, sr = vd.generate_voice_design(text=texts, instruct=[persona]*len(texts), **gk)
        meta = {}
        for k, (t, w) in enumerate(zip(texts, wavs)):
            sf.write(set_dir / f"syn_{ri:03d}_{k:02d}.wav", np.asarray(w), sr)
            meta[f"syn_{ri:03d}_{k:02d}.wav"] = {"target_text": t, "speaker_id": spk, "reference_audio": str(audio_path)}
        json.dump(meta, open(set_dir / "metadata.json", "w"), indent=2)
        print(f"[{args.tag}] set {ri:03d} done (persona={persona[:40]!r})", flush=True)
    print(f"[{args.tag}] WORKER DONE", flush=True)

if __name__ == "__main__":
    main()
