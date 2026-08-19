"""True Voice-Cloning method2 generation: Base model clones from the REAL dataset
reference audio (audio_path) per persona, generating syn0..syn9 for the same 10
personas/texts as the other method2 dirs. Lets us compute the same COS the standard
way. Writes results/Qwen/..._vc_default/. GPU 1 or 2 only."""
import os, json, argparse, shutil, random, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, soundfile as sf

MC = "Qwen3TTSForConditionalGeneration"
COMP = [
 "At least, no friend came forwards immediately, and mrs Thornton is not one, I fancy, to wait till tardy kindness comes to find her out.",
 "And the poor men around him-they were poor because they were vicious-out of the pale of his sympathies because they had not his iron nature, and the capabilities that it gives him for being rich.'",
 "The modes of treatment may be ranged under three heads: (one) To eliminate the poison; (two) to antagonize its action; (three) to avert the tendency to death.",
 "Visiting register offices, seeing all manner of unlikely people, and very few in the least likely, absorbed Margaret's time and thoughts for several days.",
 "But though she received caresses and fond words back again, in such profusion as would have gladdened her formerly, yet she felt that there was a secret withheld from her, and she believed it bore serious reference to her mother's health.",
 "And the poor men around him-they were poor because they were vicious-out of the pale of his sympathies because they had not his iron nature, and the capabilities that it gives him for being rich.",
 "mr Bell said they absolutely lived upon water porridge for years-how, he did not know; but long after the creditors had given up hope of any payment of old mr Thornton's debts, this young man returned to Milton, and went quietly round to each creditor.",
 "In using the elastic stomach tube, some fluid should be introduced into the stomach before attempting to empty it, or a portion of the mucous membrane may be sucked into the aperture.",
 "'Margaret!' said mr Hale, as he returned from showing his guest downstairs; 'I could not help watching your face with some anxiety, when mr Thornton made his confession of having been a shop boy.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--tag", default="vc_default")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import transformers; transformers.logging.set_verbosity_error()
    from voicestudio._qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    if getattr(base.model, "speech_tokenizer", None) is None:
        from voicestudio._qwen3_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        from transformers.utils import cached_file
        cf = cached_file("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "speech_tokenizer/preprocessor_config.json")
        base.model.load_speech_tokenizer(Qwen3TTSTokenizer.from_pretrained(os.path.dirname(cf)))
    bgk = dict(do_sample=True, top_k=50, temperature=0.9)

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

    out = Path(f"results/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign_{args.tag}")
    syn_base = out / "syn" / "libritts" / MC / "method2"; ref_base = out / "ref" / "libritts" / "method2"
    syn_base.mkdir(parents=True, exist_ok=True); ref_base.mkdir(parents=True, exist_ok=True)

    for ri, si in enumerate(idxs):
        transcript, audio_path, style, spk = ds.get_sample(si)
        try: shutil.copy(audio_path, ref_base / f"ref_{ri:03d}.wav")
        except Exception: pass
        set_dir = syn_base / f"set_{ri:03d}"; set_dir.mkdir(parents=True, exist_ok=True)
        rw, rsr = sf.read(str(audio_path)); rw = np.asarray(rw, dtype=np.float32)
        if rw.ndim > 1: rw = rw.mean(1)
        prompt = base.create_voice_clone_prompt(ref_audio=(rw, rsr), ref_text=transcript, x_vector_only_mode=False)
        texts = [transcript] + [COMP[k] for k in range(syn_per - 1)]
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        wavs, sr = base.generate_voice_clone(text=texts, voice_clone_prompt=prompt * len(texts), **bgk)
        meta = {}
        for k, (t, w) in enumerate(zip(texts, wavs)):
            sf.write(set_dir / f"syn_{ri:03d}_{k:02d}.wav", np.asarray(w), sr)
            meta[f"syn_{ri:03d}_{k:02d}.wav"] = {"target_text": t, "speaker_id": spk, "reference_audio": str(audio_path)}
        json.dump(meta, open(set_dir / "metadata.json", "w"), indent=2)
        print(f"[{args.tag}] set {ri:03d} done", flush=True)
    print(f"[{args.tag}] DONE", flush=True)

if __name__ == "__main__":
    main()
