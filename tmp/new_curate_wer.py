"""Curate a HIGH-INTELLIGIBILITY training subset by ASR-WER: transcribe candidate
LibriTTS-P audios with Whisper-large-v3, compute WER vs the ground-truth transcript,
keep the LOWEST-WER (clearest) samples. Rationale: clear/intelligible training audio ->
cleaner anchor -> lower generation WER (the WER-relevant quality, unlike UTMOS)."""
import os, json, argparse, re
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cand", type=int, default=1200)
    ap.add_argument("--keep", type=int, default=600)
    ap.add_argument("--out", default="results/_summary/curated_wer_idx.json")
    args = ap.parse_args()
    import transformers; transformers.logging.set_verbosity_error()
    import jiwer, torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    dev = "cuda:0"
    wp = WhisperProcessor.from_pretrained("openai/whisper-small")
    wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", dtype=torch.float16).to(dev).eval()
    tf = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveWhiteSpace(replace_by_space=True),
                        jiwer.RemoveMultipleSpaces(), jiwer.ReduceToListOfListOfWords(word_delimiter=" ")])

    from spk_incon.datasets import LIBRITTS_P_Custom
    from spk_incon.datasets.libritts_p3 import download_libritts_p_metadata
    download_libritts_p_metadata(root="./data", annotator="df1")
    ds = LIBRITTS_P_Custom(root="./data", download=True, max_z_score=2, min_group_size=0)
    import random
    idxs = list(range(len(ds))); random.Random(0).shuffle(idxs)

    @torch.no_grad()
    def transcribe(wav, sr):
        w = torch.as_tensor(np.asarray(wav), dtype=torch.float32)
        if w.dim() > 1: w = w.mean(0)
        if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
        feats = wp(w.numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(dev).half()
        ids = wm.generate(feats, language="en", task="transcribe", max_new_tokens=200)
        return wp.batch_decode(ids, skip_special_tokens=True)[0].strip()

    scored = []
    n = 0
    for idx in idxs:
        if n >= args.n_cand: break
        it = ds[idx]; txt = (it.get("normalized_text") or "").strip()
        if len(txt) < 5: continue
        wav = np.asarray(it["waveform"]).squeeze()
        if wav.ndim > 1: wav = wav.mean(0)
        if len(wav) < 8000: continue
        try:
            hyp = transcribe(wav, it["sample_rate"])
            wer = float(jiwer.wer(txt, hyp, reference_transform=tf, hypothesis_transform=tf))
        except Exception:
            continue
        scored.append((idx, min(wer, 1.0)))
        n += 1
        if n % 100 == 0: print(f"  transcribed {n}, running WER mean={np.mean([s[1] for s in scored]):.3f}", flush=True)

    scored.sort(key=lambda x: x[1])          # ascending WER = clearest first
    top = scored[:args.keep]
    wers = [w for _, w in scored]
    json.dump({"top_idx": [i for i, _ in top], "kept": len(scored), "keep": len(top),
               "wer_all_mean": float(np.mean(wers)),
               "wer_kept_mean": float(np.mean([w for _, w in top])), "wer_kept_max": float(top[-1][1])},
              open(args.out, "w"), indent=2)
    print(f"ASR-WER all: mean={np.mean(wers):.3f} | kept top {len(top)} lowest-WER: "
          f"mean={np.mean([w for _,w in top]):.3f} max={top[-1][1]:.3f}", flush=True)
    print("SAVED", args.out, flush=True)

if __name__ == "__main__":
    main()
