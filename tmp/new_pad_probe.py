"""Padding-artifact probe: for a method2 result dir, compute the within-set Gen-Gen COS
three ways and aggregate over all sets:
  (a) per-file, no padding
  (b) zero-pad batch, NO wav_lens  <- exactly what official calculate_batch_optimized does
  (c) zero-pad batch, WITH wav_lens (correct usage)
Also reports the length disparity (max/min duration) per set, which drives the artifact.
Usage: CUDA_VISIBLE_DEVICES=1 python new_pad_probe.py <tag> [<tag> ...]"""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np, torch, soundfile as sf, torchaudio
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from speechbrain.inference.speaker import EncoderClassifier

DEV = "cuda:0"
MC = "Qwen3TTSForConditionalGeneration"
ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV})

def load(p):
    w, sr = sf.read(str(p)); w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1: w = w.mean(1)
    t = torch.tensor(w)
    if sr != 16000: t = torchaudio.functional.resample(t, sr, 16000)
    return t

def nrm(e): return torch.nn.functional.normalize(e, dim=-1)

@torch.no_grad()
def emb_perfile(wavs):
    out = []
    for w in wavs:
        out.append(nrm(ec.encode_batch(w.unsqueeze(0).to(DEV)).squeeze()).cpu())
    return torch.stack(out)

@torch.no_grad()
def emb_padded(wavs, use_lens):
    pad = pad_sequence(wavs, batch_first=True).to(DEV)
    if use_lens:
        maxlen = pad.shape[1]; wl = torch.tensor([len(w)/maxlen for w in wavs]).to(DEV)
        e = ec.encode_batch(pad, wav_lens=wl).squeeze(1)
    else:
        e = ec.encode_batch(pad).squeeze(1)
    return nrm(e).cpu()

def within(E):  # cos(syn0, syn_i) floored, averaged
    return [max(0.0, float(E[0] @ E[j])) for j in range(1, len(E))]

def probe(tag):
    sb = Path(f"results/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign_{tag}/syn/libritts/{MC}/method2")
    sets = sorted(d for d in sb.iterdir() if d.is_dir() and d.name.startswith("set_"))
    A, B, C, disp = [], [], [], []
    for sd in sets:
        wavs = [load(p) for p in sorted(sd.glob("syn_*.wav"))]
        if len(wavs) < 2: continue
        durs = [len(w)/16000 for w in wavs]
        disp.append(max(durs)/max(1e-6, min(durs)))
        A += within(emb_perfile(wavs))
        B += within(emb_padded(wavs, False))
        C += within(emb_padded(wavs, True))
    return np.mean(A), np.mean(B), np.mean(C), np.mean(disp), np.median(disp)

if __name__ == "__main__":
    print(f"{'tag':24s} {'(a)perfile':>10s} {'(b)pad-noWL':>11s} {'(c)pad-WL':>10s} {'disp_mean':>9s} {'disp_med':>8s}")
    for tag in sys.argv[1:]:
        a, b, c, dm, dmed = probe(tag)
        print(f"{tag:24s} {a:10.3f} {b:11.3f} {c:10.3f} {dm:9.1f} {dmed:8.1f}", flush=True)
