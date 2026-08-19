"""Curate a HIGH-QUALITY training subset by UTMOS: dump candidate LibriTTS-P audios,
score with UTMOSv2, keep the top-K by MOS. Saves their dataset indices so the trainer
can learn the VQ query on clean/well-articulated speech (-> cleaner anchor -> lower WER)."""
import os, sys, json, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np, torch, soundfile as sf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cand", type=int, default=1400, help="candidates to score")
    ap.add_argument("--keep", type=int, default=800, help="top-K by UTMOS to keep")
    ap.add_argument("--out", default="results/_summary/curated_idx.json")
    args = ap.parse_args()

    from spk_incon.datasets import LIBRITTS_P_Custom
    from spk_incon.datasets.libritts_p3 import download_libritts_p_metadata
    download_libritts_p_metadata(root="./data", annotator="df1")
    ds = LIBRITTS_P_Custom(root="./data", download=True, max_z_score=2, min_group_size=0)

    import random
    idxs = list(range(len(ds))); random.Random(0).shuffle(idxs)
    tmp = "results/_curate_wavs"; os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    kept = []
    for idx in idxs[:args.n_cand]:
        it = ds[idx]
        w = np.asarray(it["waveform"]).squeeze()
        if w.ndim > 1: w = w.mean(0)
        if len(w) < 8000: continue                 # skip <0.5s
        sf.write(f"{tmp}/idx{idx}.wav", w, it["sample_rate"])
        kept.append(idx)
    print(f"dumped {len(kept)} candidate wavs", flush=True)

    from utmosv2 import create_model
    model = create_model(pretrained=True, config="fusion_stage3", fold=0, seed=42, device="cuda:0")
    model.eval()
    from contextlib import redirect_stdout, redirect_stderr
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        res = model.predict(input_dir=tmp)
    # res: list of {file_path, predicted_mos}
    scored = []
    for r in res:
        fn = os.path.basename(r["file_path"])
        idx = int(fn.replace("idx", "").replace(".wav", ""))
        scored.append((idx, float(r["predicted_mos"])))
    scored.sort(key=lambda x: -x[1])
    top = scored[:args.keep]
    top_idx = [i for i, _ in top]
    mos = [m for _, m in scored]
    json.dump({"top_idx": top_idx,
               "kept": len(scored), "keep": len(top_idx),
               "utmos_all_mean": float(np.mean(mos)), "utmos_all_min": float(min(mos)), "utmos_all_max": float(max(mos)),
               "utmos_kept_mean": float(np.mean([m for _, m in top])), "utmos_kept_min": float(top[-1][1])},
              open(args.out, "w"), indent=2)
    print(f"UTMOS all: mean={np.mean(mos):.3f} [{min(mos):.2f},{max(mos):.2f}]  "
          f"kept top {len(top_idx)}: mean={np.mean([m for _,m in top]):.3f} min={top[-1][1]:.3f}", flush=True)
    print("SAVED", args.out, flush=True)

if __name__ == "__main__":
    main()
