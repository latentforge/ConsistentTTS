"""FIXED VQ Global-Query trainer for k=32 (corrects the recovered notebook):
  (1) no skip-guard -> actually trains,  (2) STEPS=600, LR=0.02 (original recipe),
  (3) VQ quantization in the PROJECTED space (text_projection(vocab)) — matches the
      original checkpoint (verified earlier: 100% token_id match). Teacher-forcing forward
      verified (ce0~2.0). Saves ckpt/query_vq_k32.pt.
"""
import os, argparse, random
import numpy as np, torch, torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--subw", type=float, default=0.3)
    ap.add_argument("--out", default="ckpt/query_vq_k32.pt")
    ap.add_argument("--sched", default="cosine", choices=["cosine", "linear", "constant", "none"],
                    help="LR schedule (after optional warmup). none==constant.")
    ap.add_argument("--warmup", type=int, default=0, help="linear warmup steps (0=no warmup)")
    ap.add_argument("--vqspace", default="proj", choices=["proj", "raw"])
    ap.add_argument("--save_every", type=int, default=0, help="also save intermediate ckpts every N steps")
    ap.add_argument("--vocab_filter", default="none", choices=["none", "english"],
                    help="restrict VQ query tokens to clean ASCII-English tokens (coherent ref_text -> lower WER)")
    ap.add_argument("--init", default="random", choices=["random", "fixed_content"],
                    help="fixed_content: init query from FIXED_CONTENT tokens (coherent ref_text matching the anchor)")
    ap.add_argument("--curate_file", default="", help="json with top_idx: train only on those (high-UTMOS) samples")
    ap.add_argument("--reftextw", type=float, default=0.0,
                    help="ref_text-matching loss weight: pull query (projected) toward a coherent real "
                         "ref_text sequence (FIXED_CONTENT tiled to k). Grounds decoded ref_text -> lower WER.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import transformers; transformers.logging.set_verbosity_error()
    from transformers import AutoProcessor
    from voicestudio.models.qwen3_tts import Qwen3TTSForConditionalGeneration
    dev = torch.device("cuda:0")
    MID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    m = Qwen3TTSForConditionalGeneration.from_pretrained(
        MID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
    proc = AutoProcessor.from_pretrained(MID, device_map="cuda:0")
    tok = proc.tokenizer
    talker = m.talker; tcfg = m.config.talker_config; ncg = tcfg.num_code_groups
    for p in m.parameters(): p.requires_grad_(False)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    FIXED_CONTENT = "Hello, this is a fixed sentence used as the acoustic anchor for voice design retrieval."

    # VQ space: 'proj' = projected vocab (param lives projected); 'raw' = raw text-emb
    # (param lives raw, projected only for use) -- matches the recovered/original scheme.
    text_emb = talker.get_text_embeddings(); V = text_emb.weight.shape[0]
    RAW = text_emb.weight.detach().float()                       # (V, D_text)
    with torch.no_grad():
        PROJ = torch.cat([talker.text_projection(text_emb(torch.arange(i, min(i+8192, V), device=dev)).to(talker.dtype)).float()
                          for i in range(0, V, 8192)], 0)
    Dh = PROJ.shape[1]; print(f"PROJ {tuple(PROJ.shape)}  RAW {tuple(RAW.shape)}  vqspace={args.vqspace}", flush=True)
    # optional: restrict VQ candidates to clean ASCII-English word tokens
    allow_mask = None
    if args.vocab_filter == "english":
        import re
        allow = torch.zeros(V, dtype=torch.bool)
        pat = re.compile(r"^[ A-Za-z']+$")
        for v in range(V):
            s = tok.decode([v])
            if len(s) >= 2 and pat.match(s):
                allow[v] = True
        allow_mask = allow.to(dev)
        print(f"english vocab filter: {int(allow_mask.sum())} allowed tokens", flush=True)
    def _restrict(scores):
        if allow_mask is not None:
            scores = scores.masked_fill(~allow_mask.unsqueeze(0), float("-inf"))
        return scores
    def vq(param):  # param in projected space -> return (ids, projected_quant)
        if args.vqspace == "raw":
            pn = (RAW*RAW).sum(1)
            ids = _restrict(param @ RAW.t() - 0.5*pn.unsqueeze(0)).argmax(1)
            return ids, RAW[ids]
        pn = (PROJ*PROJ).sum(1)
        ids = _restrict(param @ PROJ.t() - 0.5*pn.unsqueeze(0)).argmax(1)
        return ids, PROJ[ids]

    def common():
        long = torch.long
        tb, te, tp = talker.text_projection(talker.get_text_embeddings()(torch.tensor(
            [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]], device=dev, dtype=long))).chunk(3, 1)
        prefill = [[tcfg.codec_nothink_id, tcfg.codec_think_bos_id, tcfg.codec_think_eos_id]]
        c0 = talker.get_input_embeddings()(torch.tensor(prefill, device=dev, dtype=long))
        c1 = talker.get_input_embeddings()(torch.tensor([[tcfg.codec_pad_id, tcfg.codec_bos_id]], device=dev, dtype=long))
        return tb, te, tp, torch.cat([c0, c1], 1)

    # cache N (persona, codec) pairs from the curated LibriTTS-P dataset
    from spk_incon.datasets import LIBRITTS_P_Custom
    from spk_incon.datasets.libritts_p3 import download_libritts_p_metadata
    download_libritts_p_metadata(root="./data", annotator="df1")
    ds = LIBRITTS_P_Custom(root="./data", download=True, max_z_score=2, min_group_size=0)
    @torch.no_grad()
    def to_codec(wav, sr):
        wav = np.asarray(wav).squeeze()
        al = proc._normalize_audio_inputs([wav], sr)
        fi = proc.feature_extractor(raw_audio=al, sampling_rate=int(proc.feature_extractor.sampling_rate),
                                    return_tensors="pt").to(dev).to(proc.audio_tokenizer.dtype)
        out = proc.audio_tokenizer.encode(fi["input_values"].squeeze(1), fi["padding_mask"].squeeze(1), return_dict=True)
        c = out.audio_codes[0]; return (c[0] if c.dim() == 3 else c)
    if args.curate_file and os.path.exists(args.curate_file):
        import json as _json
        idxs = _json.load(open(args.curate_file))["top_idx"]
        print(f"CURATED: using {len(idxs)} high-UTMOS indices from {args.curate_file}", flush=True)
    else:
        idxs = list(range(len(ds))); random.Random(0).shuffle(idxs)
    pairs = []
    for idx in idxs:
        if len(pairs) >= args.n: break
        it = ds[idx]; c = to_codec(it["waveform"], it["sample_rate"]).to(dev).long()
        if 4 <= c.shape[0] <= 400: pairs.append((it["combined_prompt"], c))
    print("cached pairs:", len(pairs), flush=True)

    def summed(codec):
        e = talker.get_input_embeddings()(codec[:, 0])
        for i in range(ncg-1): e = e + talker.code_predictor.get_input_embeddings()[i](codec[:, i+1])
        return e

    def loss_fn(persona, codec, q_ste):
        T = codec.shape[0]; long = torch.long
        tb, te, tp, cie = common()
        ins = talker.text_projection(talker.get_text_embeddings()(tok(proc._build_instruct_text(persona), return_tensors="pt").input_ids.to(dev)))
        role = talker.text_projection(talker.get_text_embeddings()(tok(proc._build_assistant_text("x"), return_tensors="pt").input_ids.to(dev)[:, :3]))
        prefix = torch.cat([tp.expand(-1, cie.shape[1]-2, -1), tb], 1) + cie[:, :-1]
        q_use = talker.text_projection(q_ste.unsqueeze(0).to(talker.dtype)) if args.vqspace == "raw" else q_ste.unsqueeze(0).to(talker.dtype)
        q = q_use if q_use.dim() == 3 else q_use.unsqueeze(0)
        cpe = talker.get_input_embeddings()(torch.tensor([[tcfg.codec_pad_id]], device=dev, dtype=long))
        qreg = torch.cat([q, te], 1) + cpe.expand(-1, args.k+1, -1)
        bospos = tp + talker.get_input_embeddings()(torch.tensor([[tcfg.codec_bos_id]], device=dev, dtype=long))
        prompt = torch.cat([ins, role, prefix, qreg, bospos], 1); P = prompt.shape[1]
        s = (summed(codec) + tp.squeeze(0)).to(talker.dtype)
        inp = torch.cat([prompt, s[:-1].unsqueeze(0)], 1)
        am = torch.ones(inp.shape[:2], device=dev, dtype=long)
        out = talker.forward(inputs_embeds=inp, attention_mask=am, use_cache=False, output_hidden_states=True)
        h = out.hidden_states[0][-1][0]; hh = h[P-1:P-1+T]
        ce0 = F.cross_entropy(talker.codec_head(hh).float(), codec[:, 0])
        _, sub = talker.forward_sub_talker_finetune(codec, hh)
        return ce0 + args.subw*sub, float(ce0), float(sub)

    torch.manual_seed(args.seed)
    if args.init == "fixed_content":
        fc_ids = tok(FIXED_CONTENT, return_tensors="pt").input_ids[0].tolist()
        fc_ids = [i for i in fc_ids if i < V]
        while len(fc_ids) < args.k: fc_ids = fc_ids + fc_ids
        init_ids = torch.tensor(fc_ids[:args.k], device=dev)
        print(f"init=fixed_content, decoded init ref_text={tok.decode(init_ids.tolist())!r}", flush=True)
    else:
        init_ids = torch.randint(0, V, (args.k,), device=dev)
    _init = (RAW[init_ids] if args.vqspace == "raw" else PROJ[init_ids])
    param = torch.nn.Parameter(_init.clone().float())
    # ref_text-matching target: coherent real ref_text (FIXED_CONTENT) tiled to k, in param's space
    ref_anchor = None
    if args.reftextw > 0:
        fc = tok(FIXED_CONTENT, return_tensors="pt").input_ids[0].tolist()
        fc = [i for i in fc if i < V]
        while len(fc) < args.k: fc = fc + fc
        ra_ids = torch.tensor(fc[:args.k], device=dev)
        ref_anchor = (RAW[ra_ids] if args.vqspace == "raw" else PROJ[ra_ids]).detach()
        print(f"reftextw={args.reftextw} target ref_text={tok.decode(ra_ids.tolist())!r}", flush=True)
    opt = torch.optim.AdamW([param], lr=args.lr)
    # Unified warmup + schedule via LambdaLR (multiplicative factor on base lr).
    # warmup=0 & sched=cosine reproduces CosineAnnealingLR(T_max=steps) exactly.
    _W = max(0, int(args.warmup)); _T = int(args.steps)
    def _lr_factor(step):
        if _W > 0 and step < _W:
            return (step + 1) / _W
        p = (step - _W) / max(1, _T - _W)          # progress in [0,1] after warmup
        p = min(max(p, 0.0), 1.0)
        if args.sched == "cosine": return 0.5 * (1.0 + np.cos(np.pi * p))
        if args.sched == "linear": return 1.0 - p
        return 1.0                                  # constant / none
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_factor)
    print(f"schedule: sched={args.sched} warmup={_W} steps={_T} lr={args.lr}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    def save_ckpt(path):
        with torch.no_grad():
            ids, qq = vq(param)
            ps = talker.text_projection(qq.unsqueeze(0).to(talker.dtype)).squeeze(0).float() if args.vqspace == "raw" else qq
        torch.save({"param": ps.detach().to(torch.bfloat16).cpu(), "k": args.k, "qtype": "vq", "D": Dh, "token_ids": ids.cpu().tolist()}, path)
        return ids
    avg = []
    for step in range(args.steps):
        persona, codec = pairs[step % len(pairs)]
        ids, qq = vq(param); q_ste = param + (qq - param).detach()
        loss, ce0, sub = loss_fn(persona, codec, q_ste)
        rt = 0.0
        if ref_anchor is not None:
            rt_loss = F.mse_loss(param, ref_anchor); rt = float(rt_loss); loss = loss + args.reftextw*rt_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None: sched.step()
        avg.append(float(loss)); avg = avg[-50:]
        if step % 50 == 0 or step == args.steps-1:
            print(f"  step {step:4d} loss={float(loss):.4f} ce0={ce0:.4f} sub={sub:.4f} rt={rt:.4f} avg50={np.mean(avg):.4f}", flush=True)
        if args.save_every and step > 0 and step % args.save_every == 0:
            p = args.out.replace(".pt", f"_step{step}.pt"); save_ckpt(p); print("  saved", p, flush=True)
    ids = save_ckpt(args.out)
    print("VQ decoded:", repr(tok.decode(ids.cpu().tolist()))[:120], flush=True)
    print("SAVED", args.out, flush=True)

if __name__ == "__main__":
    main()
