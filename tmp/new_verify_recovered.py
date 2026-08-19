"""Verify the RECOVERED trainer's teacher-forcing forward (experiment_qwen_vqr_k32.ipynb cell 27)
by replicating its exact forward for a few steps. Correct => main_loss ~2.0, total ~5.3
(matching the original training log). Uses new_infer_lib for model handles."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch, random as _rnd
import transformers; transformers.logging.set_verbosity_error()
from transformers import AutoProcessor
from voicestudio.models.qwen3_tts import Qwen3TTSForConditionalGeneration
dev = torch.device("cuda:0")
MID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
m = Qwen3TTSForConditionalGeneration.from_pretrained(MID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
proc = AutoProcessor.from_pretrained(MID, device_map="cuda:0")
tok = proc.tokenizer
talker = m.talker; tcfg = m.config.talker_config
FIXED_CONTENT = "Hello, this is a fixed sentence used as the acoustic anchor for voice design retrieval."
_TALKER_GEN_KWARGS = dict(max_new_tokens=2048, min_new_tokens=2, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
    subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9,
    repetition_penalty=1.05, output_hidden_states=True, return_dict_in_generate=True)
@torch.no_grad()
def _gen_refs(vps, gk=None):
    clean = {k:v for k,v in (gk or {}).items() if k!="pad_token_id"}
    inputs = proc.encode_voice_design(text=[FIXED_CONTENT]*len(vps), instruct=vps)
    out = m.generate(**inputs, **clean)
    return [dict(ref_codes=c.detach().to(dev), voice_prompt=vp) for vp,c in zip(vps, out.audio_codes)]
def _common_talker_embeds(language="Auto"):
    cfg=m.config; long=torch.long
    tb,te,tp = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[cfg.tts_bos_token_id,cfg.tts_eos_token_id,cfg.tts_pad_token_id]],device=dev,dtype=long))).chunk(3,dim=1)
    prefill=[[cfg.talker_config.codec_nothink_id,cfg.talker_config.codec_think_bos_id,cfg.talker_config.codec_think_eos_id]]
    c0=talker.get_input_embeddings()(torch.tensor(prefill,device=dev,dtype=long))
    c1=talker.get_input_embeddings()(torch.tensor([[cfg.talker_config.codec_pad_id,cfg.talker_config.codec_bos_id]],device=dev,dtype=long))
    return tb,te,tp,torch.cat([c0,c1],dim=1)
class _L: pass
L=_L(); L._gen_refs=_gen_refs; L._common_talker_embeds=_common_talker_embeds; L._TALKER_GEN_KWARGS=_TALKER_GEN_KWARGS
L.model=m; L.processor=proc; L.tokenizer=tok; L.device=dev



# --- replicate recovered code exactly ---
for _p in m.parameters(): _p.requires_grad_(False)
text_emb_table = talker.get_text_embeddings().weight
LQ_K = 32
_g = torch.Generator(device="cpu").manual_seed(42)
init_ids = torch.randint(low=1000, high=text_emb_table.shape[0]-1000, size=(LQ_K,), generator=_g)
raw_query = torch.nn.Parameter(text_emb_table[init_ids].detach().clone().float().to(dev))
opt = torch.optim.AdamW([raw_query], lr=1e-2)

def quantize_ste(raw):
    with torch.no_grad():
        d = torch.cdist(raw.unsqueeze(0).float(), text_emb_table.unsqueeze(0).float()).squeeze(0)
        tok_ids = d.argmin(dim=1)
    q = text_emb_table[tok_ids].to(raw.dtype)
    return raw + (q - raw).detach(), tok_ids

# reproduce their training prompts source
from spk_incon.metrics.presets import DatasetType, SynthesisConfig
from spk_incon.datasets import create_dataset
cfg = SynthesisConfig()
ds = create_dataset(DatasetType.LIBRITTS, cfg.get_dataset_config("libritts"), root_dir="./data")
# their _train_prompts = test_dataset.dataset["style_prompt"] split by ';'
try:
    prompts = [s.split(";")[0] for s in ds.dataset["style_prompt"] if s]
except Exception as e:
    print("NOTE: could not read test_dataset.dataset['style_prompt'] ->", e)
    prompts = None
if not prompts:
    # fallback: use curated style prompts
    from spk_incon.datasets import LIBRITTS_P_Custom
    from spk_incon.datasets.libritts_p3 import download_libritts_p_metadata
    download_libritts_p_metadata(root="./data", annotator="df1")
    cur = LIBRITTS_P_Custom(root="./data", download=True, max_z_score=2, min_group_size=0)
    prompts = [cur[i]["combined_prompt"] for i in range(60)]
_rnd.Random(42).shuffle(prompts)
print("num train prompts:", len(prompts), flush=True)

def loss_fn(vp):
    with torch.no_grad():
        tref = L._gen_refs([vp], L._TALKER_GEN_KWARGS)[0]
        target_codes = tref["ref_codes"]
    T = target_codes.shape[0]
    q_ste, tok_ids = quantize_ste(raw_query)
    q_proj = talker.text_projection(q_ste.unsqueeze(0).to(talker.dtype))
    tts_bos, tts_eos, tts_pad, codec_input_emb = L._common_talker_embeds()
    ins = talker.text_projection(talker.get_text_embeddings()(
        tok(proc._build_instruct_text(vp), return_tensors="pt").input_ids.to(dev)))
    role = talker.text_projection(talker.get_text_embeddings()(
        tok(proc._build_assistant_text("x"), return_tensors="pt").input_ids.to(dev)[:, :3]))
    prefix = torch.cat([tts_pad.expand(-1, codec_input_emb.shape[1]-2, -1), tts_bos], dim=1) + codec_input_emb[:, :-1]
    codec_pad_e = talker.get_input_embeddings()(torch.tensor([[tcfg.codec_pad_id]], device=dev, dtype=torch.long))
    qreg = torch.cat([q_proj, tts_eos], dim=1) + codec_pad_e.expand(-1, LQ_K+1, -1)
    bospos = tts_pad + talker.get_input_embeddings()(torch.tensor([[tcfg.codec_bos_id]], device=dev, dtype=torch.long))
    prefix_embed = torch.cat([ins, role, prefix, qreg, bospos], dim=1)
    P = prefix_embed.shape[1]
    fe = []
    for i in range(tcfg.num_code_groups):
        if i == 0: fe.append(talker.get_input_embeddings()(target_codes[:, :1]))
        else: fe.append(talker.code_predictor.get_input_embeddings()[i-1](target_codes[:, i:i+1]))
    fe = torch.cat(fe, dim=1).sum(1).unsqueeze(0) + tts_pad.expand(-1, T, -1)
    full = torch.cat([prefix_embed, fe], dim=1)
    attn = torch.ones(full.shape[:2], device=dev, dtype=torch.long)
    labels = torch.full((1, P+T), -100, device=dev, dtype=torch.long); labels[0, P:P+T] = target_codes[:, 0]
    out = talker.forward(inputs_embeds=full, attention_mask=attn, labels=labels, output_hidden_states=True)
    main = out.loss
    last_hidden = out.hidden_states[0][-1]
    sub_hidden = last_hidden[:, P-1:P-1+T, :].reshape(T, -1)
    _, sub = talker.forward_sub_talker_finetune(codec_ids=target_codes, talker_hidden_states=sub_hidden)
    return main + 0.3*sub, float(main), float(sub)

for step in range(20):
    vp = prompts[step % len(prompts)]
    loss, main, sub = loss_fn(vp)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5 == 0 or step == 19:
        print(f"step {step:3d} total={float(loss):.4f} main={main:.4f} sub={sub:.4f}", flush=True)
print("\nEXPECTED (original log): main(ce0)~2.0, total~5.3. If main is far off => forward BROKEN.")
