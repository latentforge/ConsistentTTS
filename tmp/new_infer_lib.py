"""Minimal inference lib for hybrid experiments: VoiceDesign Stage-1 anchor generation
(fixed-content OR learned VQ query) + decode anchor codec to a waveform. Used to feed a
distinctive anchor into the Base model's speaker-embedding channel."""
import os, random as _random
import numpy as _np
import torch

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
FIXED_CONTENT = "Hello, this is a fixed sentence used as the acoustic anchor for voice design retrieval."

model = processor = tokenizer = config = None
device = None
ANCHOR_MODE = "fixed"
_QUERY = None
_QK = None
REF_TEXT = FIXED_CONTENT

_TALKER_GEN_KWARGS = dict(
    max_new_tokens=2048, min_new_tokens=2,
    do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
    subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9,
    repetition_penalty=1.05, output_hidden_states=True, return_dict_in_generate=True,
)


def load_model(dev, anchor_mode="fixed", vq_ckpt="ckpt/query_vq_k32_recovered.pt", dtype=torch.bfloat16):
    global model, processor, tokenizer, config, device, ANCHOR_MODE, REF_TEXT, _QUERY, _QK
    import transformers; transformers.logging.set_verbosity_error()
    from transformers import AutoProcessor
    from voicestudio.models.qwen3_tts import Qwen3TTSForConditionalGeneration
    device = torch.device(dev) if not isinstance(dev, torch.device) else dev
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map=str(device), dtype=dtype, attn_implementation="flash_attention_2")
    config = model.config
    processor = AutoProcessor.from_pretrained(MODEL_ID, device_map=str(device))
    tokenizer = processor.tokenizer
    model.eval()
    ANCHOR_MODE = anchor_mode
    if anchor_mode == "vq":
        ck = torch.load(vq_ckpt, map_location=device)
        _QUERY = ck["param"].to(device); _QK = int(ck["k"])
        REF_TEXT = tokenizer.decode(ck["token_ids"]) if "token_ids" in ck else REF_TEXT
    else:
        REF_TEXT = FIXED_CONTENT
    return model


def _seed(s):
    _random.seed(s); _np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _common_talker_embeds(language="Auto"):
    cfg = model.config; talker = model.talker; long = torch.long
    tb, te, tp = talker.text_projection(talker.get_text_embeddings()(torch.tensor(
        [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]], device=device, dtype=long))).chunk(3, 1)
    prefill = [[cfg.talker_config.codec_nothink_id, cfg.talker_config.codec_think_bos_id, cfg.talker_config.codec_think_eos_id]]
    c0 = talker.get_input_embeddings()(torch.tensor(prefill, device=device, dtype=long))
    c1 = talker.get_input_embeddings()(torch.tensor([[cfg.talker_config.codec_pad_id, cfg.talker_config.codec_bos_id]], device=device, dtype=long))
    return tb, te, tp, torch.cat([c0, c1], 1)


@torch.no_grad()
def _gen_ref_fixed(voice_prompt, gk=None):
    clean = {k: v for k, v in (gk or {}).items() if k != "pad_token_id"}
    inputs = processor.encode_voice_design(text=[FIXED_CONTENT], instruct=[voice_prompt])
    out = model.generate(**inputs, **clean)
    return out.audio_codes[0].detach().to(device)


@torch.no_grad()
def _gen_ref_query(voice_prompt, gk=None, language="Auto"):
    cfg = model.config; tcfg = cfg.talker_config; talker = model.talker; long = torch.long
    tts_bos, tts_eos, tts_pad, cie = _common_talker_embeds(language)
    ins = talker.text_projection(talker.get_text_embeddings()(
        tokenizer(processor._build_instruct_text(voice_prompt), return_tensors="pt").input_ids.to(device)))
    role = talker.text_projection(talker.get_text_embeddings()(
        tokenizer(processor._build_assistant_text("x"), return_tensors="pt").input_ids.to(device)[:, :3]))
    prefix = torch.cat([tts_pad.expand(-1, cie.shape[1]-2, -1), tts_bos], 1) + cie[:, :-1]
    q = _QUERY.to(talker.dtype).unsqueeze(0)
    cpe = talker.get_input_embeddings()(torch.tensor([[tcfg.codec_pad_id]], device=device, dtype=long))
    qreg = torch.cat([q, tts_eos], 1) + cpe.expand(-1, _QK+1, -1)
    bospos = tts_pad + talker.get_input_embeddings()(torch.tensor([[tcfg.codec_bos_id]], device=device, dtype=long))
    inp = torch.cat([ins, role, prefix, qreg, bospos], 1)
    am = torch.ones(inp.shape[:2], device=device, dtype=long)
    tk = dict(_TALKER_GEN_KWARGS); tk["eos_token_id"] = tcfg.codec_eos_token_id
    tk["suppress_tokens"] = [i for i in range(tcfg.vocab_size-1024, tcfg.vocab_size) if i != tcfg.codec_eos_token_id]
    if gk: tk.update({k: v for k, v in gk.items() if k != "pad_token_id"})
    res = talker.generate(inputs_embeds=inp, attention_mask=am, trailing_text_hidden=tts_pad, tts_pad_embed=tts_pad, **tk)
    codes = torch.stack([h[-1] for h in res.hidden_states if h[-1] is not None], 1)
    fb = codes[:, :, 0]; stop = (fb == tcfg.codec_eos_token_id); has = stop.any(1); si = torch.argmax(stop.int(), 1)
    eff = torch.where(has, si, codes.shape[1])
    return codes[0, :int(eff[0])]


@torch.no_grad()
def make_anchor_audio(voice_prompt, gk=None, seed=42):
    """Seed-locked anchor codec -> waveform (np, sr). Uses fixed content or VQ query."""
    _seed(seed)
    ref_codes = _gen_ref_query(voice_prompt, gk) if ANCHOR_MODE == "vq" else _gen_ref_fixed(voice_prompt, gk)
    wavs, sr = processor.decode(dict(audio_codes=[ref_codes], ref_code_lengths=[0]))
    return _np.asarray(wavs[0]), sr
