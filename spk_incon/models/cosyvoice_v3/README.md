# CosyVoice v3 (Fun-CosyVoice3)

Same Qwen2-backbone speech-token language model design as CosyVoice v2, with an extended
speech-token vocabulary that folds the start/task/fill/end-of-speech ids into the speech-token
embedding table itself (rather than a separate two-entry embedding) and a diffusion-transformer
(DiT) conditional-flow-matching decoder in place of the v1/v2 U-Net estimator, reimplemented
against the `transformers` model API.

`CosyVoiceV3LLM` subclasses `CosyVoiceV2LLM` from `voicestudio/models/cosyvoice_v2/`, matching the
original repository's own `CosyVoice3LM(Qwen2LM)` inheritance. `CosyVoiceV3FlowMatchingModel`
reuses the pre-lookahead layer, Conformer encoder, and length regulator from
`cosyvoice_v2`/`cosyvoice_v1`; only the estimator (`CosyVoiceV3DiT`) is new, since CosyVoice v3's
DiT estimator has no equivalent elsewhere in this codebase. The vocoder
(`CosyVoiceV1HiFTGenerator`) is imported unchanged from `cosyvoice_v1`.

Known gaps versus the original implementation: `CosyVoiceV3DiT` is a standard AdaLN
diffusion-transformer stack conditioned on the length-regulated encoder output, the speaker
embedding, and the timestep; it does not port the original `flow/DiT/dit.py`'s specific rotary
position embedding and joint mel/mu/speaker conditioning path in full detail. The original
repository also runs the flow front end directly on 80-dim mel-rate features rather than
through the shared Conformer encoder used here; this port keeps the v1/v2 encoder pipeline for
code reuse rather than matching that difference exactly. Bi-streaming text input, the
`<|endofprompt|>`-based instruct-token conditioning path, JIT/TensorRT export, and
vLLM-accelerated decoding are not ported. The discrete speech tokenizer
(`speech_tokenizer_v3.onnx`) and speaker encoder (`campplus.onnx`) are external ONNX artifacts
with no `transformers` equivalent; `CosyVoiceV3Processor` tokenizes text only.

Checkpoint: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`.

Original model and code: https://github.com/QwenAudio/CosyVoice
