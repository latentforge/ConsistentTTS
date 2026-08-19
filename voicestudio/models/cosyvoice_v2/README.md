# CosyVoice v2

Same three-stage design as CosyVoice v1, with the speech-token language model replaced by a
pretrained Qwen2 text backbone and the flow decoder made causal/streaming-capable, reimplemented
against the `transformers` model API.

`CosyVoiceV2LLM` reuses `transformers.models.qwen2.modeling_qwen2.Qwen2Model` directly as the
speech-token language model backbone (CosyVoice v2's own `Qwen2Encoder` is a thin wrapper around
`Qwen2ForCausalLM`). `CosyVoiceV2FlowMatchingModel`, `CosyVoiceV2CausalConditionalDecoder`, and
`CosyVoiceV2Config`/`CosyVoiceV2FlowConfig` subclass the CosyVoice v1 flow-matching classes in
`voicestudio/models/cosyvoice_v1/`, matching the original repository's own
`CausalMaskedDiffWithXvec(MaskedDiffWithXvec)` / `CausalConditionalDecoder(ConditionalDecoder)`
inheritance. The vocoder (`CosyVoiceV1HiFTGenerator`) is imported unchanged from
`cosyvoice_v1`, since the original repository's v1 and v2 checkpoints both instantiate the same
`HiFTGenerator` class (only the upsample rates in the config differ).

Known gaps versus the original implementation: the ported `CosyVoiceV2CausalConditionalDecoder`
computes over the full sequence rather than causal streaming chunks (the estimator's convolutions
are not swapped for left-padded causal variants); JIT/TensorRT export, vLLM-accelerated decoding,
and bi-streaming text input are not ported. The discrete speech tokenizer
(`speech_tokenizer_v2.onnx`) and speaker encoder (`campplus.onnx`) are external ONNX artifacts
with no `transformers` equivalent; `CosyVoiceV2Processor` tokenizes text only.

Checkpoint: `FunAudioLLM/CosyVoice2-0.5B`.

Original model and code: https://github.com/QwenAudio/CosyVoice
