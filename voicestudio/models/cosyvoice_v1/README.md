# CosyVoice v1

Three-stage TTS: a relative-position Conformer/Transformer speech-token language model
(`CosyVoiceV1LLM`), a conditional-flow-matching mel decoder (`CosyVoiceV1FlowMatchingModel`),
and a HiFTNet (neural-source-filter + ISTFT) vocoder (`CosyVoiceV1HiFTGenerator`), reimplemented
against the `transformers` model API (`PreTrainedConfig`, `PreTrainedModel`, `ProcessorMixin`).

CosyVoice v1's text encoder and speech-token language model are both relative-position Conformer
stacks (not a pretrained LLM backbone, unlike v2/v3), so `CosyVoiceV1TextEncoder` and
`CosyVoiceV1LLM` reuse
`transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoder`
(the closest existing relative-position Conformer lineage in `transformers-tts`) via a thin
`CosyVoiceV1RelPositionEncoder` subclass that also accepts a precomputed causal attention bias, so
the same encoder class serves both the bidirectional text encoder and the causal speech-token
decoder. The flow-matching U-Net estimator (`CosyVoiceV1ConditionalDecoder`) and the HiFTNet
vocoder have no existing `transformers` lineage and are ported directly (the estimator's
transformer blocks are simplified to a standard pre-norm self-attention + feed-forward block with
timestep-conditioned FiLM modulation, rather than the AdaLN-zero `BasicTransformerBlock` from
`diffusers`/`matcha-tts` the original uses).

Known gaps versus the original implementation: JIT/TensorRT export, vLLM-accelerated decoding,
streaming/chunked inference, and bi-streaming text input are not ported. The discrete speech
tokenizer (`speech_tokenizer_v1.onnx`) and speaker encoder (`campplus.onnx`) are external ONNX
artifacts with no `transformers` equivalent; `CosyVoiceV1Processor` tokenizes text only, and
expects already-extracted `prompt_speech_token` ids and speaker embeddings as input.

Checkpoint: `FunAudioLLM/CosyVoice-300M`.

Original model and code: https://github.com/QwenAudio/CosyVoice
