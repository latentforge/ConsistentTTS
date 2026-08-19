# Higgs-Audio v3

Higgs-Audio v3 pairs a plain [`Qwen3Model`] text backbone with a fused multi-codebook audio embedding and output
head, replacing v2's custom Llama-derived dual-FFN decoder. This implementation reuses `transformers-tts`'s
`Qwen3Model` for the backbone and its `HiggsAudioV2Embeddings`/`HiggsAudioV2PreTrainedModel` for the fused audio
embedding and shared `PreTrainedModel` plumbing, since those pieces are architecturally unchanged from v2. The
backbone itself (`HiggsAudioV3Model`, `HiggsAudioV3ForConditionalGeneration`) is written directly in this folder
rather than inherited from `HiggsAudioV2Model`, because v2's decoder layer bakes the dual-FFN audio/text routing
into every layer, which v3 does not have; `HiggsAudioV2Model` could not be reused as-is for the v3 backbone.
Waveform encode/decode reuses the transformers-native `HiggsAudioV2TokenizerModel`
(`bosonai/higgs-audio-v2-tokenizer`), which v3 checkpoints bundle unchanged.

Boson AI ships v3 as weights only (`bosonai/higgs-audio-v3-tts-4b`, also mirrored as
`bosonai/higgs-tts-3-4b`); the upstream `boson-ai/higgs-audio` repository no longer contains v3's inference code,
recommending Boson's hosted API or the third-party SGLang-Omni/MLX-Audio integrations instead. The prompt format
(placement of the reference-audio codes and any structuring tokens around the target text) is not publicly
documented; `HiggsAudioV3Processor` places `audio_input_ids` placeholders after the tokenized text, which decodes
correctly through `HiggsAudioV3Model.forward` but has not been validated against the real checkpoint's expected
layout end-to-end.

Original model and code: https://github.com/boson-ai/higgs-audio
