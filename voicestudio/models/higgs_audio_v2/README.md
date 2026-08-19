# Higgs-Audio

This model is implemented directly in the [`transformers-tts`](https://github.com/latentforge/transformers-tts)
fork (`src/transformers/models/higgs_audio_v2/`, `higgs_audio_v2_tokenizer/`). This folder only
re-exports `HiggsAudioV2Config`, `HiggsAudioV2TokenizerConfig`, `HiggsAudioV2Model`,
`HiggsAudioV2ForConditionalGeneration`, `HiggsAudioV2PreTrainedModel`, `HiggsAudioV2Processor`, and
`HiggsAudioV2TokenizerModel`; it does not vendor or reimplement the model.

Only Higgs-Audio v2 is present in transformers-tts. Boson AI has not published a v3 checkpoint or
architecture upstream, so there is nothing to migrate for a "v3" at this time.

Original model and code: https://github.com/boson-ai/higgs-audio
