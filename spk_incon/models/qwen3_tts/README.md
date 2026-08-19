# Qwen3-TTS

This model is implemented directly in the [`transformers-tts`](https://github.com/latentforge/transformers-tts)
fork (`src/transformers/models/qwen3_tts/`, `qwen3_tts_tokenizer_multi_codebook/`,
`qwen3_tts_tokenizer_single_codebook/`). This folder only extends
`Qwen3TTSProcessor` with VoiceStudio's task-dispatching `encode` surface
(`encode`, `encode_voice_design`, `encode_custom_voice`); it does not vendor or
reimplement the model.

Original model and code: https://github.com/QwenLM/Qwen3-TTS
