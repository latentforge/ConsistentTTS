# F5-TTS

Diffusion-transformer (DiT) flow-matching text-to-speech model, reimplemented against the
`transformers` model API (`PretrainedConfig`, `PreTrainedModel`, `PreTrainedTokenizer`,
`ProcessorMixin`). F5-TTS's DiT/conditional-flow-matching architecture has no existing
lineage in `transformers-tts` (the library's TTS models are decoder-only LM backbones over
audio codec tokens; F5-TTS instead predicts a mel spectrogram directly with an ODE flow
solver), so this is a from-scratch port rather than an inherited/relayed model. `RMSNorm`
reuses `transformers.models.llama.modeling_llama.LlamaRMSNorm`.

F5-TTS predicts mel spectrograms only; rendering audio requires an external vocoder (the
original checkpoints use `vocos`) passed into `F5TTSProcessor.decode`.

Original model and code: https://github.com/SWivid/F5-TTS
