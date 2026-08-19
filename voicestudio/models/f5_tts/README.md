# F5-TTS

Diffusion-transformer (DiT) flow-matching text-to-speech model, reimplemented against the
`transformers` model API (`PretrainedConfig`, `PreTrainedModel`, `PreTrainedTokenizer`,
`ProcessorMixin`). F5-TTS's DiT/conditional-flow-matching architecture has no existing
lineage in `transformers-tts` (the library's TTS models are decoder-only LM backbones over
audio codec tokens; F5-TTS instead predicts a mel spectrogram directly with an ODE flow
solver), so this is a from-scratch port rather than an inherited/relayed model. `RMSNorm`
reuses `transformers.models.llama.modeling_llama.LlamaRMSNorm`.

F5-TTS predicts mel spectrograms only; rendering audio requires an external vocoder passed
into `F5TTSProcessor.decode` as a callable (the original checkpoints were trained against a
`vocos`-style front-end, but `decode` itself has no dependency on any specific vocoder
package: any callable mapping `(batch_size, mel_dim, sequence_length)` log-mel spectrograms
to waveforms works). No transformers-tts-native vocoder currently matches F5-TTS's mel
config (24kHz, 100 mel channels, power-1 centered STFT), so a suitable vocoder still has to
be supplied by the caller.

Original model and code: https://github.com/SWivid/F5-TTS
