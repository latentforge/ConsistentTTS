# Spark-TTS

Zero-shot text-to-speech built from a Qwen2.5 language model backbone (loaded through
`AutoModelForCausalLM`) that predicts BiCodec speech tokens, plus the BiCodec audio
tokenizer itself: a Vocos-style ConvNeXt encoder/decoder, DAC-derived decoder blocks,
a factorized VQ semantic quantizer, and an ECAPA-TDNN + Perceiver-Resampler + FSQ
speaker encoder for the global (timbre) tokens.

BiCodec has no equivalent already registered in `transformers`, so it is implemented
here directly; its waveform decoder reuses `Snake1d` from `transformers`' `dac` model
where the two architectures are identical. The LLM half is not reimplemented: it is
loaded through `AutoModelForCausalLM`/`AutoTokenizer` from the checkpoint's `LLM`
subfolder, the same way the original repository composes Qwen2.5 with BiCodec.

Original model and code: https://github.com/SparkAudio/Spark-TTS
