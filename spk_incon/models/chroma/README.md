# Chroma

Chroma is a real-time spoken dialogue model with personalized voice cloning: a Qwen2.5-Omni-based reasoner
for text/audio understanding, a Llama-based backbone and decoder that jointly generate Mimi audio codec
frames, and the Mimi codec itself for reference-audio encoding and waveform decoding. This implementation
inherits `ChromaLlamaModel` from `LlamaModel`, `ChromaForConditionalGeneration`'s reasoner from
`Qwen2_5OmniThinkerForConditionalGeneration`, and its codec from `MimiModel`, all already present in
`transformers-tts`; only the backbone/decoder frame-prediction heads and the generation loop that
interleaves them are Chroma-specific and reimplemented here. `ChromaProcessor` subclasses
`Qwen2_5OmniProcessor` to additionally build the reference-audio voice-cloning prompt.

Original model and code: https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma
