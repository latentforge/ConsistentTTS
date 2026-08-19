# Parler-TTS

This model is vendored directly in this folder (`configuration_parler_tts.py`,
`modeling_parler_tts.py`, `processing_parler_tts.py`) rather than living in the
[`transformers-tts`](https://github.com/latentforge/transformers-tts) fork's
`src/transformers/models/`, since upstream `transformers` never merged Parler-TTS.
`weight_conversion.py` converts the original checkpoint format to this
implementation.

Original model and code: https://github.com/huggingface/parler-tts
