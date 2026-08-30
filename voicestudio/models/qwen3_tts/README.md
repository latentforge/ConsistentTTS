# Qwen3-TTS

Qwen3-TTS pairs a Qwen3-based talker language model, which autoregressively predicts the first-layer speech token per step, with a separate code-predictor submodule that fills in the remaining residual codebooks, conditioned through a speaker encoder for preset/custom voices. Each checkpoint is trained for a single task (`base`, `custom_voice`, or `voice_design`); this folder's `Qwen3TTSProcessor` subclass adds `encode`/`encode_voice_design`/`encode_custom_voice` task-dispatch methods on top of the relayed `transformers` classes, raising `RuntimeError` on a task/checkpoint mismatch.

Original model and code: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)


## Usage

```python
from voicestudio.models.qwen3_tts import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id).to("cuda")
```

```python
inputs = processor.encode_voice_design(
    text="The sun rises in the east and sets in the west.",
    instruct="A calm, warm female voice.",
).to(model.device)

outputs = model.generate(**inputs)
audio_values, sr = processor.decode(outputs)
```
