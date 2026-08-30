# Dia

This folder only re-exports `DiaConfig`, `DiaDecoderConfig`, `DiaEncoderConfig`, `DiaFeatureExtractor`, `DiaForConditionalGeneration`, `DiaModel`, `DiaPreTrainedModel`, `DiaProcessor`, and `DiaTokenizer`; it does not vendor or reimplement the model.

Original model and code: [nari-labs/dia](https://github.com/nari-labs/dia)


## Usage

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "nari-labs/Dia-1.6B-0626"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
model.to("cuda")
```

```python
import soundfile as sf

inputs = processor(text="[S1] The sun rises in the east and sets in the west.", return_tensors="pt").to(model.device)

output_sequences = model.generate(**inputs)
waveform = processor.decode(output_sequences)
sf.write("output.wav", waveform.numpy(), processor.audio_tokenizer.config.sampling_rate)
```
