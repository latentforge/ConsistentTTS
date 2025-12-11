<div align="center">

# Chroma-4B: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/FlashLabs/Chroma-4B)
[![Technical Report](https://img.shields.io/badge/Technical-Report-red?logo=adobeacrobatreader)](https://arxiv.org/)
[![Playground](https://img.shields.io/badge/Chroma-Playground-9C276A)](https://chroma.flashintel.ai)
</div>

<div style="display: flex; justify-content: center; margin: 2rem 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <div style="display: flex; align-items: center; gap: 18px; font-size: 1.25rem;">
    <svg width="80" height="50" viewBox="0 0 48 28" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 16px rgba(168, 85, 247, 0.5));">
      <path
        d="M14 21C10.134 21 7 17.866 7 14C7 10.134 10.134 7 14 7C17.5 7 20 9 21.5 11L26.5 17C28 19 30.5 21 34 21C37.866 21 41 17.866 41 14C41 10.134 37.866 7 34 7C30.5 7 28 9 26.5 11L21.5 17C20 19 17.5 21 14 21Z"
        stroke="url(#apple_chroma_gradient)"
        stroke-width="3.5"
        stroke-linecap="round"
        stroke-linejoin="round"
      />
      <defs>
        <linearGradient id="apple_chroma_gradient" x1="7" y1="21" x2="41" y2="7" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#A855F7" />
          <stop offset="0.5" stop-color="#EC4899" />
          <stop offset="1" stop-color="#F97316" />
        </linearGradient>
      </defs>
    </svg>
    <div style="display: flex; align-items: baseline; gap: 6px;">
      <span style="background: linear-gradient(to right, #A855F7, #EC4899, #F97316); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.1rem; letter-spacing: -0.5px; white-space: nowrap;">
        Chroma
      </span>
      <span style="color: #94a3b8; font-size: 1.15rem; font-weight: 500; letter-spacing: 0.5px; white-space: nowrap;">
        by <span style="color: #e2e8f0;">Flash</span><span style="color: #60a5fa;">Labs</span>
      </span>
    </div>
  </div>
</div>

## Model Description

**Chroma-4B** is an advanced multimodal model developed by **[FlashLabs](https://flashlabs.ai)**. It is designed to understand and generate content across multiple modalities, including text and audio. As a virtual human model, Chroma possesses the ability to process auditory inputs and respond with both text and synthesized speech, enabling natural voice interactions.

- **Model Type:** Multimodal Causal Language Model
- **Developed by:** FlashLabs
- **Language(s):** English
- **License:** Apache-2.0
- **Model Architecture:** 
  - **Reasoner:** Based on Qwen2.5-Omni-3B
  - **Backbone:** Based on Llama3 (16 layers, 2048 hidden size)
  - **Decoder:** Based on Llama3 (4 layers, 1024 hidden size)
  - **Codec:** Mimi (24kHz sampling rate)

## Model Architecture

<img src="figures/architecture.png" alt="Model Architecture" width="800" />


## Capabilities

Chroma-4B is capable of:
- **Speech Understanding:** Processing user audio input directly.
- **Multimodal Generation:** Generating coherent text and speech responses simultaneously.
- **Voice Cloning:** utilizing reference audio prompts to guide speech generation style.

## Usage

### Installation

Ensure you have the necessary dependencies installed. You may need the latest versions of `transformers` and `torch`.

```bash
pip install transformers torch
```

### Loading the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "FlashLabs/Chroma-4B" # Or local path

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    device_map="auto"
)

# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

### Inference Example

Here is how to perform a simple conversation with audio input and audio output:

```python
import torch
from IPython.display import Audio



# Construct conversation history
system_prompt = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)
conversation = [[
    {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt}
        ],
    },
    {
        "role": "user",
        "content": [
            # Input audio file path
            {"type": "audio", "audio": "assets/make_taco.wav"}, 
        ],
    },
]]

# Provide reference audio/text for style or context
prompt_text = ["War and bloodshed throughout the world."]
prompt_audio = ["assets/reference_audio.wav"]

# Process inputs
inputs = processor(
    conversation,
    add_generation_prompt=True, 
    tokenize=False,
    prompt_audio=prompt_audio,
    prompt_text=prompt_text
)

# Move inputs to device
device = model.device
inputs = {k: v.to(device) for k, v in inputs.items()}

# 2. Generate
output = model.generate(
    **inputs, 
    max_new_tokens=100, 
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    use_cache=True
)

# 3. Decode Audio
# The model outputs raw tokens; we decode the audio part using the codec
audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values

# Save or play audio (e.g., in Jupyter)
Audio(audio_values[0].cpu().detach().numpy(), rate=24_000)
```
