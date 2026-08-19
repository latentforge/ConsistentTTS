# OmniVoice

`OmniVoiceForConditionalGeneration` wraps an arbitrary `transformers` causal language model backbone
(`config.llm_config`) with a multi-codebook audio embedding table and audio head, and generates speech as a
sequence of discrete audio tokens refined through iterative parallel unmasking. `OmniVoiceProcessor` combines
the text tokenizer and the `HiggsAudioV2TokenizerModel` audio tokenizer into a single processor that builds the
model's interleaved style/text/audio conditioning, estimates target durations, validates voice-design instruct
strings, and encodes/decodes reference and generated audio.

Original model and code: https://github.com/k2-fsa/OmniVoice

Checkpoint: https://huggingface.co/k2-fsa/OmniVoice
