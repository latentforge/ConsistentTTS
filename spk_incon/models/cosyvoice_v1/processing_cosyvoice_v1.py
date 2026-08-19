"""Processor class for CosyVoice v1."""

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin


class CosyVoiceV1Processor(ProcessorMixin):
    r"""
    Constructs a CosyVoice v1 processor which wraps a text tokenizer.

    [`CosyVoiceV1Processor`] tokenizes text into the ids consumed by [`CosyVoiceV1LLM`] and renders a generated
    mel spectrogram into a waveform with [`CosyVoiceV1HiFTGenerator`]. Discrete speech token extraction is done
    upstream by the original repository's `speech_tokenizer_v1.onnx` model, which has no `transformers`
    equivalent; callers that need voice cloning from a reference waveform must run that ONNX model themselves and
    pass the resulting token ids as `prompt_speech_token`.

    Args:
        tokenizer ([`PreTrainedTokenizerBase`]):
            The text tokenizer.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __call__(self, text: str | list[str], **kwargs) -> BatchFeature:
        """
        Args:
            text (`str` or `list[str]`):
                Input text to tokenize.

        Returns:
            [`BatchFeature`] with `text_token` and `text_token_len`.
        """
        kwargs.setdefault("return_tensors", "pt")
        encoded = self.tokenizer(text, **kwargs)
        lengths = encoded["attention_mask"].sum(dim=-1) if "attention_mask" in encoded else torch.tensor([encoded["input_ids"].shape[-1]])
        return BatchFeature(data={"text_token": encoded["input_ids"], "text_token_len": lengths})

    def decode(self, waveform: torch.Tensor, sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
        """
        Args:
            waveform (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
                Waveform produced by [`~CosyVoiceV1ForConditionalGeneration.generate`].
            sample_rate (`int`, *optional*):
                Overrides the model's configured output sample rate.

        Returns:
            `tuple(torch.FloatTensor, int)`: The waveform and its sample rate.
        """
        return waveform, sample_rate or 22050


__all__ = ["CosyVoiceV1Processor"]
