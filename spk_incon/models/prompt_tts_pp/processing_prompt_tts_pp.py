"""Processor class for PromptTTS++."""

from typing import Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin


class PromptTTSppProcessor(ProcessorMixin):
    r"""
    Constructs a PromptTTS++ processor, combining a [`FastSpeech2ConformerTokenizer`] that phonemizes the text to
    synthesize with a BERT tokenizer that tokenizes the natural-language style/speaker prompt consumed by
    [`PromptTTSppPromptEncoder`].

    Args:
        tokenizer ([`FastSpeech2ConformerTokenizer`]):
            The phoneme tokenizer for the text to synthesize.
        prompt_tokenizer ([`BertTokenizer`] or [`BertTokenizerFast`]):
            The tokenizer for the natural-language style/speaker prompt.
    """

    attributes = ["tokenizer", "prompt_tokenizer"]
    tokenizer_class = "FastSpeech2ConformerTokenizer"
    prompt_tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, tokenizer=None, prompt_tokenizer=None, chat_template=None):
        super().__init__(tokenizer, prompt_tokenizer, chat_template=chat_template)

    def encode(
        self,
        text: Union[str, list[str]],
        style_prompt: Union[str, list[str]],
        return_tensors: str = "pt",
    ) -> BatchFeature:
        """
        Prepare inputs for [`PromptTTSppForConditionalGeneration`].

        Args:
            text (`str` or `List[str]`):
                The text to synthesize.
            style_prompt (`str` or `List[str]`):
                Natural-language description of the desired speaker identity and speaking style.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Tensor type to return.

        Returns:
            [`BatchFeature`]: Ready to be passed to [`PromptTTSppForConditionalGeneration`], with `input_ids`/
            `attention_mask` for `text` and `prompt_input_ids`/`prompt_attention_mask` for `style_prompt`.
        """
        text_inputs = self.tokenizer(text, return_tensors=return_tensors, padding=True)
        prompt_inputs = self.prompt_tokenizer(style_prompt, return_tensors=return_tensors, padding=True)

        return BatchFeature(
            {
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "prompt_input_ids": prompt_inputs["input_ids"],
                "prompt_attention_mask": prompt_inputs["attention_mask"],
            }
        )


__all__ = ["PromptTTSppProcessor"]
