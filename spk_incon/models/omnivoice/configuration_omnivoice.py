"""Configuration class for OmniVoice."""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig


class OmniVoiceConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`OmniVoiceForConditionalGeneration`]. It is
    used to instantiate an OmniVoice model according to the specified arguments, defining the model architecture.

    OmniVoice wraps an arbitrary causal language model (given by `llm_config`) with an audio codebook embedding
    table and a multi-codebook audio head, and trains/decodes speech as a sequence of masked audio tokens refined
    through iterative parallel unmasking.

    Args:
        audio_vocab_size (`int`, *optional*, defaults to 1025):
            Vocabulary size of one audio codebook, including the mask token.
        audio_mask_id (`int`, *optional*, defaults to 1024):
            Token id used to mark a still-masked audio position.
        num_audio_codebook (`int`, *optional*, defaults to 8):
            Number of parallel audio codebooks (RVQ layers) produced by the audio tokenizer.
        audio_codebook_weights (`list[float]`, *optional*):
            Per-codebook weights used to combine the per-layer cross-entropy losses. Defaults to
            `[8, 8, 6, 6, 4, 4, 2, 2]`, favoring the coarser (earlier) codebooks.
        llm_config (`PretrainedConfig` or `dict`, *optional*):
            Configuration of the wrapped causal language model backbone.
    """

    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: list[float] | None = None,
        llm_config: dict | PretrainedConfig | None = None,
        **kwargs,
    ):
        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        self.llm_config = llm_config

        super().__init__(**kwargs)
        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook
        if audio_codebook_weights is None:
            audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]
        self.audio_codebook_weights = audio_codebook_weights


__all__ = ["OmniVoiceConfig"]
