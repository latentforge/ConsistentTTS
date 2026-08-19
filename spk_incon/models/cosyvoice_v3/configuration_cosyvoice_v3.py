"""Configuration class for CosyVoice v3."""

from ..cosyvoice_v1.configuration_cosyvoice_v1 import CosyVoiceV1Config
from ..cosyvoice_v2.configuration_cosyvoice_v2 import CosyVoiceV2Config, CosyVoiceV2FlowConfig, CosyVoiceV2LLMConfig


class CosyVoiceV3LLMConfig(CosyVoiceV2LLMConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV3LLM`]. Identical to
    [`CosyVoiceV2LLMConfig`]; CosyVoice v3 reuses the same Qwen2 backbone and field set, only changing how the
    start/task/fill/end-of-speech ids are placed inside the speech-token embedding table (see
    [`CosyVoiceV3LLM`]).

    Args:
        `**kwargs`:
            Keyword arguments passed to [`CosyVoiceV2LLMConfig`].
    """

    model_type = "cosyvoice_v3_llm"
    base_config_key = "llm_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CosyVoiceV3FlowConfig(CosyVoiceV2FlowConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV3FlowMatchingModel`]. Extends
    [`CosyVoiceV2FlowConfig`] with the fields of the diffusion-transformer (DiT) estimator that CosyVoice v3 uses
    in place of the CosyVoice v1/v2 U-Net estimator.

    Args:
        dit_hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the DiT backbone.
        dit_num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of DiT blocks.
        dit_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in each DiT block.
        dit_head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of each attention head.
        dit_ff_mult (`int`, *optional*, defaults to 2):
            Hidden layer size multiplier for the DiT feed-forward blocks, relative to `dit_hidden_size`.
        `**kwargs`:
            Additional keyword arguments passed to [`CosyVoiceV2FlowConfig`].
    """

    model_type = "cosyvoice_v3_flow"

    def __init__(
        self,
        dit_hidden_size: int = 1024,
        dit_num_hidden_layers: int = 22,
        dit_num_attention_heads: int = 16,
        dit_head_dim: int = 64,
        dit_ff_mult: int = 2,
        **kwargs,
    ):
        self.dit_hidden_size = dit_hidden_size
        self.dit_num_hidden_layers = dit_num_hidden_layers
        self.dit_num_attention_heads = dit_num_attention_heads
        self.dit_head_dim = dit_head_dim
        self.dit_ff_mult = dit_ff_mult
        super().__init__(**kwargs)


class CosyVoiceV3Config(CosyVoiceV2Config):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV3ForConditionalGeneration`]. It is
    used to instantiate a CosyVoice v3 model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a configuration close to that of the
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint.

    Args:
        llm_config (`CosyVoiceV3LLMConfig`, *optional*):
            Configuration for the Qwen2-backbone speech-token language-model sub-model.
        flow_config (`CosyVoiceV3FlowConfig`, *optional*):
            Configuration for the DiT conditional-flow-matching decoder sub-model.
        hift_config (`CosyVoiceV1HiftConfig`, *optional*):
            Configuration for the NSF/ISTFT vocoder sub-model.
        sample_rate (`int`, *optional*, defaults to 24000):
            Output waveform sample rate, in Hz.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing weight matrices.
    """

    model_type = "cosyvoice_v3"
    sub_configs = {
        "llm_config": CosyVoiceV3LLMConfig,
        "flow_config": CosyVoiceV3FlowConfig,
        "hift_config": CosyVoiceV2Config.sub_configs["hift_config"],
    }

    def __init__(
        self,
        llm_config: dict | None = None,
        flow_config: dict | None = None,
        hift_config: dict | None = None,
        sample_rate: int = 24000,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.llm_config = CosyVoiceV3LLMConfig(**(llm_config or {}))
        self.flow_config = CosyVoiceV3FlowConfig(**(flow_config or {}))
        self.hift_config = self.sub_configs["hift_config"](**(hift_config or {}))
        self.sample_rate = sample_rate
        self.initializer_range = initializer_range
        super(CosyVoiceV1Config, self).__init__(**kwargs)


__all__ = ["CosyVoiceV3Config", "CosyVoiceV3LLMConfig", "CosyVoiceV3FlowConfig"]
