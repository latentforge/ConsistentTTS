from transformers import AutoConfig, AutoProcessor
from transformers.models.auto.modeling_auto import AutoModel

from .configuration_cosyvoice_v3 import CosyVoiceV3Config, CosyVoiceV3FlowConfig, CosyVoiceV3LLMConfig
from .modeling_cosyvoice_v3 import (
    CosyVoiceV3DiT,
    CosyVoiceV3DiTBlock,
    CosyVoiceV3FlowMatchingModel,
    CosyVoiceV3ForConditionalGeneration,
    CosyVoiceV3LLM,
    CosyVoiceV3Model,
)
from .processing_cosyvoice_v3 import CosyVoiceV3Processor


AutoConfig.register("cosyvoice_v3", CosyVoiceV3Config)
AutoModel.register(CosyVoiceV3Config, CosyVoiceV3ForConditionalGeneration)
AutoProcessor.register(CosyVoiceV3Config, CosyVoiceV3Processor)


__all__ = [
    "CosyVoiceV3Config",
    "CosyVoiceV3LLMConfig",
    "CosyVoiceV3FlowConfig",
    "CosyVoiceV3ForConditionalGeneration",
    "CosyVoiceV3Model",
    "CosyVoiceV3LLM",
    "CosyVoiceV3FlowMatchingModel",
    "CosyVoiceV3DiT",
    "CosyVoiceV3DiTBlock",
    "CosyVoiceV3Processor",
]
