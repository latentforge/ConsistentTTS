from transformers import AutoConfig, AutoProcessor
from transformers.models.auto.modeling_auto import AutoModel

from .configuration_cosyvoice_v2 import CosyVoiceV2Config, CosyVoiceV2FlowConfig, CosyVoiceV2LLMConfig
from .modeling_cosyvoice_v2 import (
    CosyVoiceV2CausalConditionalDecoder,
    CosyVoiceV2FlowMatchingModel,
    CosyVoiceV2ForConditionalGeneration,
    CosyVoiceV2LLM,
    CosyVoiceV2LLMOutput,
    CosyVoiceV2Model,
    CosyVoiceV2PreLookaheadLayer,
)
from .processing_cosyvoice_v2 import CosyVoiceV2Processor


AutoConfig.register("cosyvoice_v2", CosyVoiceV2Config)
AutoModel.register(CosyVoiceV2Config, CosyVoiceV2ForConditionalGeneration)
AutoProcessor.register(CosyVoiceV2Config, CosyVoiceV2Processor)


__all__ = [
    "CosyVoiceV2Config",
    "CosyVoiceV2LLMConfig",
    "CosyVoiceV2FlowConfig",
    "CosyVoiceV2ForConditionalGeneration",
    "CosyVoiceV2Model",
    "CosyVoiceV2LLM",
    "CosyVoiceV2LLMOutput",
    "CosyVoiceV2FlowMatchingModel",
    "CosyVoiceV2CausalConditionalDecoder",
    "CosyVoiceV2PreLookaheadLayer",
    "CosyVoiceV2Processor",
]
