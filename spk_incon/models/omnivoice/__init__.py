from transformers import AutoConfig, AutoModel

from .configuration_omnivoice import OmniVoiceConfig
from .modeling_omnivoice import (
    GenerationTask,
    OmniVoiceForConditionalGeneration,
    OmniVoiceGenerationConfig,
    OmniVoiceModelOutput,
    OmniVoicePreTrainedModel,
    VoiceClonePrompt,
)
from .processing_omnivoice import OmniVoiceProcessor


AutoConfig.register("omnivoice", OmniVoiceConfig)
AutoModel.register(OmniVoiceConfig, OmniVoiceForConditionalGeneration)


__all__ = [
    "OmniVoiceConfig",
    "GenerationTask",
    "OmniVoiceForConditionalGeneration",
    "OmniVoiceGenerationConfig",
    "OmniVoiceModelOutput",
    "OmniVoicePreTrainedModel",
    "VoiceClonePrompt",
    "OmniVoiceProcessor",
]
