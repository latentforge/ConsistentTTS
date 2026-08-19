from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_chroma import ChromaBackboneConfig, ChromaConfig, ChromaDecoderConfig
from .modeling_chroma import (
    ChromaBackboneForCausalLM,
    ChromaDecoderForCausalLM,
    ChromaForConditionalGeneration,
    ChromaGenerationMixin,
    ChromaPreTrainedModel,
)
from .processing_chroma import ChromaAudioKwargs, ChromaProcessor, ChromaProcessorKwargs


AutoConfig.register("chroma", ChromaConfig)
AutoConfig.register("chroma_backbone", ChromaBackboneConfig)
AutoConfig.register("chroma_decoder", ChromaDecoderConfig)

AutoModel.register(ChromaConfig, ChromaForConditionalGeneration)
AutoModel.register(ChromaBackboneConfig, ChromaBackboneForCausalLM)
AutoModel.register(ChromaDecoderConfig, ChromaDecoderForCausalLM)

AutoProcessor.register(ChromaConfig, ChromaProcessor)


__all__ = [
    "ChromaBackboneConfig",
    "ChromaConfig",
    "ChromaDecoderConfig",
    "ChromaBackboneForCausalLM",
    "ChromaDecoderForCausalLM",
    "ChromaForConditionalGeneration",
    "ChromaGenerationMixin",
    "ChromaPreTrainedModel",
    "ChromaAudioKwargs",
    "ChromaProcessor",
    "ChromaProcessorKwargs",
]
