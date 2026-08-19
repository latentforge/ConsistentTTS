from transformers import AutoConfig, AutoModel

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .modeling_parler_tts import (
    ParlerTTSForCausalLM,
    ParlerTTSForConditionalGeneration,
    ParlerTTSLogitsProcessor,
    ParlerTTSModel,
    ParlerTTSPreTrainedModel,
)
from .processing_parler_tts import ParlerTTSProcessor


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoConfig.register("parler_tts_decoder", ParlerTTSDecoderConfig)
AutoModel.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM)
AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)


__all__ = [
    "ParlerTTSConfig",
    "ParlerTTSDecoderConfig",
    "ParlerTTSForCausalLM",
    "ParlerTTSForConditionalGeneration",
    "ParlerTTSLogitsProcessor",
    "ParlerTTSModel",
    "ParlerTTSPreTrainedModel",
    "ParlerTTSProcessor",
]
