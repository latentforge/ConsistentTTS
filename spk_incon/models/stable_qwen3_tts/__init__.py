from transformers import AutoConfig, AutoModel

from .configuration_stable_qwen3_tts import StableQwen3TTSConfig
from .modeling_stable_qwen3_tts import StableQwen3TTSForConditionalGeneration


AutoConfig.register("stable_qwen3_tts", StableQwen3TTSConfig)
AutoModel.register(StableQwen3TTSConfig, StableQwen3TTSForConditionalGeneration)


__all__ = [
    "StableQwen3TTSConfig",
    "StableQwen3TTSForConditionalGeneration",
]
