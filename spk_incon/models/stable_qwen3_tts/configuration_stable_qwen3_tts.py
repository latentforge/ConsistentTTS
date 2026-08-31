"""Configuration class for Stable Qwen3-TTS."""

from transformers.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig


class StableQwen3TTSConfig(Qwen3TTSConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`StableQwen3TTSForConditionalGeneration`]. It extends [`Qwen3TTSConfig`] with the
    learnable vector-quantized query that conditions acoustic anchoring, and with the
    anchor schedule used by single pass generation.

    Args:
        num_query_tokens (`int`, *optional*, defaults to 32):
            Number of learnable query slots. Each slot is quantized to the nearest entry
            of the projected text embedding table, so the query stays on the manifold the
            talker was trained on.
        anchor_num_frames (`int`, *optional*, defaults to 32):
            Number of leading codec frames generated before the content text starts
            streaming. These frames establish the speaker identity and are dropped from
            the decoded waveform.
        query_token_ids (`list[int]`, *optional*):
            Text vocabulary ids the query slots quantize to. When set, the query is
            materialized from the frozen embedding table at load time and
            `num_query_tokens` is taken from its length.
        sub_talker_loss_weight (`float`, *optional*, defaults to 0.3):
            Weight of the residual code group cross entropy relative to the code group 0
            cross entropy in the training objective.
    """

    model_type = "stable_qwen3_tts"

    def __init__(
        self,
        num_query_tokens: int = 32,
        anchor_num_frames: int = 32,
        query_token_ids: list[int] | None = None,
        sub_talker_loss_weight: float = 0.3,
        **kwargs,
    ):
        # A stock Qwen3-TTS config.json carries its own model_type, which would otherwise
        # shadow this class attribute on the instance.
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        if query_token_ids is not None:
            query_token_ids = [int(i) for i in query_token_ids]
            num_query_tokens = len(query_token_ids)
        self.num_query_tokens = num_query_tokens
        self.anchor_num_frames = anchor_num_frames
        self.query_token_ids = query_token_ids
        self.sub_talker_loss_weight = sub_talker_loss_weight


__all__ = ["StableQwen3TTSConfig"]
