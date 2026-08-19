"""Configuration class for Chroma."""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from transformers.models.mimi.configuration_mimi import MimiConfig
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ChromaBackboneConfig(PretrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`ChromaBackboneForCausalLM`]. It is used to
    instantiate the Llama-based backbone that consumes the thinker's hidden states and text/audio prompt
    embeddings and autoregressively predicts the first-codebook audio token at each frame.

    Args:
        audio_num_codebooks (`int`, *optional*, defaults to 8):
            Number of codec codebooks per audio frame.
        vocab_size (`int`, *optional*, defaults to 2051):
            Vocabulary size of the codebook-0 head, i.e. the number of distinct codec codes per codebook plus the
            padding and end-of-sequence tokens.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length the backbone can process.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the backbone's hidden states.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads for grouped-query attention.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the MLP.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/value attentions.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Rotary position embedding configuration.
        head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of each attention head.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the MLP layers.
    """

    model_type = "chroma_backbone"

    def __init__(
        self,
        audio_num_codebooks: Optional[int] = 8,
        vocab_size: Optional[int] = 2051,
        max_position_embeddings: Optional[int] = 2048,
        hidden_size: Optional[int] = 2048,
        intermediate_size: Optional[int] = 8192,
        num_hidden_layers: Optional[int] = 16,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 8,
        hidden_act: Optional[str] = "silu",
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-5,
        use_cache: Optional[bool] = True,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        head_dim: Optional[int] = 64,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        mlp_bias: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_num_codebooks = audio_num_codebooks
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters or {}
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", 500000.0))
        self.standardize_rope_params()
        self.validate_rope()


class ChromaDecoderConfig(PretrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`ChromaDecoderForCausalLM`]. It is used to
    instantiate the small Llama-based decoder that autoregressively predicts codebooks 1 through
    `audio_num_codebooks - 1` of an audio frame, conditioned on the backbone's hidden state for that frame.

    Args:
        audio_num_codebooks (`int`, *optional*, defaults to 8):
            Number of codec codebooks per audio frame.
        audio_embedding_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the audio token embeddings shared with [`ChromaBackboneForCausalLM`], before the
            decoder's input projection.
        vocab_size (`int`, *optional*, defaults to 2051):
            Vocabulary size of each codebook head.
        max_position_embeddings (`int`, *optional*, defaults to 33):
            The maximum sequence length the decoder can process, i.e. one backbone hidden state plus one token
            per remaining codebook.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the decoder's hidden states.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads for grouped-query attention.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the MLP.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/value attentions.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Rotary position embedding configuration.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of each attention head.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the MLP layers.
    """

    model_type = "chroma_decoder"

    def __init__(
        self,
        audio_num_codebooks: Optional[int] = 8,
        audio_embedding_dim: Optional[int] = 2048,
        vocab_size: Optional[int] = 2051,
        max_position_embeddings: Optional[int] = 33,
        hidden_size: Optional[int] = 1024,
        intermediate_size: Optional[int] = 8192,
        num_hidden_layers: Optional[int] = 4,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 2,
        hidden_act: Optional[str] = "silu",
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-5,
        use_cache: Optional[bool] = True,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        head_dim: Optional[int] = 128,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        mlp_bias: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_embedding_dim = audio_embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim

        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters or {}
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", 500000.0))
        self.standardize_rope_params()
        self.validate_rope()


class ChromaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChromaForConditionalGeneration`]. It is
    used to instantiate a Chroma model according to the specified sub-configurations, defining the reasoner
    (thinker), backbone, decoder, and audio codec.

    Args:
        thinker_config (`dict` or [`Qwen2_5OmniThinkerConfig`], *optional*):
            Configuration for the Qwen2.5-Omni-based multimodal reasoner that consumes text and audio input and
            produces text tokens and hidden states conditioning speech generation.
        backbone_config (`dict` or [`ChromaBackboneConfig`], *optional*):
            Configuration for the backbone that predicts the first codebook of each audio frame.
        decoder_config (`dict` or [`ChromaDecoderConfig`], *optional*):
            Configuration for the decoder that predicts the remaining codebooks of each audio frame.
        codec_config (`dict` or [`MimiConfig`], *optional*):
            Configuration for the Mimi audio codec used to encode reference audio and decode generated audio
            codes into waveforms.
        codebook_pad_token_id (`int`, *optional*, defaults to 2050):
            Token id used to pad finished sequences in a batch during generation.
        codebook_eos_token_id (`int`, *optional*, defaults to 0):
            Token id that marks the end of the generated audio when present in every codebook of a frame.
        audio_num_codebooks (`int`, *optional*, defaults to 8):
            Number of codec codebooks per audio frame.
        text_start_token_id (`int`, *optional*, defaults to 151665):
            Token id marking the start of the text prompt segment in the backbone's input sequence.
        text_end_token_id (`int`, *optional*, defaults to 151666):
            Token id marking the end of the text prompt segment in the backbone's input sequence.
        im_end_token_id (`int`, *optional*, defaults to 151645):
            Token id that marks the end of a thinker generation turn.
        audio_frame_freq (`int`, *optional*, defaults to 1920):
            Number of audio samples per codec frame, used to align reference audio cutoffs with codec frames.
    """

    model_type = "chroma"
    sub_configs = {
        "thinker_config": Qwen2_5OmniThinkerConfig,
        "codec_config": MimiConfig,
        "backbone_config": ChromaBackboneConfig,
        "decoder_config": ChromaDecoderConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        backbone_config=None,
        decoder_config=None,
        codec_config=None,
        codebook_pad_token_id=2050,
        codebook_eos_token_id=0,
        audio_num_codebooks=8,
        text_start_token_id=151665,
        text_end_token_id=151666,
        im_end_token_id=151645,
        audio_frame_freq=1920,
        **kwargs,
    ):
        if isinstance(thinker_config, dict):
            self.thinker_config = Qwen2_5OmniThinkerConfig(**thinker_config)
        elif isinstance(thinker_config, Qwen2_5OmniThinkerConfig):
            self.thinker_config = thinker_config
        else:
            self.thinker_config = Qwen2_5OmniThinkerConfig()

        if isinstance(backbone_config, dict):
            self.backbone_config = ChromaBackboneConfig(**backbone_config)
        elif isinstance(backbone_config, ChromaBackboneConfig):
            self.backbone_config = backbone_config
        else:
            self.backbone_config = ChromaBackboneConfig(audio_num_codebooks=audio_num_codebooks)

        if isinstance(decoder_config, dict):
            self.decoder_config = ChromaDecoderConfig(**decoder_config)
        elif isinstance(decoder_config, ChromaDecoderConfig):
            self.decoder_config = decoder_config
        else:
            self.decoder_config = ChromaDecoderConfig(audio_num_codebooks=audio_num_codebooks)

        if isinstance(codec_config, dict):
            self.codec_config = MimiConfig(**codec_config)
        elif isinstance(codec_config, MimiConfig):
            self.codec_config = codec_config
        else:
            self.codec_config = MimiConfig(num_quantizers=audio_num_codebooks, frame_rate=12.5)

        self.audio_num_codebooks = audio_num_codebooks
        self.codebook_pad_token_id = codebook_pad_token_id
        self.codebook_eos_token_id = codebook_eos_token_id
        self.text_start_token_id = text_start_token_id
        self.text_end_token_id = text_end_token_id
        self.im_end_token_id = im_end_token_id
        self.audio_frame_freq = audio_frame_freq
        super().__init__(**kwargs)


__all__ = ["ChromaBackboneConfig", "ChromaDecoderConfig", "ChromaConfig"]
