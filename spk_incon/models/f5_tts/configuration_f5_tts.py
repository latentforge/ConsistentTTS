"""Configuration class for F5-TTS."""

from transformers.configuration_utils import PretrainedConfig


class F5TTSConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`F5TTSModel`]. It is used to instantiate an
    F5-TTS model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a configuration close to that of the F5TTS_v1_Base checkpoint.

    Args:
        vocab_size (`int`, *optional*, defaults to 2545):
            Vocabulary size of the character/pinyin text tokenizer used by [`F5TTSTokenizer`].
        mel_dim (`int`, *optional*, defaults to 100):
            Number of channels of the mel spectrogram the model predicts and conditions on.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the DiT backbone.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of [`F5TTSDiTBlock`] layers in the backbone.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of each attention head.
        ff_mult (`int`, *optional*, defaults to 2):
            Hidden layer size multiplier for the feed-forward blocks, relative to `hidden_size`.
        text_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the text token embeddings before they are concatenated with the noised mel input.
        text_conv_layers (`int`, *optional*, defaults to 4):
            Number of ConvNeXt-V2 blocks applied to the text embedding stream.
        text_mask_padding (`bool`, *optional*, defaults to `True`):
            Whether padding positions of the text stream are masked out during text embedding.
        conv_layers_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability applied inside attention and feed-forward blocks.
        qk_norm (`str`, *optional*):
            Query/key normalization applied inside attention, `"rms_norm"` or `None`.
        pe_attn_head (`int`, *optional*):
            Number of attention heads that receive rotary position embeddings. Defaults to all heads when `None`.
        long_skip_connection (`bool`, *optional*, defaults to `False`):
            Whether to add a long skip connection from the input embedding to the final backbone output.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the rotary position embeddings.
        sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate, in Hz, of the audio the mel spectrogram front-end expects.
        n_fft (`int`, *optional*, defaults to 1024):
            FFT window size used to compute the mel spectrogram.
        hop_length (`int`, *optional*, defaults to 256):
            Hop length used to compute the mel spectrogram.
        win_length (`int`, *optional*, defaults to 1024):
            Window length used to compute the mel spectrogram.
        sigma (`float`, *optional*, defaults to 0.0):
            Standard deviation of the conditional flow used during training.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing weight matrices.
    """

    model_type = "f5_tts"

    def __init__(
        self,
        vocab_size: int = 2545,
        mel_dim: int = 100,
        hidden_size: int = 1024,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        ff_mult: int = 2,
        text_dim: int = 512,
        text_conv_layers: int = 4,
        text_mask_padding: bool = True,
        conv_layers_dropout: float = 0.1,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        long_skip_connection: bool = False,
        rope_theta: float = 10000.0,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sigma: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.ff_mult = ff_mult
        self.text_dim = text_dim
        self.text_conv_layers = text_conv_layers
        self.text_mask_padding = text_mask_padding
        self.conv_layers_dropout = conv_layers_dropout
        self.qk_norm = qk_norm
        self.pe_attn_head = pe_attn_head
        self.long_skip_connection = long_skip_connection
        self.rope_theta = rope_theta
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sigma = sigma
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


__all__ = ["F5TTSConfig"]
