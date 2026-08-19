"""Processor class for F5-TTS."""

import torch
import torchaudio

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from .tokenization_f5_tts import F5TTSTokenizer


class F5TTSMelFeatureExtractor(torch.nn.Module):
    """
    Extracts the log mel spectrogram F5-TTS conditions on and predicts, matching the `vocos`-style front-end
    (power 1, centered STFT) the released F5-TTS checkpoints were trained with.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        mel = self.mel_spec.to(waveform.device)(waveform)
        return mel.clamp(min=1e-5).log().transpose(1, 2)


class F5TTSProcessor(ProcessorMixin):
    r"""
    Constructs an F5-TTS processor which wraps an [`F5TTSTokenizer`] and a mel spectrogram feature extractor into
    a single object.

    Args:
        tokenizer ([`F5TTSTokenizer`]):
            The text tokenizer.
        sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate, in Hz, the mel feature extractor expects.
        n_fft (`int`, *optional*, defaults to 1024):
            FFT window size used to compute the mel spectrogram.
        hop_length (`int`, *optional*, defaults to 256):
            Hop length used to compute the mel spectrogram.
        win_length (`int`, *optional*, defaults to 1024):
            Window length used to compute the mel spectrogram.
        n_mel_channels (`int`, *optional*, defaults to 100):
            Number of mel channels.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "F5TTSTokenizer"

    def __init__(
        self,
        tokenizer: F5TTSTokenizer,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 100,
        **kwargs,
    ):
        self.feature_extractor = F5TTSMelFeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
        )
        super().__init__(tokenizer, **kwargs)

    def encode(
        self,
        text: str,
        ref_audio: torch.Tensor,
        ref_text: str = "",
        speed: float = 1.0,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        """
        Prepare inputs for voice cloning: a reference waveform plus its transcript, and the text to synthesize
        in the reference speaker's voice.

        Args:
            text (`str`):
                The text to synthesize.
            ref_audio (`torch.FloatTensor` of shape `(1, num_samples)` or `(num_samples,)`):
                Reference waveform at `self.feature_extractor.sample_rate`.
            ref_text (`str`, *optional*, defaults to `""`):
                Transcript of `ref_audio`. An empty string relies on the model's `generate` speech-to-duration
                heuristic instead of a known reference length.
            speed (`float`, *optional*, defaults to 1.0):
                Multiplier controlling the generated duration relative to the reference-to-text length ratio.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Tensor type to return.

        Returns:
            [`BatchFeature`]: Ready to be passed to [`F5TTSForConditionalGeneration.generate`], with keys
            `cond_mel`, `text_ids`, and `duration`.
        """
        if ref_audio.ndim == 1:
            ref_audio = ref_audio.unsqueeze(0)
        cond_mel = self.feature_extractor(ref_audio)

        combined_text = ref_text + text
        text_ids = torch.tensor([self.tokenizer.encode(combined_text, add_special_tokens=False)], dtype=torch.long)

        ref_mel_len = cond_mel.shape[1]
        if len(ref_text) > 0:
            ref_text_len = len(self.tokenizer.tokenize(ref_text))
            gen_text_len = len(self.tokenizer.tokenize(text))
            duration = ref_mel_len + int(ref_mel_len / max(ref_text_len, 1) * gen_text_len / speed)
        else:
            duration = ref_mel_len + int(ref_mel_len / speed)

        data = {
            "cond_mel": cond_mel,
            "text_ids": text_ids,
            "duration": torch.tensor([duration], dtype=torch.long),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def decode(self, mel: torch.Tensor, vocoder=None) -> torch.Tensor:
        """
        Render a predicted mel spectrogram to a waveform.

        Args:
            mel (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
                Mel spectrogram produced by [`F5TTSForConditionalGeneration.generate`].
            vocoder (callable, *optional*):
                A vocoder mapping `(batch_size, mel_dim, sequence_length)` log-mel spectrograms to waveforms
                (e.g. a `vocos` `Vocos` instance). Required, since F5-TTS itself only predicts mel spectrograms.

        Returns:
            `torch.FloatTensor`: Waveform of shape `(batch_size, num_samples)`.

        Raises:
            `ValueError`: If `vocoder` is not given.
        """
        if vocoder is None:
            raise ValueError(
                "F5TTSProcessor.decode requires a vocoder (e.g. a `vocos` Vocos instance) to render mel "
                "spectrograms to waveforms; F5-TTS itself only predicts mel spectrograms."
            )
        return vocoder(mel.transpose(1, 2))


__all__ = ["F5TTSProcessor"]
