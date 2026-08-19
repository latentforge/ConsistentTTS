"""Processor class for Chroma."""

import base64
import logging
from io import BytesIO
from typing import Optional, Union

import av
import librosa
import numpy as np
import torch
import torchaudio
from typing_extensions import Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.processing_utils import AudioKwargs, ProcessingKwargs


def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    return any(stream.type == "audio" for stream in container.streams)


def process_audio_info(conversations: Union[list[dict], list[list[dict]]], use_audio_in_video: bool):
    """Extracts and loads every audio (and, if requested, audio-in-video) reference found in a batch of
    chat-template conversations, resampling all of them to 16kHz for the reasoner's feature extractor."""
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" not in ele:
                        raise ValueError(f"Unknown audio {ele}")
                    path = ele["audio"]
                    if isinstance(path, np.ndarray):
                        if path.ndim > 1:
                            raise ValueError("Support only mono audio")
                        audios.append(path)
                    elif path.startswith("data:audio"):
                        _, base64_data = path.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        audios.append(librosa.load(BytesIO(data), sr=16000)[0])
                    elif path.startswith(("http://", "https://")):
                        audios.append(librosa.load(av.datasets.download(path), sr=16000)[0])
                    elif path.startswith("file://"):
                        audios.append(librosa.load(path[len("file://") :], sr=16000)[0])
                    else:
                        audios.append(librosa.load(path, sr=16000)[0])
                if use_audio_in_video and ele["type"] == "video":
                    if "video" not in ele:
                        raise ValueError(f"Unknown video {ele}")
                    path = ele["video"]
                    if not _check_if_video_has_audio(path):
                        raise ValueError("Video must has audio track when use_audio_in_video=True")
                    if path.startswith(("http://", "https://", "file://")):
                        path = path[len("file://") :] if path.startswith("file://") else path
                    audios.append(librosa.load(path, sr=16000)[0])
    return audios if len(audios) > 0 else None


class ChromaAudioKwargs(AudioKwargs, total=False):
    target_sample_rate: Optional[int]


class ChromaProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: ChromaAudioKwargs
    prompt_text: Optional[str]
    prompt_audio: Optional[Union[str, torch.Tensor]]
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "target_sample_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ChromaProcessor(Qwen2_5OmniProcessor):
    r"""
    Constructs a Chroma processor which wraps a [`Qwen2_5OmniProcessor`] to additionally build the
    reference-audio voice-cloning prompt Chroma's backbone consumes.

    [`ChromaProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`],
    [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`], plus resampling and batching a reference audio
    prompt and its transcript. See [`~ChromaProcessor.__call__`] for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor.
        video_processor ([`Qwen2VLVideoProcessor`], *optional*):
            The video processor.
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`str`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat
            template is used.
    """

    def __call__(
        self,
        conversations: list[list[dict]],
        prompt_audio: list[str],
        prompt_text: list[str],
        **kwargs: Unpack[ChromaProcessorKwargs],
    ) -> BatchFeature:
        """
        Args:
            conversations (`list[list[dict]]`):
                Batch of chat-template conversations, each a list of role/content messages. Any `"audio"`
                content is understood by the reasoner directly.
            prompt_audio (`list[str]`):
                Reference audio file path for each conversation in the batch, used to condition the
                generated speech's voice.
            prompt_text (`list[str]`):
                Transcript of `prompt_audio` for each conversation in the batch.

        Returns:
            [`BatchFeature`]: With the reasoner's `thinker_*`-prefixed inputs, the backbone's prompt text
            `input_ids`/`attention_mask`, and `input_values`/`input_values_cutoffs` holding the reference
            audio waveform.
        """
        if prompt_audio is None:
            raise ValueError("prompt_audio can not be empty")
        if prompt_text is None:
            raise ValueError("prompt_text can not be empty")

        batch_size = len(conversations)
        if len(prompt_audio) != batch_size:
            raise ValueError(f"prompt_audio length {len(prompt_audio)} != conversations length {batch_size}")
        if len(prompt_text) != batch_size:
            raise ValueError(f"prompt_text length {len(prompt_text)} != conversations length {batch_size}")

        output_kwargs = self._merge_kwargs(
            ChromaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if self.tokenizer is not None else {},
            **kwargs,
        )
        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        text, audios = self.apply_chat_template(conversations, **kwargs)
        thinker_inputs = super().__call__(
            text=text, audio=audios, use_audio_in_video=False, **text_kwargs, **common_kwargs
        )
        thinker_inputs = {f"thinker_{k}": v for k, v in thinker_inputs.items()}

        inputs = super().__call__(text=prompt_text, **text_kwargs, **common_kwargs)
        target_sample_rate = audio_kwargs.get("target_sample_rate", 24000)
        prompt_audio_wavs = [self.load_audio(audio, target_sample_rate) for audio in prompt_audio]
        prompt_audio_cutoffs = torch.tensor([len(audio) for audio in prompt_audio_wavs], dtype=torch.long)
        prompt_audio_tensor = torch.nn.utils.rnn.pad_sequence(prompt_audio_wavs, batch_first=True).unsqueeze(1)

        return BatchFeature(
            data={
                **thinker_inputs,
                **inputs,
                "input_values": prompt_audio_tensor,
                "input_values_cutoffs": prompt_audio_cutoffs,
            },
            tensor_type=common_kwargs.get("return_tensors"),
        )

    def load_audio(self, audio_path: str, target_sample_rate: int = 24000) -> torch.Tensor:
        """
        Loads an audio file, downmixes it to mono, and resamples it.

        Args:
            audio_path (`str`):
                Path to the audio file.
            target_sample_rate (`int`, *optional*, defaults to 24000):
                Sample rate to resample the loaded audio to.

        Returns:
            `torch.Tensor` of shape `(num_samples,)`.
        """
        try:
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            return torchaudio.functional.resample(
                audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=target_sample_rate
            )
        except Exception as e:
            logging.error(f"load audio file error: {e}")
            raise

    def apply_chat_template(self, conversations, chat_template=None, **kwargs) -> tuple[str, Optional[list]]:
        """
        Args:
            conversations (`list[dict]` or `list[list[dict]]`):
                One conversation, or a batch of conversations.
            chat_template (`str`, *optional*):
                The Jinja template to use for formatting. If not provided, `self.tokenizer`'s default chat
                template is used.

        Returns:
            `tuple(str, list)`: The formatted text and the audio arrays referenced by the conversation(s).
        """
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        audios = process_audio_info(conversations, use_audio_in_video=False)
        return self.tokenizer.apply_chat_template(conversations, chat_template, **kwargs), audios


__all__ = ["ChromaAudioKwargs", "ChromaProcessorKwargs", "ChromaProcessor"]
