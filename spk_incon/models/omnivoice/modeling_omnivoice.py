# Copyright 2026 Xiaomi Corp. and the LatentForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch OmniVoice model."""

import math
from dataclasses import dataclass, fields
from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface, PreTrainedModel
from transformers.utils import logging

from .configuration_omnivoice import OmniVoiceConfig


try:
    from torch.nn.attention.flex_attention import create_block_mask

    _flex_attention_available = True
except ImportError:
    _flex_attention_available = False


logger = logging.get_logger(__name__)

_AUTOCAST_FLEX_ATTENTION = "omnivoice_flex_attention"


def _autocast_flex_attention(module, query, key, value, *args, **kwargs):
    """flex_attention with the same autocast treatment SDPA already gets.

    Mixed-precision training keeps fp32 master weights, so the backbone's q_norm/k_norm (fp32 weight x
    bf16 activation) silently promote q/k, and through the fp32 RoPE constants v as well, back to fp32.
    SDPA is on autocast's cast list and is downcast at the kernel boundary; flex_attention is not, so with
    attn_implementation="flex_attention" all attention math runs in fp32. Casting here restores the
    treatment autocast applies to every other matmul; softmax accumulation inside the kernel is fp32
    either way.
    """
    if torch.is_autocast_enabled(query.device.type) and query.dtype == torch.float32:
        dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (t.to(dtype) for t in (query, key, value))
    return ALL_ATTENTION_FUNCTIONS["flex_attention"](module, query, key, value, *args, **kwargs)


@dataclass
class VoiceClonePrompt:
    r"""
    A reusable voice-clone conditioning prompt, built by `OmniVoiceProcessor.encode_reference`.

    Args:
        ref_audio_tokens (`torch.Tensor` of shape `(num_audio_codebook, T)`):
            Reference audio, encoded into discrete tokens.
        ref_text (`str`):
            Transcript of the reference audio.
        ref_rms (`float`):
            RMS amplitude of the reference waveform before normalization, used to restore the generated
            audio's volume.
    """

    ref_audio_tokens: torch.Tensor
    ref_text: str
    ref_rms: float

    _FORMAT_VERSION = 1

    def save(self, path: str) -> None:
        """Save this prompt to `path` for reuse in a later session.

        Args:
            path: Destination file path (e.g. `"my_voice.pt"`).
        """
        torch.save(
            {
                "format_version": self._FORMAT_VERSION,
                "ref_audio_tokens": self.ref_audio_tokens.detach().cpu(),
                "ref_text": self.ref_text,
                "ref_rms": float(self.ref_rms),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "VoiceClonePrompt":
        """Load a prompt saved with [`~VoiceClonePrompt.save`].

        Args:
            path: File path previously written by [`~VoiceClonePrompt.save`].
            map_location: Device to load the audio tokens onto.

        Returns:
            The restored [`VoiceClonePrompt`].

        Raises:
            `ValueError`: If the file's format version is not supported.
        """
        data = torch.load(path, map_location=map_location, weights_only=True)
        version = data.get("format_version")
        if version != cls._FORMAT_VERSION:
            raise ValueError(f"Unsupported VoiceClonePrompt format version: {version}")
        return cls(
            ref_audio_tokens=data["ref_audio_tokens"],
            ref_text=data["ref_text"],
            ref_rms=data["ref_rms"],
        )


@dataclass
class OmniVoiceGenerationConfig:
    r"""
    Controls the iterative parallel-decoding generation loop of [`OmniVoiceForConditionalGeneration.generate`].

    Args:
        num_step (`int`, *optional*, defaults to 32):
            Number of iterative unmasking steps.
        guidance_scale (`float`, *optional*, defaults to 2.0):
            Classifier-free guidance scale.
        t_shift (`float`, *optional*, defaults to 0.1):
            Time-step schedule shift; smaller values emphasize low-SNR (early) steps.
        layer_penalty_factor (`float`, *optional*, defaults to 5.0):
            Penalty subtracted from later codebook layers' confidence scores, encouraging earlier layers to
            unmask first.
        position_temperature (`float`, *optional*, defaults to 5.0):
            Gumbel temperature applied to position-selection scores (`0` disables sampling).
        class_temperature (`float`, *optional*, defaults to 0.0):
            Gumbel temperature applied to token sampling (`0` is greedy).
        denoise (`bool`, *optional*, defaults to `True`):
            Whether to prepend the `<|denoise|>` style token when a reference prompt is given.
        postprocess_output (`bool`, *optional*, defaults to `True`):
            Whether to remove long silences, fade in/out, and pad the decoded waveform.
        audio_chunk_duration (`float`, *optional*, defaults to 15.0):
            If `> 0`, split long text into chunks of this duration in seconds and generate chunk by chunk.
        audio_chunk_threshold (`float`, *optional*, defaults to 30.0):
            Only apply chunking if the estimated audio duration exceeds this threshold, in seconds.
        pad_duration (`float`, *optional*, defaults to 0.1):
            Silence padding duration per side, in seconds (`0` to disable).
        fade_duration (`float`, *optional*, defaults to 0.1):
            Fade-in/out curve duration, in seconds (`0` to disable).
    """

    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1

    @classmethod
    def from_dict(cls, kwargs_dict: dict) -> "OmniVoiceGenerationConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class GenerationTask:
    """Per-batch bookkeeping for one call to [`OmniVoiceForConditionalGeneration.generate`]."""

    batch_size: int
    texts: List[str]
    target_lens: List[int]
    langs: List[Optional[str]]
    instructs: List[Optional[str]]
    ref_texts: List[Optional[str]]
    ref_audio_tokens: List[Optional[torch.Tensor]]
    ref_rms: List[Optional[float]]
    speed: Optional[List[float]] = None

    def get_indices(self, config: OmniVoiceGenerationConfig, frame_rate: int):
        threshold = int(config.audio_chunk_threshold * frame_rate)
        short_idx = [i for i, l in enumerate(self.target_lens) if l <= threshold]
        long_idx = [i for i, l in enumerate(self.target_lens) if l > threshold]
        return short_idx, long_idx

    def slice_task(self, indices: List[int]):
        if not indices:
            return None
        return GenerationTask(
            batch_size=len(indices),
            texts=[self.texts[i] for i in indices],
            target_lens=[self.target_lens[i] for i in indices],
            langs=[self.langs[i] for i in indices],
            instructs=[self.instructs[i] for i in indices],
            ref_texts=[self.ref_texts[i] for i in indices],
            ref_audio_tokens=[self.ref_audio_tokens[i] for i in indices],
            ref_rms=[self.ref_rms[i] for i in indices],
            speed=[self.speed[i] for i in indices] if self.speed else None,
        )


@dataclass
class OmniVoiceModelOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Weighted mean per-codebook cross-entropy loss.
    logits (`torch.FloatTensor` of shape `(batch_size, num_audio_codebook, sequence_length, audio_vocab_size)`):
        Per-codebook prediction scores for each audio position.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class OmniVoicePreTrainedModel(PreTrainedModel):
    config_class = OmniVoiceConfig
    base_model_prefix = "model"
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class OmniVoiceForConditionalGeneration(OmniVoicePreTrainedModel):
    r"""
    OmniVoice wraps a causal language model backbone with a multi-codebook audio embedding table and audio head,
    and generates speech as a sequence of discrete audio tokens refined through iterative parallel unmasking
    (see [`~OmniVoiceForConditionalGeneration.generate`]).

    Args:
        config ([`OmniVoiceConfig`]):
            Model configuration.
        llm (`PreTrainedModel`, *optional*):
            An already-instantiated backbone to wrap. If not given, the backbone is built from
            `config.llm_config`.
    """

    def __init__(self, config: OmniVoiceConfig, llm: Optional[PreTrainedModel] = None):
        super().__init__(config)

        self.llm = llm if llm is not None else AutoModel.from_config(config.llm_config)

        if self.llm.config._attn_implementation == "flex_attention":
            AttentionInterface.register(_AUTOCAST_FLEX_ATTENTION, _autocast_flex_attention)
            self.llm.set_attn_implementation(_AUTOCAST_FLEX_ATTENTION)

        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size, config.llm_config.hidden_size
        )
        self.register_buffer(
            "codebook_layer_offsets", torch.arange(config.num_audio_codebook) * config.audio_vocab_size
        )
        self.audio_heads = nn.Linear(
            config.llm_config.hidden_size, config.num_audio_codebook * config.audio_vocab_size, bias=False
        )
        self.normalized_audio_codebook_weights = [
            w / sum(config.audio_codebook_weights) for w in config.audio_codebook_weights
        ]

        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def _prepare_embed_inputs(self, input_ids: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """Build input embeddings from `input_ids` of shape `(batch_size, num_audio_codebook, seq_length)`.

        Text positions embed only their first codebook row; audio positions sum the embeddings of all
        `num_audio_codebook` rows, each offset into its own codebook's slice of the audio embedding table.
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])

        shifted_ids = (input_ids * audio_mask.unsqueeze(1)) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> OmniVoiceModelOutput:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, num_audio_codebook, sequence_length)`):
                Interleaved style/text/audio token ids. Text positions repeat the same id across the
                codebook dimension; audio positions carry one id per codebook.
            audio_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Boolean mask, `True` at audio positions and `False` at text positions.
            labels (`torch.LongTensor` of shape `(batch_size, num_audio_codebook, sequence_length)`, *optional*):
                Target audio token ids for computing the masked audio-token cross-entropy loss. Positions
                that should not contribute to the loss are set to `-100`.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask or `flex_attention` block mask override. Built from `document_ids` when not
                given.
            document_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Per-token document id for sequence-packed training with `flex_attention`; tokens only attend
                within their own document.
            position_ids (`torch.LongTensor`, *optional*):
                Position ids forwarded to the backbone.

        Returns:
            [`OmniVoiceModelOutput`]
        """
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

        if attention_mask is None and document_ids is not None:
            if not _flex_attention_available:
                raise RuntimeError(
                    "flex_attention is not available in the current environment. If you do not need "
                    'flex_attention, set "attn_implementation": "sdpa" in your training config.'
                )
            attention_mask = create_block_mask(
                _get_packed_mask(document_ids[0].to(inputs_embeds.device)),
                B=None,
                H=None,
                Q_LEN=input_ids.size(-1),
                KV_LEN=input_ids.size(-1),
                _compile=True,
                device=inputs_embeds.device,
            )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, position_ids=position_ids
        )
        hidden_states = llm_outputs[0]

        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        audio_logits = logits_flat.view(
            batch_size, seq_len, self.config.num_audio_codebook, self.config.audio_vocab_size
        ).permute(0, 2, 1, 3)

        loss = None
        if labels is not None:
            per_token_loss = F.cross_entropy(
                audio_logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=-100
            )
            valid_mask = (labels != -100).float()
            layer_means = (per_token_loss * valid_mask).sum(dim=(0, 2)) / valid_mask.sum(dim=(0, 2)).clamp(min=1.0)
            weights = torch.tensor(self.normalized_audio_codebook_weights, device=audio_logits.device)
            loss = (layer_means * weights).sum()

        return OmniVoiceModelOutput(loss=loss, logits=audio_logits)

    @torch.inference_mode()
    def generate(
        self,
        processor,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        voice_clone_prompt: Union[VoiceClonePrompt, list[VoiceClonePrompt], None] = None,
        instruct: Union[str, list[str], None] = None,
        duration: Union[float, list, None] = None,
        speed: Union[float, list, None] = None,
        generation_config: Optional[OmniVoiceGenerationConfig] = None,
        normalize_text: bool = False,
        **kwargs,
    ) -> list:
        r"""
        Generate speech audio for `text`, in one of three modes.

        1. **Voice clone**: pass `voice_clone_prompt` (from `processor.encode_reference`) to clone the
           reference speaker's voice.
        2. **Voice design**: pass `instruct` describing the desired voice style; no reference audio needed.
        3. **Auto**: pass neither; the model picks a voice itself.

        Args:
            processor ([`OmniVoiceProcessor`]):
                The processor paired with this model, used to tokenize text, decode audio tokens, and
                estimate/normalize durations.
            text (`str` or `List[str]`):
                Target text (a single string, or a list for batched generation).
            language (`str` or `List[str]`, *optional*):
                Language name (e.g. `"English"`) or code (e.g. `"en"`). `None` for language-agnostic mode.
            ref_text (`str` or `List[str]`, *optional*):
                Reference transcript, when `voice_clone_prompt` is not given but voice cloning is desired
                through raw text (call `processor.encode_reference` directly to build the prompt).
            voice_clone_prompt ([`VoiceClonePrompt`] or `List[VoiceClonePrompt]`, *optional*):
                Reusable prompt from `processor.encode_reference` or [`VoiceClonePrompt.load`].
            instruct (`str` or `List[str]`, *optional*):
                Style instruction for voice design mode, validated by `processor.resolve_instruct`.
            duration (`float` or `List[float]`, *optional*):
                Fixed output duration in seconds. Overrides `speed` when both are given. `None` estimates
                duration from text.
            speed (`float` or `List[float]`, *optional*):
                Speaking speed factor; `> 1.0` for faster, `< 1.0` for slower.
            generation_config ([`OmniVoiceGenerationConfig`], *optional*):
                Explicit generation config, taking precedence over `**kwargs`.
            normalize_text (`bool`, *optional*, defaults to `False`):
                Whether to run `processor.normalize_text` on the target text before synthesis.
            **kwargs:
                Fields of [`OmniVoiceGenerationConfig`], used when `generation_config` is not given.

        Returns:
            `list[np.ndarray]`: One 1-D waveform per input text, at `processor.sampling_rate`.
        """
        gen_config = generation_config if generation_config is not None else OmniVoiceGenerationConfig.from_dict(kwargs)

        self.eval()

        full_task = self._preprocess_all(
            processor,
            text=text,
            language=language,
            ref_text=ref_text,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            speed=speed,
            duration=duration,
            normalize_text=normalize_text,
        )

        short_idx, long_idx = full_task.get_indices(gen_config, processor.audio_tokenizer.config.frame_rate)

        results = [None] * full_task.batch_size

        if short_idx:
            short_task = full_task.slice_task(short_idx)
            short_results = self._generate_iterative(processor, short_task, gen_config)
            for idx, res in zip(short_idx, short_results):
                results[idx] = res

        if long_idx:
            long_task = full_task.slice_task(long_idx)
            long_results = self._generate_chunked(processor, long_task, gen_config)
            for idx, res in zip(long_idx, long_results):
                results[idx] = res

        generated_audios = []
        for i in range(full_task.batch_size):
            generated_audios.append(
                processor.decode(
                    results[i],
                    ref_rms=full_task.ref_rms[i],
                    postprocess=gen_config.postprocess_output,
                    pad_duration=gen_config.pad_duration,
                    fade_duration=gen_config.fade_duration,
                )
            )

        return generated_audios

    def _generate_chunked(self, processor, task: GenerationTask, gen_config: OmniVoiceGenerationConfig):
        """Generate long audio by splitting text into chunks and batching across items per chunk index."""
        from .processing_omnivoice import chunk_text_punctuation

        all_chunks = []
        for i in range(task.batch_size):
            avg_tokens_per_char = task.target_lens[i] / len(task.texts[i])
            text_chunk_len = int(
                gen_config.audio_chunk_duration * processor.audio_tokenizer.config.frame_rate / avg_tokens_per_char
            )
            chunks = chunk_text_punctuation(text=task.texts[i], chunk_len=text_chunk_len, min_chunk_len=3)
            all_chunks.append(chunks)

        has_ref = [t is not None for t in task.ref_audio_tokens]
        assert all(has_ref) or not any(has_ref), (
            "Chunked inference requires all items to either have or not have ref_audio."
        )

        max_num_chunks = max(len(c) for c in all_chunks)
        chunk_results = [[] for _ in range(task.batch_size)]

        def _run_batch(indices, texts, ref_audios, ref_texts):
            speed_list = task.speed
            target_lens = [
                processor.estimate_target_length(
                    texts[j],
                    ref_texts[j],
                    ref_audios[j].size(-1) if ref_audios[j] is not None else None,
                    speed=speed_list[i] if speed_list else 1.0,
                )
                for j, i in enumerate(indices)
            ]
            sub_task = GenerationTask(
                batch_size=len(indices),
                texts=texts,
                target_lens=target_lens,
                langs=[task.langs[i] for i in indices],
                instructs=[task.instructs[i] for i in indices],
                ref_texts=ref_texts,
                ref_audio_tokens=ref_audios,
                ref_rms=[task.ref_rms[i] for i in indices],
                speed=[task.speed[i] for i in indices] if task.speed else None,
            )
            gen_tokens = self._generate_iterative(processor, sub_task, gen_config)
            for j, idx in enumerate(indices):
                chunk_results[idx].append(gen_tokens[j])

        if all(has_ref):
            for ci in range(max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[task.ref_audio_tokens[i] for i in indices],
                    ref_texts=[task.ref_texts[i] for i in indices],
                )
        else:
            indices_0 = [i for i in range(task.batch_size) if len(all_chunks[i]) > 0]
            _run_batch(
                indices_0,
                texts=[all_chunks[i][0] for i in indices_0],
                ref_audios=[None] * len(indices_0),
                ref_texts=[None] * len(indices_0),
            )
            first_chunk_map = {idx: chunk_results[idx][0] for idx in indices_0}

            for ci in range(1, max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[first_chunk_map[i] for i in indices],
                    ref_texts=[all_chunks[i][0] for i in indices],
                )

        return chunk_results

    def _preprocess_all(
        self,
        processor,
        text,
        language=None,
        ref_text=None,
        voice_clone_prompt=None,
        instruct=None,
        speed=None,
        duration=None,
        normalize_text=False,
    ) -> GenerationTask:
        text_list = [text] if isinstance(text, str) else list(text)
        batch_size = len(text_list)

        language_list = self._ensure_list(language, batch_size)
        language_list = [processor.resolve_language(lang) for lang in language_list]

        if normalize_text:
            text_list = [processor.normalize_text(t, lang) for t, lang in zip(text_list, language_list)]

        instruct_list = self._ensure_list(instruct, batch_size)
        for i, s in enumerate(instruct_list):
            if s is None:
                continue
            from .processing_omnivoice import _ZH_RE

            use_zh = bool(text_list[i] and _ZH_RE.search(text_list[i]))
            instruct_list[i] = processor.resolve_instruct(s, use_zh=use_zh)

        voice_clone_prompt_list = self._ensure_list(voice_clone_prompt, batch_size)
        if voice_clone_prompt_list[0] is not None:
            ref_text_list = [vc.ref_text for vc in voice_clone_prompt_list]
            ref_audio_tokens_list = [vc.ref_audio_tokens for vc in voice_clone_prompt_list]
            ref_rms_list = [vc.ref_rms for vc in voice_clone_prompt_list]
        else:
            ref_text_list = self._ensure_list(ref_text, batch_size) if ref_text is not None else [None] * batch_size
            ref_audio_tokens_list = [None] * batch_size
            ref_rms_list = [None] * batch_size

        if speed is not None:
            user_speed = [float(speed)] * batch_size if isinstance(speed, (int, float)) else list(speed)
        else:
            user_speed = None

        if duration is not None:
            durations = [float(duration)] * batch_size if isinstance(duration, (int, float)) else list(duration)
        else:
            durations = None

        num_target_tokens_list = []
        for i in range(batch_size):
            has_dur = durations is not None and durations[i] is not None
            item_speed = 1.0 if has_dur else (user_speed[i] if user_speed else 1.0)
            est = processor.estimate_target_length(
                text_list[i],
                ref_text_list[i],
                ref_audio_tokens_list[i].size(-1) if ref_audio_tokens_list[i] is not None else None,
                speed=item_speed,
            )
            num_target_tokens_list.append(est)

        speed_list = None
        if durations is not None:
            frame_rate = processor.audio_tokenizer.config.frame_rate
            speed_list = []
            for i in range(batch_size):
                if durations[i] is not None:
                    target_tokens = max(1, int(durations[i] * frame_rate))
                    est = num_target_tokens_list[i]
                    speed_list.append(est / target_tokens if target_tokens > 0 else 1.0)
                    num_target_tokens_list[i] = target_tokens
                else:
                    s = user_speed[i] if user_speed else None
                    speed_list.append(s if s is not None else 1.0)
        elif user_speed is not None:
            speed_list = [s if s is not None else 1.0 for s in user_speed]

        return GenerationTask(
            batch_size=batch_size,
            texts=text_list,
            target_lens=num_target_tokens_list,
            langs=language_list,
            instructs=instruct_list,
            ref_texts=ref_text_list,
            ref_audio_tokens=ref_audio_tokens_list,
            ref_rms=ref_rms_list,
            speed=speed_list,
        )

    def _ensure_list(self, x, batch_size: int):
        x_list = x if isinstance(x, list) else [x]
        if len(x_list) not in (1, batch_size):
            raise ValueError(f"should be either the number of the text or 1, but got {len(x_list)}")
        if len(x_list) == 1 and batch_size is not None:
            x_list = x_list * batch_size
        return x_list

    def _generate_iterative(
        self, processor, task: GenerationTask, gen_config: OmniVoiceGenerationConfig
    ) -> List[torch.Tensor]:
        """N-step iterative parallel unmasking decoding.

        Returns:
            List of generated audio token tensors of shape `(num_audio_codebook, T)`, one per input text.
        """
        B = task.batch_size

        inputs_list = [
            processor(
                task.texts[i],
                num_target_tokens=task.target_lens[i],
                ref_text=task.ref_texts[i],
                ref_audio_tokens=task.ref_audio_tokens[i],
                language=task.langs[i],
                instruct=task.instructs[i],
                denoise=gen_config.denoise,
            )
            for i in range(B)
        ]

        c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id

        batch_input_ids = torch.full(
            (2 * B, self.config.num_audio_codebook, max_c_len), pad_id, dtype=torch.long, device=self.device
        )
        batch_audio_mask = torch.zeros((2 * B, max_c_len), dtype=torch.bool, device=self.device)
        batch_attention_mask = torch.zeros((2 * B, 1, max_c_len, max_c_len), dtype=torch.bool, device=self.device)

        for i, inp in enumerate(inputs_list):
            c_len, u_len = c_lens[i], task.target_lens[i]

            batch_input_ids[i, :, :c_len] = inp["input_ids"][0].to(self.device)
            batch_audio_mask[i, :c_len] = inp["audio_mask"][0].to(self.device)
            batch_attention_mask[i, :, :c_len, :c_len] = True

            batch_input_ids[B + i, :, :u_len] = inp["input_ids"][0, :, -u_len:].to(self.device)
            batch_audio_mask[B + i, :u_len] = inp["audio_mask"][0, -u_len:].to(self.device)
            batch_attention_mask[B + i, :, :u_len, :u_len] = True
            if max_c_len > u_len:
                pad_diag = torch.arange(u_len, max_c_len, device=self.device)
                batch_attention_mask[B + i, :, pad_diag, pad_diag] = True

        tokens = torch.full(
            (B, self.config.num_audio_codebook, max(task.target_lens)),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        timesteps = _get_time_steps(
            t_start=0.0, t_end=1.0, num_step=gen_config.num_step, t_shift=gen_config.t_shift
        ).tolist()
        schedules = []
        for t_len in task.target_lens:
            total_mask = t_len * self.config.num_audio_codebook
            rem = total_mask
            sched = []
            for step in range(gen_config.num_step):
                num = (
                    rem
                    if step == gen_config.num_step - 1
                    else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        layer_ids = torch.arange(self.config.num_audio_codebook, device=self.device).view(1, -1, 1)

        for step in range(gen_config.num_step):
            batch_logits = self(
                input_ids=batch_input_ids, audio_mask=batch_audio_mask, attention_mask=batch_attention_mask
            ).logits.to(torch.float32)

            for i in range(B):
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len, t_len = c_lens[i], task.target_lens[i]

                c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :]
                u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :]

                pred_tokens, scores = self._predict_tokens_with_scoring(c_logits, u_logits, gen_config)

                scores = scores - (layer_ids * gen_config.layer_penalty_factor)

                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, gen_config.position_temperature)

                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(sample_tokens != self.config.audio_mask_id, -float("inf"))

                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                tokens[i : i + 1, :, :t_len] = sample_tokens
                batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = sample_tokens
                batch_input_ids[B + i : B + i + 1, :, :t_len] = sample_tokens

        return [tokens[i, :, : task.target_lens[i]] for i in range(B)]

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config: OmniVoiceGenerationConfig):
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs), dim=-1
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        log_probs[..., self.config.audio_mask_id] = -float("inf")

        if gen_config.class_temperature > 0.0:
            filtered_probs = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(filtered_probs, gen_config.class_temperature).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)

        confidence_scores = log_probs.max(dim=-1)[0]
        return pred_tokens, confidence_scores


def _get_packed_mask(document_ids):
    return partial(_mask_mod_packed, document_ids)


def _mask_mod_packed(document_ids, b, h, q_idx, kv_idx):
    return document_ids[q_idx] == document_ids[kv_idx]


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    return t_shift * timesteps / (1 + (t_shift - 1) * timesteps)


__all__ = [
    "GenerationTask",
    "OmniVoiceForConditionalGeneration",
    "OmniVoiceGenerationConfig",
    "OmniVoiceModelOutput",
    "OmniVoicePreTrainedModel",
    "VoiceClonePrompt",
]
