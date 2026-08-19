# coding=utf-8
# Copyright 2025 FlashLabs. All rights reserved.
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
"""PyTorch Chroma model."""

from dataclasses import dataclass, fields
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache
from transformers.generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.mimi.modeling_mimi import MimiModel
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers.utils import ModelOutput, logging

from .configuration_chroma import ChromaBackboneConfig, ChromaConfig, ChromaDecoderConfig


if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

# Model kwargs that must survive from one autoregressive step to the next because they carry the
# reasoner's (thinker's) own generation state, which advances on a different schedule than the backbone.
PASSTHROUGH_KEYS = [
    "thinker_input_ids",
    "thinker_attention_mask",
    "thinker_cache_position",
    "thinker_past_key_values",
    "thinker_input_features",
    "thinker_feature_attention_mask",
    "thinker_eos",
    "thinker_hidden_states",
    "thinker_logits",
    "thinker_flag",
    "prefilled",
    "attention_mask",
]

# Inputs consumed only on the first generation step; carrying them forward would re-encode the prompt
# audio or re-inject the prompt features on every subsequent step.
ONE_TIME_KEYS = [
    "input_values",
    "thinker_input_features",
    "thinker_feature_attention_mask",
]


def multinomial_sample_one_no_sync(probs: torch.Tensor) -> torch.Tensor:
    """Multinomial sampling without a CUDA synchronization, via the Gumbel-max trick."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    """Samples a single token from `logits` using top-k sampling with temperature."""
    logits = logits / temperature

    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = F.log_softmax(scores_processed, dim=-1)
    probs = F.softmax(scores_processed, dim=-1)

    return multinomial_sample_one_no_sync(probs)


@dataclass
class ChromaOutputWithPast(ModelOutput):
    """
    Base class for Chroma outputs, carrying the backbone's step output alongside the reasoner
    (thinker) state that must be threaded through to the next autoregressive step.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            The backbone's codebook-0 language modeling loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            The backbone's hidden states at the output of each layer.
        logits (`torch.FloatTensor`, *optional*):
            The backbone's codebook-0 prediction scores.
        past_key_values (`tuple(torch.FloatTensor)`, *optional*):
            The backbone's cached key/value states.
        cache_position (`int`, *optional*):
            The backbone's current cache position.
        attention_mask (`torch.LongTensor`, *optional*):
            The attention mask covering the concatenated prompt and generated sequence so far.
        thinker_loss (`torch.FloatTensor`, *optional*):
            The reasoner's language modeling loss.
        thinker_logits (`torch.FloatTensor`, *optional*):
            The reasoner's next-token prediction scores.
        thinker_past_key_values (`tuple(torch.FloatTensor)`, *optional*):
            The reasoner's cached key/value states.
        thinker_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            The reasoner's hidden states at the output of each layer.
        thinker_attentions (`tuple(torch.FloatTensor)`, *optional*):
            The reasoner's attention weights.
        thinker_input_ids (`torch.LongTensor`, *optional*):
            The reasoner's next input ids, or `None` once the reasoner has produced its end-of-turn token.
        thinker_attention_mask (`torch.FloatTensor`, *optional*):
            The reasoner's attention mask.
        thinker_input_features (`torch.FloatTensor`, *optional*):
            The reasoner's input audio features, only present on the first step.
        thinker_feature_attention_mask (`torch.FloatTensor`, *optional*):
            The reasoner's input audio feature attention mask, only present on the first step.
        thinker_cache_position (`torch.FloatTensor`, *optional*):
            The reasoner's current cache position.
        thinker_flag (`bool`, *optional*):
            Whether the reasoner should generate its next token on the following step.
        thinker_eos (`torch.BoolTensor`, *optional*):
            Whether the reasoner has produced its end-of-turn token, per batch entry.
        backbone_loss (`torch.FloatTensor`, *optional*):
            Duplicate of `loss`, kept for symmetry with the decoder fields.
        backbone_logits (`torch.FloatTensor`, *optional*):
            Duplicate of `logits`, kept for symmetry with the decoder fields.
        backbone_past_key_values (`tuple(torch.FloatTensor)`, *optional*):
            Duplicate of `past_key_values`, kept for symmetry with the decoder fields.
        backbone_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Duplicate of `hidden_states`, kept for symmetry with the decoder fields.
        backbone_attentions (`tuple(torch.FloatTensor)`, *optional*):
            The backbone's attention weights.
        decoder_loss (`torch.FloatTensor`, *optional*):
            The per-frame codebook decoder's language modeling loss.
        decoder_logits (`torch.FloatTensor`, *optional*):
            The per-frame codebook decoder's prediction scores.
        decoder_past_key_values (`tuple(torch.FloatTensor)`, *optional*):
            The per-frame codebook decoder's cached key/value states.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            The per-frame codebook decoder's hidden states.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*):
            The per-frame codebook decoder's attention weights.
    """

    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[torch.FloatTensor, ...]] = None
    cache_position: Optional[int] = None
    attention_mask: Optional[torch.LongTensor] = None

    thinker_loss: Optional[torch.FloatTensor] = None
    thinker_logits: Optional[torch.FloatTensor] = None
    thinker_past_key_values: Optional[tuple[torch.FloatTensor, ...]] = None
    thinker_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    thinker_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    thinker_input_ids: Optional[torch.FloatTensor] = None
    thinker_attention_mask: Optional[torch.FloatTensor] = None
    thinker_input_features: Optional[torch.FloatTensor] = None
    thinker_feature_attention_mask: Optional[torch.FloatTensor] = None
    thinker_cache_position: Optional[torch.FloatTensor] = None
    thinker_flag: Optional[bool] = None
    thinker_eos: Optional[torch.BoolTensor] = None

    backbone_loss: Optional[torch.FloatTensor] = None
    backbone_logits: Optional[torch.FloatTensor] = None
    backbone_past_key_values: Optional[tuple[torch.FloatTensor, ...]] = None
    backbone_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    backbone_attentions: Optional[tuple[torch.FloatTensor, ...]] = None

    decoder_loss: Optional[torch.FloatTensor] = None
    decoder_logits: Optional[torch.FloatTensor] = None
    decoder_past_key_values: Optional[tuple[torch.FloatTensor, ...]] = None
    decoder_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class ChromaGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of [`ChromaForConditionalGeneration.generate`].

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length, audio_num_codebooks)`):
            The generated audio codes.
        scores (`tuple(torch.FloatTensor)`, *optional*):
            Processed prediction scores of the backbone's codebook-0 head at each generation step.
        logits (`tuple(torch.FloatTensor)`, *optional*):
            Unprocessed prediction scores of the backbone's codebook-0 head at each generation step.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*):
            The backbone's attention weights at each generation step.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            The backbone's hidden states at each generation step.
        past_key_values (`Cache`, *optional*):
            The backbone's cache, used to speed up decoding.
        audio (`list(torch.FloatTensor)`, *optional*):
            The generated audio waveforms, one per batch entry, returned when `output_audio=True`.
    """

    audio: Optional[list[torch.Tensor]] = None


class ChromaLlamaModel(LlamaModel):
    """
    A [`LlamaModel`] with the token embedding replaced by the identity, since Chroma's backbone and decoder
    both consume pre-computed input embeddings rather than token ids.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Identity()


class ChromaPreTrainedModel(PreTrainedModel):
    config_class = ChromaConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ChromaCodebookHead):
            module.weight.data.normal_(mean=0.0, std=std)


class ChromaAudioEmbedding(nn.Module):
    """Embeds per-codebook audio token ids into a shared hidden space, offsetting each codebook into its own
    slice of the embedding table so that the same token id in different codebooks maps to different vectors."""

    def __init__(self, audio_num_codebooks: int, audio_vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_audio_tokens = nn.Embedding(
            num_embeddings=audio_num_codebooks * audio_vocab_size,
            embedding_dim=hidden_size,
        )
        self.audio_vocab_size = audio_vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (`torch.Tensor` of shape `(..., num_codebooks)`):
                Per-codebook audio token ids.

        Returns:
            `torch.Tensor` of shape `(..., num_codebooks, hidden_size)`.
        """
        num_codebooks = input_ids.shape[-1]
        audio_frames = input_ids + (self.audio_vocab_size * torch.arange(num_codebooks, device=input_ids.device))
        return self.embed_audio_tokens(audio_frames.view(-1)).reshape(
            audio_frames.shape + (self.embed_audio_tokens.embedding_dim,)
        )


class ChromaBackboneForCausalLM(ChromaPreTrainedModel):
    """
    The Llama-based backbone that consumes text/audio prompt embeddings and reasoner hidden states, and
    autoregressively predicts the first codebook of each audio frame.
    """

    config_class = ChromaBackboneConfig
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, config: ChromaBackboneConfig):
        super().__init__(config)
        self.model = ChromaLlamaModel(LlamaConfig(**config.to_dict()))
        self.codebook0_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.hidden_size,
        )
        self.post_init()

    def emb_audio_frames(self, audio_frames: torch.Tensor, add_frame: bool = True) -> torch.Tensor:
        """
        Args:
            audio_frames (`torch.Tensor` of shape `(..., num_codebooks)`):
                Per-codebook audio token ids for one or more frames. Entries equal to `-100` (padding label)
                are replaced with codebook id 0 before lookup.
            add_frame (`bool`, *optional*, defaults to `True`):
                Whether to sum the per-codebook embeddings into a single per-frame embedding.

        Returns:
            `torch.Tensor` of shape `(..., hidden_size)` if `add_frame`, otherwise
            `(..., num_codebooks, hidden_size)`.
        """
        audio_frames = audio_frames.contiguous()
        audio_frames = audio_frames.masked_fill(audio_frames == -100, 0)
        audio_embeddings = self.audio_embedding(audio_frames)
        if add_frame:
            audio_embeddings = audio_embeddings.sum(dim=-2)
        return audio_embeddings

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        logits = logits.float()
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        logits = logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.to(logits.device)
        return F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)

    def forward(
        self,
        input_embeddings: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_embeddings (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of input embeddings, built from the text/audio prompt and the reasoner's hidden
                states.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Codebook-0 target ids for computing the language modeling loss.

        Returns:
            [`~transformers.modeling_outputs.CausalLMOutputWithPast`]
        """
        if input_embeddings is None:
            raise ValueError("input_embeddings is required")
        if input_embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError(f"input_embeddings must have {self.config.hidden_size} dimensions")

        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=input_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = self.codebook0_head(output.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.clone().detach())

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class ChromaCodebookHead(nn.Module):
    """A per-codebook linear projection head, batched over codebooks via `torch.bmm` instead of a
    `ModuleList` of `Linear` layers."""

    def __init__(self, audio_num_codebooks: int, audio_vocab_size: int, hidden_size: int):
        super().__init__()
        self.num_codebooks = audio_num_codebooks
        self.vocab_size = audio_vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_codebooks, self.hidden_size, self.vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (`torch.Tensor` of shape `(batch_size, num_codebooks, hidden_size)`):

        Returns:
            `torch.Tensor` of shape `(batch_size, num_codebooks, vocab_size)`.
        """
        codebook_num = x.shape[1]
        output = torch.bmm(x.transpose(0, 1), self.weight[:codebook_num, :, :])
        return output.transpose(0, 1)

    def get_logits(self, x: torch.Tensor, codebook_id: int) -> torch.Tensor:
        """
        Args:
            x (`torch.Tensor` of shape `(batch_size, hidden_size)`):
            codebook_id (`int`):
                Codebook index, between 1 and `num_codebooks` inclusive. Codebook 0 is predicted by the
                backbone, not this head.

        Returns:
            `torch.Tensor` of shape `(batch_size, vocab_size)`.
        """
        if codebook_id == 0 or codebook_id > self.num_codebooks:
            raise ValueError(f"codebook_id must be between 1 and {self.num_codebooks}, but got {codebook_id}")
        return torch.mm(x, self.weight[codebook_id - 1, :, :])


class ChromaDecoderForCausalLM(ChromaPreTrainedModel, GenerationMixin):
    """
    The small Llama-based decoder that autoregressively predicts codebooks 1 through
    `audio_num_codebooks - 1` of an audio frame, conditioned on the backbone's hidden state for that frame.
    """

    config_class = ChromaDecoderConfig
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, config: ChromaDecoderConfig):
        super().__init__(config)
        self.projection = nn.Linear(self.config.audio_embedding_dim, self.config.hidden_size, bias=False)
        self.model = ChromaLlamaModel(LlamaConfig(**config.to_dict()))
        self.codebook_head = ChromaCodebookHead(
            audio_num_codebooks=self.config.audio_num_codebooks - 1,
            audio_vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
        )
        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.audio_embedding_dim,
        )
        self.post_init()

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        vocab_size = logits.size(-1)
        logits_flat = logits.contiguous().view(-1, vocab_size)
        labels_flat = labels.contiguous().view(-1)
        return F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        During training, `inputs_embeds` is passed directly (already embedded by
        [`ChromaBackboneForCausalLM.emb_audio_frames`]). During incremental generation, `input_ids` and
        `backbone_last_hidden_state` are passed instead, following the standard `transformers` decoding
        contract.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, num_codebooks)`, *optional*):
                Per-codebook audio token ids for one frame.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
                Target codebook ids, `-100` where a codebook has no target.
            backbone_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
                The backbone's hidden state for the frame being decoded.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_codebooks, hidden_size)`, *optional*):
                Pre-computed frame embeddings, used during training.

        Returns:
            [`~transformers.modeling_outputs.CausalLMOutputWithPast`]
        """
        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids is required")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("inputs_embeds and input_ids cannot be used at the same time")

        loss = None

        if inputs_embeds is None:
            past_codebook_num = past_key_values.get_seq_length() - 1 if past_key_values is not None else 0
            if past_codebook_num > self.config.audio_num_codebooks - 1:
                raise ValueError(
                    f"past_codebook_num is greater than audio_num_codebooks - 1, "
                    f"{past_codebook_num} > {self.config.audio_num_codebooks - 1}"
                )
            offset = (
                torch.arange(input_ids.shape[-1], device=input_ids.device) + past_codebook_num
            ) * self.config.vocab_size
            audio_ids_embed = self.audio_embedding.embed_audio_tokens(input_ids + offset)
            inputs_embeds = (
                torch.cat([backbone_last_hidden_state.unsqueeze(1), audio_ids_embed], dim=1)
                if backbone_last_hidden_state is not None
                else audio_ids_embed
            )

        orig_shape = inputs_embeds.shape

        if inputs_embeds.dim() == 4:
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            labels = labels.reshape(-1, labels.shape[-1])

        has_eos = inputs_embeds.shape[1] == self.config.audio_num_codebooks + 1
        inputs_embeds = inputs_embeds[:, : self.config.audio_num_codebooks, :]

        inputs_embeds = self.projection(inputs_embeds)
        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if past_key_values is not None:
            logits = self.codebook_head.get_logits(
                output.last_hidden_state.squeeze(1), past_codebook_num + 1
            ).unsqueeze(1)
        else:
            logits = self.codebook_head(output.last_hidden_state[:, 1:, :])

        if labels is not None:
            if labels.shape[1] != self.config.audio_num_codebooks - 1:
                raise ValueError(
                    f"labels must have {self.config.audio_num_codebooks - 1} tokens, but got {labels.shape[1]}"
                )
            if logits.shape[1] != self.config.audio_num_codebooks - 1:
                raise ValueError(
                    f"logits must have {self.config.audio_num_codebooks - 1} tokens, but got {logits.shape[1]}"
                )
            loss = self.loss_fn(logits, labels.clone().detach())

        # Pads the codebook-1..N logits back out to align with the frame's full sequence length: a leading
        # slot for codebook 0 (predicted by the backbone, not this head) and, if present, a trailing slot
        # for the end-of-sequence frame.
        pad_left = 1 if backbone_last_hidden_state is not None or has_eos or input_ids is None else 0
        pad_right = 1 if has_eos else 0
        logits = F.pad(logits, (0, 0, pad_left, pad_right), value=0)
        logits = logits.reshape(*orig_shape[:-1], logits.shape[-1])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs
        )
        # backbone_last_hidden_state seeds only the first codebook of a frame; later codebooks condition
        # on the cache instead.
        if past_key_values is not None:
            model_inputs.pop("backbone_last_hidden_state")
        return model_inputs


class ChromaGenerationMixin(GenerationMixin):
    """Overrides the standard sampling loop to interleave the backbone (first codebook per frame) with a
    nested full `generate` call on [`ChromaDecoderForCausalLM`] (remaining codebooks per frame)."""

    def _get_stopping_criteria(self, *args, **kwargs) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)
        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if not isinstance(criterion, MaxLengthCriteria):
                logger.warning(
                    f"Chroma does not support {criterion.__class__.__name__} stopping criteria, it will be ignored."
                )
            else:
                kept_criteria.append(criterion)
        return kept_criteria

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        # `decoder_*` kwargs configure the nested per-frame codebook decoder's own generate() call.
        depth_decoder_kwargs = {k[len("decoder_") :]: v for k, v in kwargs.items() if k.startswith("decoder_")}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("decoder_")}

        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self.decoder.generation_config.update(**depth_decoder_kwargs)

        decoder_min_new_tokens = getattr(self.decoder.generation_config, "min_new_tokens") or (
            self.decoder.config.audio_num_codebooks - 1
        )
        decoder_max_new_tokens = getattr(self.decoder.generation_config, "max_new_tokens") or (
            self.decoder.config.audio_num_codebooks - 1
        )
        if {decoder_min_new_tokens, decoder_max_new_tokens} != {self.decoder.config.audio_num_codebooks - 1}:
            raise ValueError(
                f"decoder_generation_config's min_new_tokens ({decoder_min_new_tokens}) and max_new_tokens "
                f"({decoder_max_new_tokens}) must be equal to self.config.num_codebooks - 1 "
                f"({self.decoder.config.audio_num_codebooks - 1})"
            )
        elif self.decoder.generation_config.return_dict_in_generate:
            self.decoder.generation_config.return_dict_in_generate = False

        original_get_generation_mode = generation_config.get_generation_mode

        def patched_get_generation_mode(assistant_model=None):
            generation_mode = original_get_generation_mode(assistant_model)
            if generation_mode not in (GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE):
                raise ValueError(
                    f"Generation mode {generation_mode} is not supported for Chroma model. Please set "
                    "generation parameters to use greedy or sampling generation."
                )
            return generation_mode

        generation_config.get_generation_mode = patched_get_generation_mode
        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        pad_token_id = self.config.codebook_pad_token_id
        has_eos_stopping_criteria = generation_config._eos_token_tensor is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        top_k = generation_config.top_k
        temperature = generation_config.temperature

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        if input_ids.ndim == 2 and model_kwargs.get("inputs_embeds") is None:
            for criterion in stopping_criteria:
                if isinstance(criterion, MaxLengthCriteria):
                    criterion.max_length -= cur_len

        generated_frames = []

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": True})

            if is_prefill:
                backbone_outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                backbone_outputs = model_forward(**model_inputs, return_dict=True)

            next_token_logits = backbone_outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            backbone_last_hidden_state = backbone_outputs.hidden_states[-1][:, -1, :]

            model_kwargs = self._update_model_kwargs_for_generation(backbone_outputs, model_kwargs)

            if synced_gpus and this_peer_finished:
                continue

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (backbone_outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (backbone_outputs.hidden_states,)

            if do_sample:
                next_tokens = sample_topk(next_token_logits, top_k, temperature)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            frame_codes = self.decoder.generate(
                input_ids=next_tokens,
                backbone_last_hidden_state=backbone_last_hidden_state.clone(),
                max_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                min_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                do_sample=do_sample,
                use_cache=True,
                temperature=temperature,
                top_k=top_k,
            )
            if frame_codes.shape[-1] != self.config.decoder_config.audio_num_codebooks:
                raise ValueError(
                    f"Generated codebooks shape {frame_codes.shape[-1]} does not match expected "
                    f"audio_num_codebooks {self.config.decoder_config.audio_num_codebooks}"
                )
            next_tokens = frame_codes

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + pad_token_id * (
                    1 - unfinished_sequences.unsqueeze(-1)
                )

            if next_tokens.sum() != 0:
                generated_frames.append(next_tokens.unsqueeze(1))

            input_ids = next_tokens[:, None, :]

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # The eos stopping criterion assumes the eos token is shared across codebooks.
            unfinished_sequences = unfinished_sequences & ~(
                input_ids[:, -1, :-1] == self.config.codebook_eos_token_id
            ).all(-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(torch.cat(generated_frames, dim=1), scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del backbone_outputs
            del frame_codes

        if streamer is not None:
            streamer.end()

        sequences = torch.cat(generated_frames, dim=1) if len(generated_frames) > 0 else input_ids
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return sequences

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        output_audio: Optional[bool] = False,
        bos_token_id: Optional[int] = 0,
        **kwargs: Any,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Overrides [`~transformers.generation.GenerationMixin.generate`] to match the specifics of the Chroma
        model:

        1. Infer the backbone model to sample the first codebook token of a frame.
        2. Call `generate` on the depth decoder with the first codebook token as `input_ids` to sample the
           rest of the frame's codebook tokens.
        3. Use the generated frame as `input_ids` to sample the next frame's first codebook token with the
           backbone model.
        4. Repeat until a stopping criterion is met.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the backbone model.
            input_values (`torch.Tensor` of shape `(batch_size, channels, max_concatenated_audio_length)`, *optional*):
                The batched reference audio input values, encoded into codebook tokens using the codec model
                and merged with the text input ids in `input_ids`.
            input_values_cutoffs (`torch.Tensor` of shape `(batch_size, max_num_audio)`, *optional*):
                The end position of each audio segment within a batch entry's concatenated audio input,
                padded with `-1` for batch entries with fewer segments than the maximum.
            generation_config ([`~transformers.generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria.
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length.
            streamer (`BaseStreamer`, *optional*):
                Streamer object used to stream the generated audio codes.
            output_audio (`bool`, *optional*, defaults to `False`):
                Whether to decode and return the generated audio.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs.
                Depth decoder specific kwargs must be prefixed with `decoder_`.

        Returns:
            [`ChromaGenerateOutput`] or `torch.LongTensor` or `list[torch.FloatTensor]`: A
            [`ChromaGenerateOutput`] (if `return_dict_in_generate=True`), a `torch.LongTensor` of generated
            audio codes when `output_audio=False`, or a `list[torch.FloatTensor]` of audio waveforms
            otherwise.
        """
        generate_output = super().generate(
            input_ids=input_ids,
            input_values=input_values,
            input_values_cutoffs=input_values_cutoffs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            streamer=streamer,
            bos_token_id=bos_token_id,
            **kwargs,
        )
        generate_returned_dict = not isinstance(generate_output, torch.Tensor)
        audio = None
        if output_audio:
            generated_audio_codes = generate_output.sequences if generate_returned_dict else generate_output
            audio = []
            with torch.no_grad():
                for audio_codes_batch in generated_audio_codes:
                    eos_idxs = (audio_codes_batch == self.config.codebook_eos_token_id).all(dim=-1).nonzero()
                    cutoff_idx = eos_idxs.min() if eos_idxs.numel() != 0 else audio_codes_batch.shape[0]
                    audio_codes_batch = audio_codes_batch[:cutoff_idx]
                    codec_decode_output = self.codec_model.decode(audio_codes_batch.transpose(0, 1).unsqueeze(0))
                    audio.append(codec_decode_output.audio_values)

        if generate_returned_dict:
            return ChromaGenerateOutput(audio=audio, **generate_output)
        elif output_audio:
            return audio
        return generate_output


class ChromaForConditionalGeneration(ChromaPreTrainedModel, ChromaGenerationMixin):
    """
    The full Chroma model: a Qwen2.5-Omni-based reasoner that understands text and audio input and produces
    text tokens and hidden states, a Llama-based backbone and decoder that jointly predict codec audio
    frames conditioned on the reasoner, and a Mimi codec that encodes reference audio and decodes generated
    audio codes into a waveform.
    """

    base_model_prefix = "chroma"
    _supports_flash_attn = True
    _supports_attention_backend = True
    _supports_cache_class = True

    _tied_weights_keys = {
        "backbone.audio_embedding.embed_audio_tokens.weight": "decoder.audio_embedding.embed_audio_tokens.weight",
    }

    def __init__(self, config: ChromaConfig):
        super().__init__(config)
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.backbone = ChromaBackboneForCausalLM._from_config(config.backbone_config)
        self.decoder = ChromaDecoderForCausalLM._from_config(config.decoder_config)
        self.codec_model = MimiModel._from_config(config.codec_config)

        if self.backbone.config.audio_num_codebooks != config.audio_num_codebooks:
            raise ValueError(
                f"backbone.config.audio_num_codebooks {self.backbone.config.audio_num_codebooks} != "
                f"config.audio_num_codebooks {config.audio_num_codebooks}"
            )
        if self.decoder.config.audio_num_codebooks != config.audio_num_codebooks:
            raise ValueError(
                f"decoder.config.audio_num_codebooks {self.decoder.config.audio_num_codebooks} != "
                f"config.audio_num_codebooks {config.audio_num_codebooks}"
            )

        self.post_init()
        self._prompt_embeddings_initialized = False

    def _tie_weights(self):
        self._tie_or_clone_weights(
            self.backbone.audio_embedding.embed_audio_tokens,
            self.decoder.audio_embedding.embed_audio_tokens,
        )

    def _embed_text_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.thinker.model.embed_tokens(ids.to(self.device))

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_values_cutoffs: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        thinker_input_ids: Optional[torch.LongTensor] = None,
        thinker_attention_mask: Optional[torch.LongTensor] = None,
        thinker_cache_position: Optional[torch.LongTensor] = None,
        thinker_past_key_values: Optional[Cache] = None,
        thinker_hidden_states: Optional[torch.FloatTensor] = None,
        thinker_input_features: Optional[torch.FloatTensor] = None,
        thinker_feature_attention_mask: Optional[torch.LongTensor] = None,
        thinker_logits: Optional[torch.FloatTensor] = None,
        thinker_flag: bool = True,
        thinker_eos: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        Builds the backbone's `input_embeddings` for the next step, using `input_values` to build the
        initial prompt on the first step, or the previous step's generated audio frame on later steps. On
        steps where `thinker_flag` is set, also advances the reasoner by one token and injects its hidden
        state and next-token embedding into the sequence, following a 1:2 backbone-to-reasoner token ratio.
        """
        if input_values is not None:
            inputs_embeds, attention_mask = self._build_prompt_embeds(
                input_ids, attention_mask, input_values, input_values_cutoffs
            )
        else:
            inputs_embeds = self.backbone.emb_audio_frames(input_ids.to(self.device))

        if thinker_eos is None:
            reference = thinker_input_ids if thinker_input_ids is not None else inputs_embeds
            thinker_eos = torch.zeros(reference.shape[0], dtype=torch.bool, device=reference.device)

        if thinker_input_ids is not None and thinker_flag:
            thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values = (
                self._update_thinker_model_kwargs(
                    thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values
                )
            )

            thinker_outputs = self.thinker(
                input_ids=thinker_input_ids,
                input_features=thinker_input_features,
                attention_mask=thinker_attention_mask,
                feature_attention_mask=thinker_feature_attention_mask,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                past_key_values=thinker_past_key_values,
                cache_position=thinker_cache_position,
                use_audio_in_video=False,
            )

            thinker_hidden_states = thinker_outputs.hidden_states[-1]
            thinker_past_key_values = thinker_outputs.past_key_values
            thinker_logits = thinker_outputs.logits

            thinker_next_ids = thinker_logits[:, -1:, :].argmax(dim=-1)
            next_token_emb = self._embed_text_tokens(thinker_next_ids)

            next_token_eos = thinker_next_ids.squeeze(-1) == self.config.im_end_token_id
            new_thinker_eos = thinker_eos | next_token_eos

            thinker_input_embeddings = torch.cat([thinker_hidden_states[:, -1:, :], next_token_emb], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, thinker_input_embeddings], dim=1)

            # The two injected reasoner tokens (hidden state + next-token embedding) attend fully in this
            # step even if the reasoner just reached its end-of-turn token; only tokens injected after that
            # point get masked out.
            thinker_attention_values = (~thinker_eos).long().unsqueeze(1)
            attention_mask = torch.cat([attention_mask, thinker_attention_values, thinker_attention_values], dim=1)

            thinker_eos = new_thinker_eos
            thinker_input_ids = thinker_next_ids if not thinker_eos.all() else None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        expected_attention_mask_length = past_seen_tokens + inputs_embeds.shape[1]
        if attention_mask.shape[1] != expected_attention_mask_length:
            raise ValueError(
                f"attention_mask.shape[1] {attention_mask.shape[1]} != expected_attention_mask_length "
                f"{expected_attention_mask_length}"
            )

        return {
            "input_ids": None,
            "input_embeddings": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": True,
            "output_hidden_states": True,
            "thinker_past_key_values": thinker_past_key_values,
            "thinker_hidden_states": thinker_hidden_states,
            "thinker_logits": thinker_logits,
            "thinker_input_ids": thinker_input_ids,
            "thinker_attention_mask": thinker_attention_mask,
            "thinker_input_features": thinker_input_features,
            "thinker_feature_attention_mask": thinker_feature_attention_mask,
            "thinker_cache_position": thinker_cache_position,
            "thinker_flag": not thinker_flag if thinker_input_ids is not None else False,
            "thinker_eos": thinker_eos,
        }

    @torch.no_grad()
    def _register_prompt_embeddings(self):
        text_start_ids = torch.tensor([self.config.text_start_token_id], dtype=torch.long, device=self.device)
        self.register_buffer(
            "text_start_emb", self.thinker.model.embed_tokens(text_start_ids).unsqueeze(0), persistent=False
        )

        text_end_ids = torch.tensor([self.config.text_end_token_id], dtype=torch.long, device=self.device)
        self.register_buffer(
            "text_end_emb", self.thinker.model.embed_tokens(text_end_ids).unsqueeze(0), persistent=False
        )

        self.register_buffer(
            "eos_token_audio",
            torch.zeros(
                (1, 1, self.config.backbone_config.hidden_size), dtype=self.text_start_emb.dtype, device=self.device
            ),
            persistent=False,
        )
        self.register_buffer("attention_mask", torch.ones(1, 1, dtype=torch.long, device=self.device), persistent=False)
        self.register_buffer(
            "arr", torch.arange(self.config.backbone_config.max_position_embeddings, device=self.device), persistent=False
        )

        self._prompt_embeddings_initialized = True

    def _build_prompt_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the initial backbone input sequence: `<text_start> prompt_text <text_end> prompt_audio_codes
        <eos>`, where `prompt_audio_codes` are the reference audio encoded through the Mimi codec and then
        embedded the same way as generated audio frames.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Prompt text ids.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            input_values (`torch.Tensor` of shape `(batch_size, channels, audio_seq_len)`):
                Reference audio waveform.
            input_values_cutoffs (`torch.Tensor` of shape `(batch_size, max_num_audio)`):
                End position of the reference audio within `input_values`.

        Returns:
            `tuple(torch.Tensor, torch.Tensor)`: The input embeddings and their attention mask.
        """
        if not self._prompt_embeddings_initialized:
            self._register_prompt_embeddings()

        batch_size = input_ids.shape[0]
        if batch_size != input_values.shape[0]:
            raise ValueError(f"input_values.shape[0] {input_values.shape[0]} != input_ids.shape[0] {batch_size}")
        if batch_size != input_values_cutoffs.shape[0]:
            raise ValueError(
                f"input_values_cutoffs.shape[0] {input_values_cutoffs.shape[0]} != input_ids.shape[0] {batch_size}"
            )
        if batch_size != attention_mask.shape[0]:
            raise ValueError(f"attention_mask.shape[0] {attention_mask.shape[0]} != input_ids.shape[0] {batch_size}")

        audio_codes = self.codec_model.encode(input_values).audio_codes
        audio_codes = audio_codes[:, : self.config.audio_num_codebooks, :]

        prompt_audio_emb = self.backbone.emb_audio_frames(audio_codes.permute(0, 2, 1).to(self.device))
        prompt_audio_attention_mask = torch.ones((batch_size, prompt_audio_emb.shape[1]), device=self.device)

        audio_codes_cutoffs = torch.ceil(input_values_cutoffs / self.config.audio_frame_freq).long().unsqueeze(1)
        arr = self.arr[: prompt_audio_emb.shape[1]].unsqueeze(0).expand(batch_size, -1)
        prompt_audio_attention_mask[arr >= audio_codes_cutoffs] = 0

        prompt_text_emb = self._embed_text_tokens(input_ids.to(self.device))
        prompt_text_attention_mask = attention_mask.clone()

        input_embeddings = torch.cat(
            [
                self.text_start_emb.expand(batch_size, 1, -1),
                prompt_text_emb,
                self.text_end_emb.expand(batch_size, 1, -1),
                prompt_audio_emb,
                self.eos_token_audio.expand(batch_size, 1, -1),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                self.attention_mask.expand(batch_size, 1),
                prompt_text_attention_mask,
                self.attention_mask.expand(batch_size, 1),
                prompt_audio_attention_mask,
                self.attention_mask.expand(batch_size, 1),
            ],
            dim=1,
        )
        return input_embeddings, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[torch.FloatTensor, ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        input_embeddings: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ChromaOutputWithPast:
        """
        Args:
            input_embeddings (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The backbone's input embeddings, built by [`~ChromaForConditionalGeneration.prepare_inputs_for_generation`].
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Codebook-0 target ids for computing the backbone's language modeling loss.

        Returns:
            [`ChromaOutputWithPast`]
        """
        backbone_outputs: CausalLMOutputWithPast = self.backbone(
            input_embeddings=input_embeddings,
            labels=labels,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return self._build_outputs(
            loss=backbone_outputs.loss,
            logits=backbone_outputs.logits,
            hidden_states=backbone_outputs.hidden_states,
            past_key_values=backbone_outputs.past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )

    def _build_outputs(self, **kwargs) -> ChromaOutputWithPast:
        field_names = {f.name for f in fields(ChromaOutputWithPast)}
        return ChromaOutputWithPast(**{k: v for k, v in kwargs.items() if k in field_names})

    def _update_model_kwargs_for_generation(
        self,
        outputs: ChromaOutputWithPast,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        for key in PASSTHROUGH_KEYS:
            model_kwargs[key] = getattr(outputs, key, None)
        for key in ONE_TIME_KEYS:
            model_kwargs[key] = None

        # The backbone always advances by exactly one audio frame per step, regardless of how many
        # reasoner tokens were injected into that step's input embeddings.
        return super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, 1)

    def _update_thinker_model_kwargs(
        self,
        thinker_input_ids: torch.Tensor,
        thinker_attention_mask: Optional[torch.Tensor] = None,
        thinker_cache_position: Optional[torch.Tensor] = None,
        thinker_past_key_values: Optional[Cache] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Cache]]:
        past_seen_tokens = thinker_past_key_values.get_seq_length() if thinker_past_key_values is not None else 0
        num_new_tokens = thinker_input_ids.shape[1]

        if thinker_cache_position is None:
            thinker_cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + thinker_input_ids.shape[1], device=thinker_input_ids.device
            )
        else:
            thinker_cache_position = thinker_cache_position[-num_new_tokens:] + num_new_tokens

        if thinker_attention_mask is None:
            thinker_attention_mask = torch.ones(
                (thinker_input_ids.shape[0], num_new_tokens), device=thinker_input_ids.device
            )
        elif thinker_past_key_values is not None:
            thinker_attention_mask = torch.cat(
                [thinker_attention_mask, thinker_attention_mask.new_ones((thinker_attention_mask.shape[0], num_new_tokens))],
                dim=-1,
            )

        return thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values


__all__ = [
    "ChromaPreTrainedModel",
    "ChromaLlamaModel",
    "ChromaBackboneForCausalLM",
    "ChromaDecoderForCausalLM",
    "ChromaForConditionalGeneration",
    "ChromaGenerationMixin",
    "ChromaOutputWithPast",
    "ChromaGenerateOutput",
]
