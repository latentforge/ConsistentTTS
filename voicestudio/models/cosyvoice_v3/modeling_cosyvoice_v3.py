# coding=utf-8
# Copyright 2025 Alibaba Inc and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch CosyVoice v3 model."""

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from ..cosyvoice_v1.modeling_cosyvoice_v1 import (
    CosyVoiceV1ConditionalCFM,
    CosyVoiceV1HiFTGenerator,
    CosyVoiceV1InterpolateRegulator,
    CosyVoiceV1PreTrainedModel,
    CosyVoiceV1RelPositionEncoder,
    CosyVoiceV1SinusoidalPosEmb,
    _conformer_config,
    _make_pad_mask,
)
from ..cosyvoice_v2.modeling_cosyvoice_v2 import CosyVoiceV2LLM, CosyVoiceV2LLMOutput, CosyVoiceV2PreLookaheadLayer
from .configuration_cosyvoice_v3 import CosyVoiceV3Config, CosyVoiceV3FlowConfig, CosyVoiceV3LLMConfig


class CosyVoiceV3LLM(CosyVoiceV2LLM):
    r"""
    The CosyVoice v3 speech-token language model. Identical Qwen2-backbone architecture to [`CosyVoiceV2LLM`],
    except the start/task/fill/end-of-speech ids live inside the (further extended) speech-token embedding table
    instead of a separate two-entry embedding, matching the original repository's `CosyVoice3LM(Qwen2LM)`.
    """

    config_class = CosyVoiceV3LLMConfig

    def __init__(self, config: CosyVoiceV3LLMConfig):
        CosyVoiceV1PreTrainedModel.__init__(self, config)
        vocab = config.speech_token_size + 200
        self.speech_token_size = config.speech_token_size
        self.sos_eos = config.speech_token_size
        self.eos_token_id = config.speech_token_size + 1
        self.task_id = config.speech_token_size + 2
        self.fill_token_id = config.speech_token_size + 3

        self.llm = Qwen2Model(config)
        self.llm_decoder = nn.Linear(config.hidden_size, vocab, bias=False)
        self.speech_embedding = nn.Embedding(vocab, config.hidden_size)
        self.post_init()

    def forward(
        self,
        text_token: torch.LongTensor,
        speech_token: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> CosyVoiceV2LLMOutput:
        """
        Args:
            text_token (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Text token ids, tokenized by the Qwen2 backbone's tokenizer.
            speech_token (`torch.LongTensor` of shape `(batch_size, speech_sequence_length)`):
                Discrete speech token ids.
            labels (`torch.LongTensor` of shape `(batch_size, speech_sequence_length + 1)`, *optional*):
                Target speech token ids (including the trailing end-of-speech token). When given, a cross-entropy
                loss is computed.

        Returns:
            [`CosyVoiceV2LLMOutput`]
        """
        text_emb = self.llm.embed_tokens(text_token)
        speech_emb = self.speech_embedding(speech_token)
        sos_emb = self.speech_embedding.weight[self.sos_eos].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)
        task_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)

        lm_input = torch.cat([sos_emb, text_emb, task_emb, speech_emb], dim=1)
        hidden_states = self.llm(inputs_embeds=lm_input).last_hidden_state
        logits = self.llm_decoder(hidden_states)

        loss = None
        if labels is not None:
            prefix_len = lm_input.size(1) - speech_emb.size(1)
            target = torch.full((logits.size(0), lm_input.size(1)), -100, dtype=torch.long, device=logits.device)
            target[:, prefix_len - 1 :] = labels
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
        return CosyVoiceV2LLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        text_token: torch.LongTensor,
        prompt_speech_token: torch.LongTensor | None = None,
        max_new_tokens: int = 2000,
        min_new_tokens: int = 0,
        top_k: int = 25,
    ) -> torch.LongTensor:
        """See [`~CosyVoiceV2LLM.generate`]."""
        device = text_token.device
        text_emb = self.llm.embed_tokens(text_token)
        sos_emb = self.speech_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        prompt_emb = (
            self.speech_embedding(prompt_speech_token)
            if prompt_speech_token is not None and prompt_speech_token.size(1) > 0
            else torch.zeros(1, 0, text_emb.size(-1), device=device, dtype=text_emb.dtype)
        )
        lm_input = torch.cat([sos_emb, text_emb, task_emb, prompt_emb], dim=1)

        past_key_values = None
        generated = []
        for step in range(max_new_tokens):
            output = self.llm(inputs_embeds=lm_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            logits = self.llm_decoder(output.last_hidden_state[:, -1])
            logits[:, self.speech_token_size + 1 :] = -float("inf")
            if step < min_new_tokens:
                logits[:, self.eos_token_id] = -float("inf")
            top_logits, top_indices = logits.topk(top_k, dim=-1)
            probs = torch.softmax(top_logits, dim=-1)
            next_token = top_indices.gather(-1, torch.multinomial(probs, 1))
            token_id = next_token.item()
            if token_id == self.eos_token_id:
                break
            generated.append(token_id)
            lm_input = self.speech_embedding(next_token)
        return torch.tensor(generated, device=device, dtype=torch.long).unsqueeze(0)


class CosyVoiceV3DiTBlock(nn.Module):
    """AdaLN-modulated diffusion-transformer block, conditioned on the flow-matching timestep."""

    def __init__(self, hidden_size: int, num_heads: int, ff_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ff_mult), nn.GELU(approximate="tanh"), nn.Linear(hidden_size * ff_mult, hidden_size)
        )
        self.ada_ln = nn.Linear(hidden_size, hidden_size * 6)

    def forward(self, hidden_states: torch.Tensor, time_emb: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(time_emb).unsqueeze(1).chunk(6, dim=-1)
        normed = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        hidden_states = hidden_states + gate_msa * attn_out
        normed = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        return hidden_states + gate_mlp * self.ff(normed)


class CosyVoiceV3DiT(nn.Module):
    r"""
    Diffusion-transformer estimator of the conditional-flow-matching velocity field, replacing the CosyVoice
    v1/v2 U-Net estimator. Shares the `forward(x, mask, mu, t, spks, cond)` signature of
    [`CosyVoiceV1ConditionalDecoder`], so it plugs into the same [`CosyVoiceV1ConditionalCFM`] wrapper.
    """

    def __init__(self, config: CosyVoiceV3FlowConfig):
        super().__init__()
        hidden_size = config.dit_hidden_size
        in_channels = config.output_size * 3
        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.time_embeddings = CosyVoiceV1SinusoidalPosEmb(hidden_size)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(
            [CosyVoiceV3DiTBlock(hidden_size, config.dit_num_attention_heads, config.dit_ff_mult) for _ in range(config.dit_num_hidden_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj_out = nn.Linear(hidden_size, config.output_size)

    def forward(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        """See [`CosyVoiceV1ConditionalDecoder.forward`]."""
        time_emb = self.time_mlp(self.time_embeddings(t).to(t.dtype))
        spk_expanded = spks.unsqueeze(-1).expand(-1, -1, x.size(-1))
        hidden_states = torch.cat([x, mu, spk_expanded], dim=1).transpose(1, 2)
        hidden_states = self.input_proj(hidden_states)

        attn_mask = None if mask.all() else (~mask.squeeze(1).bool()).unsqueeze(1).expand(-1, mask.size(-1), -1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, time_emb, attn_mask)
        hidden_states = self.norm_out(hidden_states)
        return self.proj_out(hidden_states).transpose(1, 2) * mask


class CosyVoiceV3FlowMatchingModel(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v3 conditional-flow-matching decoder. Same pre-lookahead + Conformer encoder + length regulator
    front end as [`CosyVoiceV2FlowMatchingModel`], with the U-Net estimator replaced by [`CosyVoiceV3DiT`],
    matching the original repository's `CausalMaskedDiffWithDiT`.
    """

    config_class = CosyVoiceV3FlowConfig

    def __init__(self, config: CosyVoiceV3FlowConfig):
        super().__init__(config)
        self.token_mel_ratio = config.token_mel_ratio
        self.input_embedding = nn.Embedding(config.vocab_size, config.input_size)
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.output_size)
        self.pre_lookahead_layer = CosyVoiceV2PreLookaheadLayer(config.input_size, config.pre_lookahead_len)
        self.encoder = CosyVoiceV1RelPositionEncoder(
            _conformer_config(
                config.encoder_hidden_size,
                config.encoder_num_hidden_layers,
                config.encoder_num_attention_heads,
                config.encoder_intermediate_size,
                0.1,
                0.1,
            )
        )
        self.encoder_proj = nn.Linear(config.encoder_hidden_size, config.output_size)
        self.length_regulator = CosyVoiceV1InterpolateRegulator(config.output_size, config.output_size)
        self.decoder = CosyVoiceV1ConditionalCFM(config, CosyVoiceV3DiT(config))
        self.post_init()

    def forward(
        self,
        speech_token: torch.LongTensor,
        speech_token_len: torch.LongTensor,
        embedding: torch.FloatTensor,
        speech_feat: torch.FloatTensor | None = None,
        speech_feat_len: torch.LongTensor | None = None,
        n_timesteps: int = 10,
    ) -> ModelOutput:
        """See [`CosyVoiceV2FlowMatchingModel.forward`]."""
        spk_emb = F.normalize(embedding, dim=1)
        spk_emb = self.spk_embed_affine_layer(spk_emb)

        mask = (~_make_pad_mask(speech_token_len, speech_token.size(1))).to(spk_emb.dtype).unsqueeze(-1)
        token_emb = self.input_embedding(speech_token.clamp(min=0)) * mask
        token_emb = self.pre_lookahead_layer(token_emb)
        hidden_states = self.encoder(token_emb, attention_mask=mask.squeeze(-1).bool()).last_hidden_state
        hidden_states = self.encoder_proj(hidden_states)

        target_len = speech_feat_len if speech_feat is not None else speech_token_len * self.token_mel_ratio
        hidden_states, target_len = self.length_regulator(hidden_states, target_len)

        loss = None
        mel = None
        if speech_feat is not None:
            cond = torch.zeros_like(speech_feat).transpose(1, 2)
            out_mask = (~_make_pad_mask(target_len, speech_feat.size(1))).to(hidden_states).unsqueeze(1)
            loss = self.decoder.compute_loss(
                speech_feat.transpose(1, 2), out_mask, hidden_states.transpose(1, 2), spk_emb, cond
            )
        else:
            out_mask = (~_make_pad_mask(target_len, hidden_states.size(1))).to(hidden_states).unsqueeze(1)
            cond = torch.zeros(hidden_states.size(0), self.config.output_size, hidden_states.size(1), device=hidden_states.device, dtype=hidden_states.dtype)
            mel = self.decoder(hidden_states.transpose(1, 2), out_mask, spk_emb, cond, n_timesteps=n_timesteps)
        return ModelOutput(mel=mel, loss=loss)


class CosyVoiceV3Model(CosyVoiceV1PreTrainedModel):
    r"""
    The full CosyVoice v3 model: a Qwen2-backbone speech-token language model with an extended speech-token
    vocabulary, a DiT conditional-flow-matching mel decoder, and the CosyVoice v1/v2 HiFTNet vocoder.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__(config)
        self.llm = CosyVoiceV3LLM(config.llm_config)
        self.flow = CosyVoiceV3FlowMatchingModel(config.flow_config)
        self.hift = CosyVoiceV1HiFTGenerator(config.hift_config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV2LLMOutput:
        return self.llm(**kwargs)

    @torch.no_grad()
    def generate_speech(self, text_token, embedding, prompt_speech_token=None, **kwargs) -> torch.Tensor:
        """See [`~CosyVoiceV2Model.generate_speech`]."""
        speech_token = self.llm.generate(text_token, prompt_speech_token, **kwargs)
        speech_token_len = torch.tensor([speech_token.size(1)], device=speech_token.device)
        flow_out = self.flow(speech_token, speech_token_len, embedding)
        return self.hift(flow_out.mel)


class CosyVoiceV3ForConditionalGeneration(CosyVoiceV1PreTrainedModel):
    r"""
    CosyVoice v3 model with the Qwen2-backbone speech-token language model, the DiT flow-matching decoder, and
    the vocoder, with a `generate` method producing a waveform end-to-end. The trainable `forward` pass is the
    speech-token language model's next-token cross-entropy objective.
    """

    config_class = CosyVoiceV3Config

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__(config)
        self.model = CosyVoiceV3Model(config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV2LLMOutput:
        return self.model.llm(**kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """See [`~CosyVoiceV3Model.generate_speech`]."""
        return self.model.generate_speech(*args, **kwargs)


__all__ = [
    "CosyVoiceV3ForConditionalGeneration",
    "CosyVoiceV3Model",
    "CosyVoiceV3LLM",
    "CosyVoiceV3FlowMatchingModel",
    "CosyVoiceV3DiT",
    "CosyVoiceV3DiTBlock",
]
