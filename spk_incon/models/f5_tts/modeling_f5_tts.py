# Copyright 2024 Yushen Chen and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch F5-TTS model."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .configuration_f5_tts import F5TTSConfig


class F5TTSSinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the flow-matching timestep, projected to the backbone hidden size."""

    def __init__(self, hidden_size: int, freq_embed_dim: int = 256):
        super().__init__()
        self.freq_embed_dim = freq_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.freq_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=timestep.device).float() / (half_dim - 1)
        )
        args = 1000 * timestep.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat((args.sin(), args.cos()), dim=-1).to(timestep.dtype)
        return self.mlp(embedding)


class F5TTSConvPositionEmbedding(nn.Module):
    """Depthwise-convolutional relative position embedding added to the mel/text input stream."""

    def __init__(self, hidden_size: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            hidden_states = hidden_states.masked_fill(~mask.unsqueeze(-1), 0.0)
        hidden_states = hidden_states.permute(0, 2, 1)
        for layer in self.block:
            hidden_states = layer(hidden_states)
            if mask is not None and isinstance(layer, nn.Conv1d):
                hidden_states = hidden_states.masked_fill(~mask.unsqueeze(1), 0.0)
        return hidden_states.permute(0, 2, 1)


class F5TTSGRN(nn.Module):
    """Global response normalization used inside the ConvNeXt-V2 text encoder blocks."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(hidden_states, p=2, dim=1, keepdim=True)
        norm = norm / (norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (hidden_states * norm) + self.beta + hidden_states


class F5TTSConvNeXtBlock(nn.Module):
    """ConvNeXt-V2 block used to contextualize the text embedding stream before it is upsampled."""

    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = F5TTSGRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.grn(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        return residual + hidden_states


class F5TTSTextEmbedding(nn.Module):
    """Embeds the input text ids and upsamples them to the mel sequence length by right-padding/truncation."""

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.text_embed = nn.Embedding(config.vocab_size + 1, config.text_dim)
        self.mask_padding = config.text_mask_padding
        self.precompute_max_pos = 8192
        self.register_buffer(
            "freqs_cis",
            _precompute_sinusoidal_position_embedding(config.text_dim, self.precompute_max_pos),
            persistent=False,
        )
        self.text_blocks = nn.Sequential(
            *[F5TTSConvNeXtBlock(config.text_dim, config.text_dim * 2) for _ in range(config.text_conv_layers)]
        )

    def forward(self, text_ids: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        text_ids = text_ids + 1
        text_ids = text_ids[:, :seq_len]
        text_ids = F.pad(text_ids, (0, seq_len - text_ids.shape[1]), value=0)
        text_mask = text_ids == 0 if self.mask_padding else None

        if drop_text:
            text_ids = torch.zeros_like(text_ids)

        hidden_states = self.text_embed(text_ids)
        hidden_states = hidden_states + self.freqs_cis[:seq_len, :]

        if text_mask is not None:
            hidden_states = hidden_states.masked_fill(text_mask.unsqueeze(-1), 0.0)
            for block in self.text_blocks:
                hidden_states = block(hidden_states)
                hidden_states = hidden_states.masked_fill(text_mask.unsqueeze(-1), 0.0)
        else:
            hidden_states = self.text_blocks(hidden_states)

        return hidden_states


class F5TTSInputEmbedding(nn.Module):
    """Combines the noised mel, the masked reference mel, and the text stream into the backbone's input."""

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.proj = nn.Linear(config.mel_dim * 2 + config.text_dim, config.hidden_size)
        self.conv_pos_embed = F5TTSConvPositionEmbedding(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor,
        text_embed: torch.Tensor,
        drop_audio_cond: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        hidden_states = self.proj(torch.cat((hidden_states, cond, text_embed), dim=-1))
        return self.conv_pos_embed(hidden_states, mask=mask) + hidden_states


def _precompute_sinusoidal_position_embedding(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


class F5TTSRotaryEmbedding(nn.Module):
    """Rotary position embedding over the full (reference + generated) mel sequence length."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
        return freqs.unsqueeze(0)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.unflatten(-1, (-1, 2))
    x1, x2 = hidden_states.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_pos_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rot_dim = freqs.shape[-1]
    seq_len = hidden_states.shape[-2]
    freqs = freqs[:, -seq_len:, :].unsqueeze(1)
    rotated, unrotated = hidden_states[..., :rot_dim], hidden_states[..., rot_dim:]
    rotated = rotated * freqs.cos() + _rotate_half(rotated) * freqs.sin()
    return torch.cat((rotated, unrotated), dim=-1)


class F5TTSAttention(nn.Module):
    """Bidirectional self-attention over the concatenated noised/reference mel and text-conditioned stream."""

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        inner_dim = self.num_heads * self.head_dim

        self.to_q = nn.Linear(config.hidden_size, inner_dim)
        self.to_k = nn.Linear(config.hidden_size, inner_dim)
        self.to_v = nn.Linear(config.hidden_size, inner_dim)
        self.to_out = nn.Linear(inner_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.conv_layers_dropout)

        if config.qk_norm == "rms_norm":
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
        elif config.qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        else:
            raise ValueError(f"Unsupported qk_norm: {config.qk_norm}")

        self.pe_attn_head = config.pe_attn_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.to_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        num_rope_heads = self.pe_attn_head if self.pe_attn_head is not None else self.num_heads
        query = torch.cat(
            [_apply_rotary_pos_emb(query[:, :num_rope_heads], rope), query[:, num_rope_heads:]], dim=1
        )
        key = torch.cat([_apply_rotary_pos_emb(key[:, :num_rope_heads], rope), key[:, num_rope_heads:]], dim=1)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].expand(batch_size, self.num_heads, seq_len, seq_len)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, is_causal=False)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.dropout(self.to_out(attn_output))

        if mask is not None:
            attn_output = attn_output.masked_fill(~mask.unsqueeze(-1), 0.0)
        return attn_output


class F5TTSAdaLayerNorm(nn.Module):
    """Adaptive layer norm conditioned on the flow-matching timestep, producing gate/shift/scale for a block."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 6)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, timestep_embed: torch.Tensor):
        emb = self.linear(self.silu(timestep_embed))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class F5TTSFeedForward(nn.Module):
    def __init__(self, config: F5TTSConfig):
        super().__init__()
        inner_dim = config.hidden_size * config.ff_mult
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.conv_layers_dropout),
            nn.Linear(inner_dim, config.hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class F5TTSDiTBlock(nn.Module):
    """Diffusion-transformer block: AdaLN-modulated self-attention followed by an AdaLN-modulated MLP."""

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.attn_norm = F5TTSAdaLayerNorm(config.hidden_size)
        self.attn = F5TTSAttention(config)
        self.ff_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = F5TTSFeedForward(config)

    def forward(self, hidden_states, timestep_embed, mask, rope):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, timestep_embed)
        attn_output = self.attn(norm, mask=mask, rope=rope)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        return hidden_states + gate_mlp.unsqueeze(1) * ff_output


class F5TTSFinalLayerNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, timestep_embed: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(timestep_embed)).chunk(2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


@dataclass
class F5TTSOutput(ModelOutput):
    """
    Args:
        velocity (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
            The predicted flow-matching velocity field at the given timestep.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Flow-matching mean-squared-error loss, returned when `target` is given.
    """

    velocity: torch.FloatTensor = None
    loss: torch.FloatTensor | None = None


class F5TTSPreTrainedModel(PreTrainedModel):
    config_class = F5TTSConfig
    base_model_prefix = "model"
    main_input_name = "noisy_mel"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, (nn.LayerNorm, LlamaRMSNorm)) and module.weight is not None:
            module.weight.data.fill_(1.0)


class F5TTSModel(F5TTSPreTrainedModel):
    r"""
    The F5-TTS diffusion-transformer (DiT) backbone. Given a noised mel spectrogram, a masked reference mel
    spectrogram, and text token ids, predicts the conditional-flow-matching velocity field that transports Gaussian
    noise towards the target mel spectrogram.
    """

    def __init__(self, config: F5TTSConfig):
        super().__init__(config)
        self.time_embed = F5TTSSinusoidalTimestepEmbedding(config.hidden_size)
        self.text_embed = F5TTSTextEmbedding(config)
        self.input_embed = F5TTSInputEmbedding(config)
        self.rotary_embed = F5TTSRotaryEmbedding(config.head_dim, theta=config.rope_theta)
        self.layers = nn.ModuleList([F5TTSDiTBlock(config) for _ in range(config.num_hidden_layers)])
        self.long_skip_connection = (
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            if config.long_skip_connection
            else None
        )
        self.norm_out = F5TTSFinalLayerNorm(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

        self.post_init()

    def forward(
        self,
        noisy_mel: torch.Tensor,
        cond_mel: torch.Tensor,
        text_ids: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        target: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> F5TTSOutput:
        """
        Args:
            noisy_mel (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
                Mel spectrogram at the current flow-matching timestep.
            cond_mel (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
                Reference mel spectrogram, zeroed outside the reference span.
            text_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Text token ids, right-padded with `-1`.
            timestep (`torch.FloatTensor` of shape `(batch_size,)`):
                Flow-matching timestep in `[0, 1]`, shared across the sequence dimension.
            mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask over the mel sequence dimension, `True` for valid positions.
            drop_audio_cond (`bool`, *optional*, defaults to `False`):
                Zero out `cond_mel`, used for classifier-free guidance.
            drop_text (`bool`, *optional*, defaults to `False`):
                Zero out the text stream, used for classifier-free guidance.
            target (`torch.FloatTensor`, *optional*):
                Target flow (`x1 - x0`). When given, a flow-matching MSE loss is computed over `loss_mask`.
            loss_mask (`torch.BoolTensor`, *optional*):
                Positions to include in the loss when `target` is given.

        Returns:
            [`F5TTSOutput`]
        """
        seq_len = noisy_mel.shape[1]
        if timestep.ndim == 0:
            timestep = timestep.repeat(noisy_mel.shape[0])
        timestep_embed = self.time_embed(timestep)

        text_embed = self.text_embed(text_ids, seq_len, drop_text=drop_text)
        hidden_states = self.input_embed(
            noisy_mel, cond_mel, text_embed, drop_audio_cond=drop_audio_cond, mask=mask
        )

        rope = self.rotary_embed(seq_len)
        residual = hidden_states if self.long_skip_connection is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, timestep_embed, mask, rope)

        if self.long_skip_connection is not None:
            hidden_states = self.long_skip_connection(torch.cat((hidden_states, residual), dim=-1))

        hidden_states = self.norm_out(hidden_states, timestep_embed)
        velocity = self.proj_out(hidden_states)

        loss = None
        if target is not None:
            loss = F.mse_loss(velocity, target, reduction="none")
            loss = loss[loss_mask].mean() if loss_mask is not None else loss.mean()

        return F5TTSOutput(velocity=velocity, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        cond_mel: torch.Tensor,
        text_ids: torch.Tensor,
        duration: torch.Tensor,
        steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
    ) -> torch.Tensor:
        """
        Sample a target mel spectrogram with a fixed-step Euler ODE solver over the flow-matching velocity field.

        Args:
            cond_mel (`torch.FloatTensor` of shape `(batch_size, ref_sequence_length, mel_dim)`):
                Reference mel spectrogram to clone the voice from.
            text_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Text token ids for the reference transcript followed by the text to synthesize.
            duration (`torch.LongTensor` of shape `(batch_size,)`):
                Total mel sequence length (reference plus generated) to sample for each item.
            steps (`int`, *optional*, defaults to 32):
                Number of Euler integration steps.
            cfg_strength (`float`, *optional*, defaults to 2.0):
                Classifier-free guidance strength. Set to `0` to disable guidance.
            sway_sampling_coef (`float`, *optional*, defaults to -1.0):
                Coefficient biasing timesteps towards `t=1`, improving quality at low step counts.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, max(duration), mel_dim)`: The sampled mel spectrogram,
            with the reference span overwritten by `cond_mel`.
        """
        batch_size, ref_len = cond_mel.shape[:2]
        max_duration = int(duration.amax().item())

        cond = F.pad(cond_mel, (0, 0, 0, max_duration - ref_len), value=0.0)
        cond_mask = torch.arange(max_duration, device=cond.device)[None, :] < ref_len
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        mask = None
        if batch_size > 1:
            mask = torch.arange(max_duration, device=cond.device)[None, :] < duration[:, None]

        timesteps = torch.linspace(0, 1, steps + 1, device=cond.device, dtype=cond.dtype)
        timesteps = timesteps + sway_sampling_coef * (torch.cos(torch.pi / 2 * timesteps) - 1 + timesteps)

        hidden_states = torch.randn(batch_size, max_duration, self.config.mel_dim, device=cond.device, dtype=cond.dtype)

        for i in range(steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - t
            if cfg_strength < 1e-5:
                velocity = self(
                    noisy_mel=hidden_states,
                    cond_mel=step_cond,
                    text_ids=text_ids,
                    timestep=t,
                    mask=mask,
                ).velocity
            else:
                velocity_cond = self(
                    noisy_mel=hidden_states,
                    cond_mel=step_cond,
                    text_ids=text_ids,
                    timestep=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                ).velocity
                velocity_uncond = self(
                    noisy_mel=hidden_states,
                    cond_mel=step_cond,
                    text_ids=text_ids,
                    timestep=t,
                    mask=mask,
                    drop_audio_cond=True,
                    drop_text=True,
                ).velocity
                velocity = velocity_cond + (velocity_cond - velocity_uncond) * cfg_strength

            hidden_states = hidden_states + dt * velocity

        return torch.where(cond_mask, cond, hidden_states)

    def compute_training_loss(
        self,
        cond_mel: torch.Tensor,
        text_ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        frac_lengths_min: float = 0.7,
        frac_lengths_max: float = 1.0,
    ) -> F5TTSOutput:
        """
        Rectified-flow-matching training step, `labels` playing the role of the standard `transformers` training
        target: it is the clean mel spectrogram to learn to generate. Internally samples a Gaussian noise source
        and a flow-matching timestep, interpolates between them to build the DiT input, masks a random contiguous
        span of `labels` out of the conditioning stream (the span the model must infill), and returns the
        resulting velocity-prediction MSE loss over that span, matching the original F5-TTS `CFM` training
        objective.

        Args:
            cond_mel (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
                Reference mel spectrogram; the random infill span described above is zeroed out of it internally.
            text_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Text token ids, right-padded with `-1`.
            labels (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
                Clean target mel spectrogram to train the flow-matching objective towards.
            mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask over the mel sequence dimension, `True` for valid (non-padding) positions.
            drop_audio_cond (`bool`, *optional*, defaults to `False`):
                Zero out the reference conditioning mel, used to train the unconditional branch for CFG.
            drop_text (`bool`, *optional*, defaults to `False`):
                Zero out the text stream, used to train the unconditional branch for CFG.
            frac_lengths_min (`float`, *optional*, defaults to 0.7):
                Lower bound of the fraction of each sequence covered by the random infill span.
            frac_lengths_max (`float`, *optional*, defaults to 1.0):
                Upper bound of the fraction of each sequence covered by the random infill span.

        Returns:
            [`F5TTSOutput`] with `loss` populated.
        """
        batch_size, seq_len, _ = labels.shape
        if mask is None:
            mask = labels.new_ones((batch_size, seq_len), dtype=torch.bool)
        lengths = mask.sum(dim=-1)

        noise = torch.randn_like(labels)
        timestep = torch.rand(batch_size, device=labels.device, dtype=labels.dtype)
        t = timestep[:, None, None]
        noisy_mel = (1 - t) * noise + t * labels
        target = labels - noise

        frac = torch.empty(batch_size, device=labels.device).uniform_(frac_lengths_min, frac_lengths_max)
        span_lengths = (frac * lengths).long()
        max_start = (lengths - span_lengths).clamp(min=0)
        start = (torch.rand(batch_size, device=labels.device) * (max_start + 1).float()).long()
        positions = torch.arange(seq_len, device=labels.device)[None, :]
        span_mask = (positions >= start[:, None]) & (positions < (start + span_lengths)[:, None])
        loss_mask = span_mask & mask

        cond = torch.where(span_mask.unsqueeze(-1), torch.zeros_like(labels), cond_mel)

        return self(
            noisy_mel=noisy_mel,
            cond_mel=cond,
            text_ids=text_ids,
            timestep=timestep,
            mask=mask,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            target=target,
            loss_mask=loss_mask,
        )


class F5TTSForConditionalGeneration(F5TTSPreTrainedModel):
    r"""
    F5-TTS model with the DiT backbone plus a thin `generate` wrapper matching the
    [`transformers`] conditional generation naming convention.
    """

    def __init__(self, config: F5TTSConfig):
        super().__init__(config)
        self.model = F5TTSModel(config)
        self.post_init()

    def forward(self, *args, labels: torch.Tensor | None = None, **kwargs) -> F5TTSOutput:
        """
        Standard `transformers` training entry point. When `labels` (the clean target mel spectrogram) is given,
        delegates to [`~F5TTSModel.compute_training_loss`] to sample the flow-matching timestep/noise/infill span
        and return the resulting loss on the output, the same way any other `*ForConditionalGeneration` model
        computes its training loss from `labels`. F5-TTS has no discrete-token stage, so this loss is the model's
        native rectified-flow-matching MSE objective rather than cross-entropy. Without `labels`, forwards
        straight through to [`~F5TTSModel.forward`] for inference-style calls that already supply
        `noisy_mel`/`timestep`/`target` directly.
        """
        if labels is not None:
            return self.model.compute_training_loss(*args, labels=labels, **kwargs)
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """See [`~F5TTSModel.generate`]."""
        return self.model.generate(*args, **kwargs)


__all__ = [
    "F5TTSForConditionalGeneration",
    "F5TTSModel",
    "F5TTSOutput",
    "F5TTSPreTrainedModel",
]
