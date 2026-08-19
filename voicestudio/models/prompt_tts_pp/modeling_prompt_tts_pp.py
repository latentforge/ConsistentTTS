# Copyright 2024 LY Corporation and the LatentForge team. All rights reserved.
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
"""PyTorch PromptTTS++ model."""

from dataclasses import dataclass

import torch
from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.fastspeech2_conformer.modeling_fastspeech2_conformer import (
    FastSpeech2ConformerHifiGan,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerModelOutput,
    FastSpeech2ConformerWithHifiGanOutput,
)
from transformers.utils import auto_docstring

from .configuration_prompt_tts_pp import PromptTTSppConfig, PromptTTSppPromptEncoderConfig


@auto_docstring(
    custom_intro="""
    Output type of [`PromptTTSppForConditionalGeneration`].
    """
)
@dataclass
class PromptTTSppOutput(FastSpeech2ConformerWithHifiGanOutput):
    r"""
    style_embedding (`torch.FloatTensor` of shape `(batch_size, style_embedding_dim)`, *optional*):
        Style embedding produced by [`PromptTTSppPromptEncoder`] from the style prompt, before being consumed by
        the acoustic model.
    """

    style_embedding: torch.FloatTensor | None = None


class PromptTTSppPromptEncoder(nn.Module):
    """
    Turns a natural-language style/speaker description into a fixed-size style embedding by pooling a BERT
    encoder's `[CLS]` representation through a small adaptor MLP.
    """

    def __init__(self, config: PromptTTSppPromptEncoderConfig):
        super().__init__()
        self.bert = BertModel(config.text_config, add_pooling_layer=False)
        self.adaptor = nn.Sequential(
            nn.Linear(config.text_config.hidden_size, config.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(config.mid_channels, config.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(config.mid_channels, config.out_channels),
        )

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor | None = None) -> torch.Tensor:
        cls_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.adaptor(cls_hidden_state)


@auto_docstring
class PromptTTSppPreTrainedModel(PreTrainedModel):
    config: PromptTTSppConfig
    base_model_prefix = "prompt_tts_pp"
    main_input_name = "input_ids"


@auto_docstring(
    custom_intro="""
    The PromptTTS++ acoustic model: a [`FastSpeech2ConformerModel`] whose encoder output is conditioned on a style
    embedding produced from a natural-language style prompt by [`PromptTTSppPromptEncoder`], instead of a
    pre-computed speaker embedding.
    """
)
class PromptTTSppModel(PromptTTSppPreTrainedModel):
    def __init__(self, config: PromptTTSppConfig):
        super().__init__(config)
        self.prompt_encoder = PromptTTSppPromptEncoder(config.prompt_encoder_config)
        self.acoustic_model = FastSpeech2ConformerModel(config.model_config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        prompt_input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        prompt_attention_mask: torch.LongTensor | None = None,
        spectrogram_labels: torch.FloatTensor | None = None,
        duration_labels: torch.LongTensor | None = None,
        pitch_labels: torch.FloatTensor | None = None,
        energy_labels: torch.FloatTensor | None = None,
        return_dict: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> tuple | FastSpeech2ConformerModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Phoneme sequence of the text to synthesize.
        prompt_input_ids (`torch.LongTensor` of shape `(batch_size, prompt_sequence_length)`):
            Token ids of the natural-language style/speaker description, tokenized with the prompt encoder's BERT
            tokenizer.
        prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, prompt_sequence_length)`, *optional*):
            Mask to avoid attending to padding tokens in `prompt_input_ids`.
        spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*):
            Batch of padded target features.
        duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*):
            Batch of padded durations.
        pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*):
            Batch of padded token-averaged pitch.
        energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*):
            Batch of padded token-averaged energy.
        """
        style_embedding = self.prompt_encoder(prompt_input_ids, prompt_attention_mask)
        return self.acoustic_model(
            input_ids,
            attention_mask=attention_mask,
            spectrogram_labels=spectrogram_labels,
            duration_labels=duration_labels,
            pitch_labels=pitch_labels,
            energy_labels=energy_labels,
            speaker_embedding=style_embedding,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The full PromptTTS++ model: [`PromptTTSppModel`] paired with a [`FastSpeech2ConformerHifiGan`] vocoder, going
    from phonemes and a natural-language style prompt directly to a waveform.
    """
)
class PromptTTSppForConditionalGeneration(PromptTTSppPreTrainedModel):
    def __init__(self, config: PromptTTSppConfig):
        super().__init__(config)
        self.model = PromptTTSppModel(config)
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        prompt_input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        prompt_attention_mask: torch.LongTensor | None = None,
        spectrogram_labels: torch.FloatTensor | None = None,
        duration_labels: torch.LongTensor | None = None,
        pitch_labels: torch.FloatTensor | None = None,
        energy_labels: torch.FloatTensor | None = None,
        return_dict: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> tuple | PromptTTSppOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Phoneme sequence of the text to synthesize.
        prompt_input_ids (`torch.LongTensor` of shape `(batch_size, prompt_sequence_length)`):
            Token ids of the natural-language style/speaker description, tokenized with the prompt encoder's BERT
            tokenizer.
        prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, prompt_sequence_length)`, *optional*):
            Mask to avoid attending to padding tokens in `prompt_input_ids`.
        spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*):
            Batch of padded target features.
        duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*):
            Batch of padded durations.
        pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*):
            Batch of padded token-averaged pitch.
        energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*):
            Batch of padded token-averaged energy.

        Example:

        ```python
        >>> from voicestudio.models.prompt_tts_pp import PromptTTSppForConditionalGeneration, PromptTTSppProcessor

        >>> processor = PromptTTSppProcessor.from_pretrained("line-corporation/promptttspp")
        >>> model = PromptTTSppForConditionalGeneration.from_pretrained("line-corporation/promptttspp")

        >>> inputs = processor.encode(text="Some text to convert to speech.", style_prompt="A calm, low-pitched voice.")
        >>> waveform = model(**inputs).waveform
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.model_config.return_dict
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.model_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.model_config.output_hidden_states
        )

        style_embedding = self.model.prompt_encoder(prompt_input_ids, prompt_attention_mask)
        model_outputs = self.model.acoustic_model(
            input_ids,
            attention_mask=attention_mask,
            spectrogram_labels=spectrogram_labels,
            duration_labels=duration_labels,
            pitch_labels=pitch_labels,
            energy_labels=energy_labels,
            speaker_embedding=style_embedding,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            # `FastSpeech2ConformerModel` prepends `loss` to the tuple only when `self.training` is
            # true (the branch that computes it), regardless of which labels were passed.
            spectrogram = model_outputs[1] if self.training else model_outputs[0]
        else:
            spectrogram = model_outputs["spectrogram"]
        waveform = self.vocoder(spectrogram)

        if not return_dict:
            return model_outputs + (waveform, style_embedding)

        return PromptTTSppOutput(waveform=waveform, style_embedding=style_embedding, **model_outputs)


__all__ = [
    "PromptTTSppForConditionalGeneration",
    "PromptTTSppModel",
    "PromptTTSppPreTrainedModel",
    "PromptTTSppPromptEncoder",
]
