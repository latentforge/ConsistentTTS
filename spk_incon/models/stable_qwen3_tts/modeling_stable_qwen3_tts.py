# Copyright 2026 The Qwen team, Alibaba Group and the LatentForge team. All rights reserved.
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

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSTalkerOutputWithPast,
)

from .configuration_stable_qwen3_tts import StableQwen3TTSConfig


_VOCAB_PROJECTION_CHUNK = 8192


class StableQwen3TTSForConditionalGeneration(Qwen3TTSForConditionalGeneration):
    r"""
    Qwen3-TTS talker conditioned on a learnable vector-quantized query.

    The query occupies the text channel of the voice design prompt in place of a fixed
    reference sentence. Quantization is a training time constraint: the straight through
    estimate from [`~StableQwen3TTSForConditionalGeneration.quantize_query`] is what the
    optimizer sees, while generation runs on the continuous `query` parameter itself, so
    the query is never committed to a discrete token. Generation is a single autoregressive roll: the leading
    `config.anchor_num_frames` frames are produced with a padded text channel and fix the
    speaker identity, then the content text streams in frame by frame through
    `trailing_text_hidden` and the utterance continues from the same key value cache.

    Args:
        config ([`StableQwen3TTSConfig`]):
            Model configuration.
    """

    config_class = StableQwen3TTSConfig
    _keys_to_ignore_on_load_missing = ["query"]

    def __init__(self, config: StableQwen3TTSConfig):
        super().__init__(config)
        hidden_size = config.talker_config.hidden_size
        self.query = nn.Parameter(torch.zeros(config.num_query_tokens, hidden_size))
        self.register_buffer(
            "query_token_ids",
            torch.zeros(config.num_query_tokens, dtype=torch.long),
            persistent=False,
        )
        self._projected_text_vocab = None
        self._query_ready = False

    def get_projected_text_vocab(self) -> torch.Tensor:
        """Return the text vocabulary embedded and projected into talker hidden space.

        Returns:
            `torch.Tensor`: Codebook of shape `(text_vocab_size, hidden_size)` in float32.
                The result is cached on the instance and excluded from the state dict.
        """
        if self._projected_text_vocab is None:
            text_embedding = self.get_text_embeddings()
            vocab_size = text_embedding.weight.shape[0]
            device = text_embedding.weight.device
            chunks = []
            with torch.no_grad():
                for start in range(0, vocab_size, _VOCAB_PROJECTION_CHUNK):
                    ids = torch.arange(start, min(start + _VOCAB_PROJECTION_CHUNK, vocab_size), device=device)
                    chunks.append(self.text_projection(text_embedding(ids).to(self.dtype)).float())
            self._projected_text_vocab = torch.cat(chunks, dim=0)
        return self._projected_text_vocab

    def init_query(
        self,
        token_ids: torch.Tensor | list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Materialize the query from vocabulary entries.

        Args:
            token_ids (`torch.Tensor` or `list[int]`, *optional*):
                Text vocabulary ids to seed the query slots with. Random ids are drawn
                when omitted.
            generator (`torch.Generator`, *optional*):
                Generator used to draw random ids.

        Returns:
            `torch.Tensor`: The seeded token ids, of shape `(num_query_tokens,)`.
        """
        codebook = self.get_projected_text_vocab()
        if token_ids is None:
            token_ids = torch.randint(
                0, codebook.shape[0], (self.config.num_query_tokens,), generator=generator, device=codebook.device
            )
        else:
            token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=codebook.device)
        with torch.no_grad():
            self.query.data = codebook[token_ids].clone().to(self.query.dtype)
        self.query_token_ids = token_ids
        self._query_ready = True
        return token_ids

    def _ensure_query(self) -> None:
        if not self._query_ready:
            self.init_query(self.config.query_token_ids)

    def quantize_query(self, allowed_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Snap the query to its nearest vocabulary entries with a straight through estimator.

        Args:
            allowed_mask (`torch.Tensor`, *optional*):
                Boolean mask of shape `(text_vocab_size,)` restricting the candidates.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The selected token ids of shape
                `(num_query_tokens,)` and the quantized query of shape
                `(num_query_tokens, hidden_size)`, which carries gradient to `query`.
        """
        self._ensure_query()
        codebook = self.get_projected_text_vocab()
        query = self.query.float()
        # argmax of the inner product minus half the squared norm is argmin of the L2 distance.
        scores = query @ codebook.t() - 0.5 * (codebook * codebook).sum(dim=-1).unsqueeze(0)
        if allowed_mask is not None:
            scores = scores.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
        token_ids = scores.argmax(dim=-1)
        quantized = codebook[token_ids]
        self.query_token_ids = token_ids
        return token_ids, self.query + (quantized - self.query).detach()

    def _talker_special_embeds(self, language: str | None, dtype: torch.dtype):
        talker_config = self.config.talker_config
        special_ids = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            device=self.device,
            dtype=torch.long,
        )
        tts_bos, tts_eos, tts_pad = self.text_projection(self.get_text_embeddings()(special_ids)).chunk(3, dim=1)

        language_id = None
        if language is not None and language.lower() != "auto" and talker_config.codec_language_id is not None:
            language_id = talker_config.codec_language_id[language.lower()]
        if language_id is None:
            think_ids = [[talker_config.codec_nothink_id, talker_config.codec_think_bos_id, talker_config.codec_think_eos_id]]
        else:
            think_ids = [
                [
                    talker_config.codec_think_id,
                    talker_config.codec_think_bos_id,
                    language_id,
                    talker_config.codec_think_eos_id,
                ]
            ]
        codec_prefix = torch.cat(
            [
                self.get_input_embeddings()(torch.tensor(think_ids, device=self.device, dtype=torch.long)),
                self.get_input_embeddings()(
                    torch.tensor(
                        [[talker_config.codec_pad_id, talker_config.codec_bos_id]], device=self.device, dtype=torch.long
                    )
                ),
            ],
            dim=1,
        )
        return tts_bos.to(dtype), tts_eos.to(dtype), tts_pad.to(dtype), codec_prefix.to(dtype)

    def build_anchor_prompt(
        self,
        instruct_ids: torch.Tensor,
        role_ids: torch.Tensor,
        query: torch.Tensor | None = None,
        language: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the prefill that opens the roll with the voice prompt and the query.

        Args:
            instruct_ids (`torch.Tensor`):
                Token ids of the instruct turn, of shape `(1, instruct_length)`.
            role_ids (`torch.Tensor`):
                Token ids of the assistant turn. Only the leading three ids are used.
            query (`torch.Tensor`, *optional*):
                Query of shape `(num_query_tokens, hidden_size)`. `query` is used as is
                when omitted, so generation runs on the continuous parameter. Training
                passes the straight through estimate from
                [`~StableQwen3TTSForConditionalGeneration.quantize_query`].
            language (`str`, *optional*):
                Language tag. `None` or `"auto"` selects the no think codec prefix.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The prefill embeddings of shape
                `(1, prompt_length, hidden_size)` and the padding embedding of shape
                `(1, 1, hidden_size)`.
        """
        if query is None:
            self._ensure_query()
            query = self.query
        dtype = self.dtype
        tts_bos, tts_eos, tts_pad, codec_prefix = self._talker_special_embeds(language, dtype)

        instruct = self.text_projection(self.get_text_embeddings()(instruct_ids))
        role = self.text_projection(self.get_text_embeddings()(role_ids[:, :3]))
        prefix = torch.cat([tts_pad.expand(-1, codec_prefix.shape[1] - 2, -1), tts_bos], dim=1) + codec_prefix[:, :-1]

        codec_pad = self.get_input_embeddings()(
            torch.tensor([[self.config.talker_config.codec_pad_id]], device=self.device, dtype=torch.long)
        ).to(dtype)
        query_block = torch.cat([query.to(dtype).unsqueeze(0), tts_eos], dim=1) + codec_pad
        start = tts_pad + codec_prefix[:, -1:]

        prompt = torch.cat([instruct.to(dtype), role.to(dtype), prefix, query_block, start], dim=1)
        return prompt, tts_pad

    def build_content_prefill(self, text_ids: torch.Tensor, language: str | None = None) -> torch.Tensor:
        """Build the block prefilled into the roll once the anchor ends.

        Args:
            text_ids (`torch.Tensor`):
                Content token ids of shape `(1, text_length)`.
            language (`str`, *optional*):
                Language tag, used to pick the codec bos embedding.

        Returns:
            `torch.Tensor`: Embeddings of shape `(1, text_length + 2, hidden_size)`, the
                content text over codec padding followed by a fresh codec bos slot.
        """
        dtype = self.dtype
        _, tts_eos, tts_pad, codec_prefix = self._talker_special_embeds(language, dtype)
        text = self.text_projection(self.get_text_embeddings()(text_ids)).to(dtype)
        text = torch.cat([text, tts_eos], dim=1)
        codec_pad = self.get_input_embeddings()(
            torch.tensor([[self.config.talker_config.codec_pad_id]], device=self.device, dtype=torch.long)
        ).to(dtype)
        return torch.cat([text + codec_pad, tts_pad + codec_prefix[:, -1:]], dim=1)

    def prefill(self, inputs_embeds, attention_mask, past_key_values):
        """Push a block of embeddings through the talker and extend the cache.

        Args:
            inputs_embeds (`torch.Tensor`):
                Block of shape `(1, block_length, hidden_size)`.
            attention_mask (`torch.Tensor`):
                Mask covering the cache and the block, of shape `(1, total_length)`.
            past_key_values (`Cache`):
                Cache to extend in place.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: Code group 0 logits for the last block
                position and the talker hidden state at that position.
        """
        position_ids, rope_deltas = self.get_rope_index(attention_mask)
        self.rope_deltas = rope_deltas - (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        position_ids = position_ids[:, :, -inputs_embeds.shape[1] :]
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state
        return self.codec_head(hidden)[:, -1], hidden[:, -1:, :]

    def _sampling_processors(self, top_k, top_p, temperature, repetition_penalty):
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            SuppressTokensLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        talker_config = self.config.talker_config
        suppress = [
            i
            for i in range(talker_config.vocab_size - 1024, talker_config.vocab_size)
            if i != talker_config.codec_eos_token_id
        ]
        processors = LogitsProcessorList()
        if repetition_penalty and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        processors.append(SuppressTokensLogitsProcessor(suppress, device=self.device))
        if temperature and temperature != 1.0:
            processors.append(TemperatureLogitsWarper(temperature))
        if top_k:
            processors.append(TopKLogitsWarper(top_k))
        if top_p and top_p < 1.0:
            processors.append(TopPLogitsWarper(top_p))
        return processors

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor],
        languages: list[str] | None = None,
        max_new_tokens: int = 4096,
        max_anchor_tokens: int = 512,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        content_mode: str = "prefill",
        **kwargs,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """Roll the anchor and the utterance through one continuous key value cache.

        The voice prompt and the query open the roll. When the talker emits its codec end
        of sequence the frame is dropped, the content text is prefilled into the same
        cache, and decoding resumes for the utterance.

        Args:
            input_ids (`list[torch.Tensor]`):
                Per sample assistant turn token ids, each of shape `(1, length)`.
            instruct_ids (`list[torch.Tensor]`):
                Per sample instruct turn token ids, each of shape `(1, length)`.
            languages (`list[str]`, *optional*):
                Per sample language tags.
            max_new_tokens (`int`, *optional*, defaults to 4096):
                Cap on frames generated after the content prefill.
            max_anchor_tokens (`int`, *optional*, defaults to 512):
                Cap on anchor frames before the content is prefilled regardless of eos.

        Returns:
            `tuple[list[torch.Tensor], list[int]]`: Per sample codec frames of shape
                `(num_frames, num_code_groups)` with the anchor still attached, and the
                per sample anchor length in frames.
        """
        if languages is None:
            languages = [None] * len(input_ids)
        self._ensure_query()

        all_codes, anchor_lengths = [], []
        for index, input_id in enumerate(input_ids):
            codes, anchor = self._roll_one(
                instruct_ids[index],
                input_id,
                languages[index],
                max_new_tokens=max_new_tokens,
                max_anchor_tokens=max_anchor_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
                repetition_penalty=repetition_penalty,
                content_mode=content_mode,
            )
            all_codes.append(codes)
            anchor_lengths.append(anchor)
        return all_codes, anchor_lengths

    @torch.no_grad()
    def _roll_one(
        self,
        instruct_ids,
        input_id,
        language,
        max_new_tokens,
        max_anchor_tokens,
        do_sample,
        top_k,
        top_p,
        temperature,
        subtalker_dosample,
        subtalker_top_k,
        subtalker_top_p,
        subtalker_temperature,
        repetition_penalty,
        content_mode="prefill",
    ):
        from transformers.cache_utils import DynamicCache

        talker_config = self.config.talker_config
        eos_id = talker_config.codec_eos_token_id
        processors = self._sampling_processors(top_k, top_p, temperature, repetition_penalty)

        prompt, tts_pad = self.build_anchor_prompt(instruct_ids, input_id, language=language)
        content = self.build_content_prefill(input_id[:, 3:-5], language=language)
        _, tts_eos, _, _ = self._talker_special_embeds(language, self.dtype)
        content_text = torch.cat(
            [self.text_projection(self.get_text_embeddings()(input_id[:, 3:-5])).to(self.dtype), tts_eos], dim=1
        )
        trailing = tts_pad
        last_token = None

        text_ids = input_id[:, 3:-5]
        _, _, tts_pad_only, codec_prefix = self._talker_special_embeds(language, self.dtype)
        content_head = (
            self.text_projection(self.get_text_embeddings()(text_ids[:, :1])).to(self.dtype) + codec_prefix[:, -1:]
        )
        content_tail = torch.cat(
            [self.text_projection(self.get_text_embeddings()(text_ids[:, 1:])).to(self.dtype), tts_eos], dim=1
        )

        self.rope_deltas = None
        cache = DynamicCache(config=talker_config)
        attention_mask = torch.ones(prompt.shape[:2], dtype=torch.long, device=self.device)
        logits, past_hidden = self.prefill(prompt, attention_mask, cache)

        def draw(logits, history):
            scores = processors(history, logits.float())
            if not do_sample:
                return scores.argmax(dim=-1, keepdim=True)
            return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)

        history = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        frames, anchor_length, in_anchor, step = [], None, True, 0
        token = draw(logits, history)

        while True:
            if int(token) == eos_id:
                if in_anchor:
                    in_anchor = False
                    anchor_length = len(frames)
                    history = torch.zeros((1, 0), dtype=torch.long, device=self.device)
                    step = 0
                    if content_mode == "stream":
                        # The codec channel keeps carrying generated frames; only the text
                        # channel switches from padding to the content tokens.
                        trailing = content_text
                        token = last_token
                        continue
                    if content_mode == "restart":
                        # Reopen the standard streaming layout inside the same cache: the
                        # first content token rides a fresh codec bos, the rest stream.
                        content = content_head
                        trailing = content_tail
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, content.shape[1]), dtype=torch.long, device=self.device)],
                        dim=1,
                    )
                    logits, past_hidden = self.prefill(content, attention_mask, cache)
                    token = draw(logits, history)
                    continue
                break

            if in_anchor and len(frames) >= max_anchor_tokens:
                token = torch.tensor([[eos_id]], device=self.device)
                continue
            if not in_anchor and len(frames) - anchor_length >= max_new_tokens:
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=1
            )
            outputs = self(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                past_hidden=past_hidden,
                generation_step=step,
                trailing_text_hidden=trailing,
                tts_pad_embed=tts_pad,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_p=subtalker_top_p,
                subtalker_top_k=subtalker_top_k,
                subtalker_temperature=subtalker_temperature,
            )
            frames.append(outputs.hidden_states[1][0])
            past_hidden = outputs.past_hidden
            last_token = token
            history = torch.cat([history, token], dim=1)
            step += 1
            token = draw(outputs.logits[:, -1], history)

        if anchor_length is None:
            anchor_length = len(frames)
        codes = torch.stack(frames, dim=0) if frames else torch.zeros((0, talker_config.num_code_groups), dtype=torch.long, device=self.device)
        return codes, anchor_length

    @torch.no_grad()
    def generate_anchored(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor],
        languages: list[str] | None = None,
        max_anchor_tokens: int = 512,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Generate an anchor from the query, then the utterance conditioned on it.

        The anchor is produced by the same prompt the query was trained on. It then enters
        a fresh sequence as reference audio, with the query occupying the reference text
        positions, so the query is never decoded into a string.

        Args:
            input_ids (`list[torch.Tensor]`):
                Per sample assistant turn token ids, each of shape `(1, length)`.
            instruct_ids (`list[torch.Tensor]`):
                Per sample instruct turn token ids, each of shape `(1, length)`.
            languages (`list[str]`, *optional*):
                Per sample language tags.
            max_anchor_tokens (`int`, *optional*, defaults to 512):
                Cap on anchor frames.
            max_new_tokens (`int`, *optional*, defaults to 4096):
                Cap on utterance frames.

        Returns:
            `list[torch.Tensor]`: Per sample codec frames of the utterance, of shape
                `(num_frames, num_code_groups)`.
        """
        from transformers.cache_utils import DynamicCache

        if languages is None:
            languages = [None] * len(input_ids)
        self._ensure_query()
        talker_config = self.config.talker_config
        eos_id = talker_config.codec_eos_token_id
        processors = self._sampling_processors(top_k, top_p, temperature, repetition_penalty)
        dtype = self.dtype

        def draw(logits, history):
            scores = processors(history, logits.float())
            if not do_sample:
                return scores.argmax(dim=-1, keepdim=True)
            return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)

        def roll(prompt, trailing, pad_embed, limit):
            self.rope_deltas = None
            cache = DynamicCache(config=talker_config)
            mask = torch.ones(prompt.shape[:2], dtype=torch.long, device=self.device)
            logits, past_hidden = self.prefill(prompt, mask, cache)
            history = torch.zeros((1, 0), dtype=torch.long, device=self.device)
            token, frames = draw(logits, history), []
            while int(token) != eos_id and len(frames) < limit:
                mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=1)
                outputs = self(
                    input_ids=token,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                    past_hidden=past_hidden,
                    generation_step=len(frames),
                    trailing_text_hidden=trailing,
                    tts_pad_embed=pad_embed,
                    subtalker_dosample=subtalker_dosample,
                    subtalker_top_p=subtalker_top_p,
                    subtalker_top_k=subtalker_top_k,
                    subtalker_temperature=subtalker_temperature,
                )
                frames.append(outputs.hidden_states[1][0])
                past_hidden = outputs.past_hidden
                history = torch.cat([history, token], dim=1)
                token = draw(outputs.logits[:, -1], history)
            return torch.stack(frames, dim=0) if frames else None

        results = []
        for index, input_id in enumerate(input_ids):
            language = languages[index]
            tts_bos, tts_eos, tts_pad, codec_prefix = self._talker_special_embeds(language, dtype)
            instruct = self.text_projection(self.get_text_embeddings()(instruct_ids[index])).to(dtype)
            role = self.text_projection(self.get_text_embeddings()(input_id[:, :3])).to(dtype)
            prefix = torch.cat([tts_pad.expand(-1, codec_prefix.shape[1] - 2, -1), tts_bos], dim=1) + codec_prefix[:, :-1]
            codec_pad = self.get_input_embeddings()(
                torch.tensor([[talker_config.codec_pad_id]], device=self.device, dtype=torch.long)
            ).to(dtype)
            query = self.query.to(dtype).unsqueeze(0)

            anchor_prompt = torch.cat(
                [instruct, role, prefix, torch.cat([query, tts_eos], dim=1) + codec_pad, tts_pad + codec_prefix[:, -1:]],
                dim=1,
            )
            anchor = roll(anchor_prompt, tts_pad, tts_pad, max_anchor_tokens)
            if anchor is None:
                results.append(torch.zeros((0, talker_config.num_code_groups), dtype=torch.long, device=self.device))
                continue

            text = torch.cat(
                [query, self.text_projection(self.get_text_embeddings()(input_id[:, 3:-5])).to(dtype), tts_eos], dim=1
            )
            summed = self.get_input_embeddings()(anchor[:, 0])
            for group in range(talker_config.num_code_groups - 1):
                summed = summed + self.code_predictor.get_input_embeddings()[group](anchor[:, group + 1])
            codec = torch.cat(
                [
                    self.get_input_embeddings()(
                        torch.tensor([[talker_config.codec_bos_id]], device=self.device, dtype=torch.long)
                    ),
                    summed.unsqueeze(0),
                ],
                dim=1,
            ).to(dtype)

            if text.shape[1] > codec.shape[1]:
                icl, trailing = text[:, : codec.shape[1]] + codec, text[:, codec.shape[1] :]
            else:
                icl = torch.cat([text, tts_pad.expand(-1, codec.shape[1] - text.shape[1], -1)], dim=1) + codec
                trailing = tts_pad
            utterance = roll(torch.cat([instruct, role, prefix, icl], dim=1), trailing, tts_pad, max_new_tokens)
            results.append(
                utterance
                if utterance is not None
                else torch.zeros((0, talker_config.num_code_groups), dtype=torch.long, device=self.device)
            )
        return results

    def trim_anchor(self, waveforms: list[torch.Tensor], upsample_rate: int, anchor_num_frames=None):
        """Drop the anchor segment from decoded waveforms.

        Args:
            waveforms (`list[torch.Tensor]`):
                Waveforms decoded from the full frame sequence.
            upsample_rate (`int`):
                Waveform samples per codec frame.
            anchor_num_frames (`int` or `list[int]`, *optional*):
                Anchor length per waveform, as returned by
                [`~StableQwen3TTSForConditionalGeneration.generate`].

        Returns:
            `list[torch.Tensor]`: Waveforms carrying only the content segment.
        """
        if anchor_num_frames is None:
            anchor_num_frames = [self.config.anchor_num_frames] * len(waveforms)
        if isinstance(anchor_num_frames, int):
            anchor_num_frames = [anchor_num_frames] * len(waveforms)
        return [waveform[frames * upsample_rate :] for waveform, frames in zip(waveforms, anchor_num_frames)]

__all__ = ["StableQwen3TTSForConditionalGeneration"]
