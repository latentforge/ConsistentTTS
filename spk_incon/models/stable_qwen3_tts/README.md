# Stable Qwen3-TTS

Migrated from [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), through the
`transformers` implementation in
[latentforge/transformers-tts](https://github.com/latentforge/transformers-tts)
(`transformers.models.qwen3_tts`).

`StableQwen3TTSForConditionalGeneration` extends `Qwen3TTSForConditionalGeneration` with a
learnable vector-quantized query that conditions acoustic anchoring, and generates the
anchor and the utterance in one autoregressive roll.

## Sequence layout

The query replaces the reference sentence in the text channel of the voice design prompt:

```
[ instruct ][ role ][ tts_pad + nothink, tts_pad + think_bos, tts_pad + think_eos, tts_bos + codec_pad ]
[ (query x k ++ tts_eos) + codec_pad ]
[ tts_pad + codec_bos ]
```

The text stream consumed by the decode loop delays the content until the anchor is laid
down:

```
trailing_text_hidden = [ tts_pad x (anchor_num_frames - 1) ] ++ [ content text ] ++ [ tts_eos ]
```

Frames `0 .. anchor_num_frames - 1` are generated with a padded text channel and fix the
speaker identity. The content text then streams in one entry per frame and the utterance
continues from the same key value cache. `trim_anchor` drops the anchor segment from the
decoded waveform.

## Query

`query` is a `(num_query_tokens, hidden_size)` parameter living in the talker hidden space
that `text_projection` maps into.

Quantization is a training time constraint. `quantize_query` snaps the parameter to the
nearest projected text embedding with a straight through estimator, so the value the
optimizer is scored on is always a real vocabulary entry and the parameter is held against
the manifold the talker was trained on. Generation takes a different path: it uses the
continuous `query` directly, so the query is never committed to a discrete token and keeps
resolution finer than the vocabulary provides.

The gap between the two only stays small while the parameter tracks the codebook. A run
whose parameter drifts far from its nearest entries has lost the constraint, and the
continuous query that generation would use no longer corresponds to what was optimized.
`query_token_ids` records the entries the parameter last snapped to, which makes that
distance measurable.

## Training

`forward` accepts `labels` of shape `(batch_size, sequence_length, num_code_groups)`
aligned with the input positions, `-100` marking positions to skip. The loss is the code
group 0 cross entropy plus `sub_talker_loss_weight` times the residual code group cross
entropy computed by `sub_talker_loss`.
