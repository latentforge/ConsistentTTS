# Higgs TTS 3

Higgs TTS 3 pairs a plain [`Qwen3Model`] text backbone with a fused multi-codebook audio embedding and output head, replacing v2's custom Llama-derived dual-FFN decoder.
This implementation reuses `Qwen3Model` for the backbone and its `HiggsTTS2Embeddings`/`HiggsTTS2PreTrainedModel` for the fused audio embedding and shared `PreTrainedModel` plumbing, since those pieces are architecturally unchanged from v2.
The backbone itself (`HiggsTTS3Model`, `HiggsTTS3ForConditionalGeneration`) is written directly in this folder rather than inherited from `HiggsTTS3Model`, because v2's decoder layer bakes the dual-FFN audio/text routing into every layer, which v3 does not have; `HiggsTTS2Model` could not be reused as-is for the v3 backbone.

Original model and code: https://github.com/boson-ai/higgs-audio


## Usage

```python

```

```python

```
