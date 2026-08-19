# VoiceStudio Migration Project

This document records the plan to migrate VoiceStudio away from the namespace-package
architecture and into a single repository where every supported model is implemented
directly against the `transformers` model API. It exists so the plan survives context
resets across sessions and agents. Update it as work progresses; do not let it go stale.

## Workspace layout

The project checkout lives on the `C:` drive, which does not have room for model
checkpoints. `dep/` (vendored source clones) and `ckpts/` (downloaded checkpoints, used
only for local testing during migration) are relocated to `D:\VoiceWork`:

- `D:\VoiceWork\dep` — vendored source clones, linked at `dep/` in this repo.
- `D:\VoiceWork\ckpts` — downloaded checkpoints for local testing, linked at `ckpts/`.
- `D:\VoiceWork\worktree` — scratch location for any git worktrees created while
  working on a migration in isolation.

`ckpts/` is a temporary scratch folder, not an artifact of the migration. Everything
under it gets deleted once the full migration is complete; do not treat anything placed
there as something to preserve or commit.

Both `dep/` and `ckpts/` are gitignored.

### Downloading models for local testing

- Always download through `hf_xet` (the accelerated Hugging Face download path), never
  the default `huggingface_hub` cache resolution.
- Downloads must land under `ckpts/` (which resolves to `D:\VoiceWork\ckpts`), never the
  default `~/.cache/huggingface` location. Pass an explicit local target, do not rely on
  `HF_HOME`/cache defaults pointing there implicitly without checking.

## Migration order

1. Qwen3-TTS (depends on the `transformers-tts` merge, see below)
2. Parler-TTS
3. Everything else, in any order after that

## Folders explicitly out of scope

`voicestudio/models/stable_ommivoice/`, `voicestudio/models/stable_parler_tts/`, and
`voicestudio/models/stable_qwen3_tts/` are not part of this migration. Leave them
untouched.

## Background

VoiceStudio originally split each model's code into a separate namespace package
(`voicestudio-parler-tts`, `voicestudio-qwen3-tts`, etc.) to keep each original author's
code attributed and isolated. In practice this made PyPI packaging painful and spread
the codebase across too many repositories to maintain. We are reversing that decision:
model code moves back into this repository, with git history preserved, and is rewritten
to match `transformers` conventions instead of being wrapped.

## Per-model migration procedure

For every model in scope, follow these steps in order:

1. **Vendor the source.** Clone the model's official code repo (or the code shipped in
   its Hugging Face model repo, if that's the only source) into `dep/`.
2. **Merge history into the model's folder.**
   - If an existing namespace implementation already exists under `voicestudio/models/`,
     keep that as the base.
   - Otherwise, merge the cloned repo's git history directly into the target model code
     folder (e.g. via `git subtree`/`git filter-repo` + merge, not a fresh copy) so
     authorship and history are preserved.
3. **Rebase onto the closest transformers model.** Analyze the model architecture, find
   the most structurally similar model already in `transformers`, and inherit from it
   instead of writing the model from scratch. Do not generate a full model implementation
   when a close relative already exists in the library.
4. **Add a README.md** in the model's folder linking back to the original code repository.
5. **Add license headers.** Every source file carries the original repo's license notice,
   formatted the way `transformers` formats its license headers.
6. **Delete the vendored copy** from `dep/` once the model's migration is complete and
   verified.

## Repos to fully migrate then delete

- https://github.com/latentforge/higgs-audio
- https://github.com/latentforge/parler-tts
- https://github.com/latentforge/Qwen3-TTS
- https://github.com/latentforge/CosyVoice
- https://github.com/latentforge/Chroma
- https://github.com/latentforge/Spark-TTS
- https://github.com/latentforge/dia
- https://github.com/latentforge/F5-TTS
- https://github.com/latentforge/promptttspp

Each of these gets deleted from GitHub only after its migration is complete, verified,
and the vendored copy is removed from `dep/`.

## Dependencies to remove

- **https://github.com/latentforge/audiotools** — analyze what depends on it, remove the
  dependency, then delete the repo.
- **https://github.com/latentforge/vocos** — analyze what depends on it, remove the
  dependency, then delete the repo.
- **https://github.com/latentforge/speechbrain** — this fork exists only to support a
  newer torch version. If upstream speechbrain has since caught up, drop the fork and
  depend on upstream (or drop the dependency entirely if unused after migration); then
  delete the fork.
- **https://github.com/sarulab-speech/UTMOSv2** — decouple via the `evaluate` library
  using https://huggingface.co/spaces/sarulab-speech/UTMOSv2/ as the reference
  implementation. A new `voicestudio/metrics/` folder may be created if needed for this.

## Rules for the rewritten model code

- Follow `transformers` model file conventions strictly: `modeling_<model>.py`,
  `configuration_<model>.py`, standard class inheritance, etc. Do not create files that
  fall outside the `transformers` model layout.
- Where a model already exists in `transformers` itself, only do an import relay — do
  not reimplement it.
- Use `WeightConvert` for any checkpoint conversion work.
- Comments follow `transformers` style: terse and technical, never narrated like a diary
  entry.
- Target `transformers` 5.0 conventions.
- Before writing a model from scratch, find the closest existing model lineage in
  `transformers` and inherit from it.
- Follow the `transformers` "Copied from" / `modular_<model>.py` conventions used
  upstream (see `transformers-tts/.ai/AGENTS.md`) wherever they apply to reduce
  duplication between model files.

## Processor standard

New models must use the tokenizer + audio_tokenizer processor pattern, i.e. all
preprocessing goes through the model's `Processor` — no separate manual preprocessing
step. For audio tokenizer models, follow the Qwen3-TTS and Higgs v2 examples. For
Parler, switch to the `dac` implementation already registered in `transformers` instead
of vendoring DAC.

Target usage shape:

```python
model_id = "eustlb/higgs-audio-v2-generation-3B-base"
model = HiggsAudioForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id).to(device)

outputs = model.generate(**inputs)

audio_values, sr = processor.decode(outputs)
```

### Qwen3-TTS specifics

Drop the existing VoiceStudio Qwen3-TTS implementation and depend on the version being
merged into `transformers-tts` instead. Preserve the VoiceStudio `Qwen3TTSProcessor`
task-dispatch behavior if the incoming `transformers-tts` processor implementation is
missing it:

- `processor.encode`: accepts all parameters; raises a runtime error if the task implied
  by the given arguments doesn't match the model's configured task.
  and only allows the arguments valid for that task).
- `processor.encode_<task>` (e.g. `encode_voice_design`): only accepts the arguments
  valid for that specific task.

If the incoming processor is missing this behavior, subclass/extend it inside
VoiceStudio to restore it — don't fork the whole processor.

## Packaging changes

- Remove all namespace-package wiring described in the old README/pyproject (the
  commented-out `voicestudio-*` package entries, the split-repo installation model).
- Flash attention: depend on the `kernels` package (the current standard) instead of
  building/vendoring `flash-attn` wheels.
- Pin the `transformers` dependency to the `transformers-tts` fork
  (https://github.com/latentforge/transformers-tts), not upstream `transformers` or the
  ShahVandit fork.

## Status tracking

Update this table as each model's migration lands.

| Model | Status | Notes |
|---|---|---|
| Qwen3-TTS | In progress | Model/config are import relays to transformers-tts. Processor subclass adds `encode`/`encode_voice_design`/`encode_custom_voice` task dispatch with `RuntimeError` on task mismatch. `encode_voice_clone` raises `NotImplementedError`: transformers-tts's `Qwen3TTSProcessor` has no reference-audio input path yet. |
| Parler-TTS | Not started | Second in migration order. Target: use HF-registered `dac`, closest transformers lineage TBD |
| Higgs-Audio (v2/v3) | Not started | Reference processor pattern for other audio-tokenizer models |
| Chroma | In progress | Backbone/decoder/generation loop reimplemented against transformers-tts's Llama, Qwen2.5-Omni thinker, and Mimi codec classes (no full Chroma architecture exists upstream in transformers-tts to relay to). Processor subclasses `Qwen2_5OmniProcessor` to add the reference-audio voice-cloning prompt. |
| Spark-TTS | Not started | Already vendored in `dep/Spark-TTS` |
| Dia | Not started | Already marked "fully tested (by HF)" in old README |
| CosyVoice (v1/v2/v3) | Not started | Already vendored in `dep/CosyVoice` |
| F5-TTS | Not started | Already vendored in `dep/F5-TTS` |
| promptttspp | Not started | |
| audiotools dependency removal | Not started | |
| vocos dependency removal | Not started | |
| speechbrain fork removal | Not started | Check if upstream now supports the required torch version |
| UTMOSv2 decoupling | Not started | Route through `evaluate`, may need `voicestudio/metrics/` |
