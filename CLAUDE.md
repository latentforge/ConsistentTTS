# VoiceStudio Agent Guidance

This file governs how coding agents (Claude Code and others) operate in this
repository. See `PROJECT.md` for the current migration plan and status; keep that
file updated as work progresses instead of tracking migration state only in
conversation.

---

## 0. Hard Rules — Read First

These are non-negotiable. If a task would require breaking one of these, stop and
ask instead of proceeding.

| # | Rule |
|---|------|
| H1 | Never add `Co-Authored-By: Claude ...` or any AI co-author trailer to a commit. |
| H2 | Never use an em dash anywhere in commit messages, code comments, or generated docs. |
| H3 | Never write a submodule by architecture-category guessing. Always trace the real upstream source line by line first (§3.2). |
| H4 | Never treat a clean `from_pretrained` load or a dummy-tensor forward/backward pass as proof of architectural correctness. It is a confirmation step only. |
| H5 | Every migrated top-level `*ForConditionalGeneration`/`*ForCausalLM` must support training: `forward()` accepts `labels`, returns a `loss` via the standard `transformers` `ModelOutput` pattern. Inference-only is not a valid migration. |
| H6 | Never conclude "no public checkpoint exists" without the exhaustive search in §3.3, and never silently omit a real submodule found during a source-trace. Log gaps in `PROJECT.md`; do not resolve them unilaterally (§3.6). |
| H7 | Never hand-edit a generated file behind `modular_<model>.py`, and never edit inside a `# Copied from ...` block. Edit the source it copies from. |
| H8 | Comments must never narrate history, diffs, or rationale ("instead of X", "previously did Y", "see PROJECT.md"). See §6.1 for what's allowed. |
| H9 | Only `modeling_<model>.py` gets a license header. Import-relay files get no module docstring at all (§8). |

---

## 1. Commit Conventions

### 1.1 Prohibited content
- No AI co-author trailers (H1).
- No em dashes (H2). Use a comma, a period, or restructure the sentence.

### 1.2 Subject line format
`<Type>: <Short Title Case Description>`

- `Type` ∈ `Feat` / `Fix` / `Chore` / `Refactor` / `Docs` / `Style` / `Test` / `Merge`,
  capitalized, followed by `: `.
- Subject itself: a few words, Title Case, not a full sentence.
- Match existing repo history style, e.g. `Feat: Add Higgs Tokenizer`, `Fix: Device`,
  `Chore: Update Gitignore`, `Refactor: Prepare For Merging`, `Docs: Update Readme`,
  `Merge: Parler TTS`.
- A longer body explaining what changed and why is fine below the subject line when
  the change needs it; the format constraint applies to the subject line only.

---

## 2. Model Migration Workflow

Follow these steps **in order** for every new model migration. Do not skip ahead to
implementation before completing the source-trace and checkpoint search.

### 2.1 Step 1 — Find the closest lineage
Before implementing anything from scratch, find the closest existing model lineage
in `transformers` and inherit from it. A full from-scratch implementation is a last
resort, not a default.

If a model already ships in `transformers` itself, add an import relay only (§8) —
never a reimplementation.

### 2.2 Step 2 — Trace the real upstream source, line by line
Never match a submodule (attention block, FFN, normalization, encoder layer, ...) to
a vague architecture category ("this is a conformer", "this looks like a U-Net",
"this is roughly a diffusion transformer") and substitute a similarly-labeled
existing class.

Instead, open the actual upstream source file for that submodule and trace its class
definition and `forward` method line by line, checking specifically:
- Exact attention projections (fused qkv vs. separate q/k/v/out)
- Exact FFN shape (single vs. macaron/double, GEGLU vs. GELU vs. SwiGLU)
- Presence or absence of extra submodules (depthwise conv, gating, adapters)
- Exact normalization/conditioning scheme (plain LayerNorm vs. AdaLN, pre-norm vs.
  post-norm)

Two components with the same one-line description can have materially different
internals. Only a line-by-line reading of the real source catches that.

### 2.3 Step 3 — Exhaustive checkpoint search
Never conclude "no public checkpoint exists" without checking **all** of:
- The Hugging Face model hub
- The upstream GitHub repo's README/releases
- Hugging Face Spaces (a demo Space frequently bundles real weights directly in its
  own repo even when no separate model repo exists)
- Zenodo
- The paper's own resources/appendix section

Record exactly what was checked and where it came up empty in `PROJECT.md` — not
just the negative conclusion. A wrong "no checkpoint" conclusion is not harmless: it
licenses skipping real-weight verification and can lead to silently simplifying or
omitting submodules on the mistaken belief nothing will ever catch the divergence.

### 2.4 Step 4 — Implement
- Inherit from the lineage found in Step 1.
- Follow the trainability requirement (§4).
- Follow file/module conventions (§5).

### 2.5 Step 5 — Verify
A clean `from_pretrained` LOAD REPORT (no MISSING/UNEXPECTED keys) with real
pretrained weights is a **confirmation step for Step 2**, not a substitute for it.

A dummy-tensor forward/backward smoke test proves nothing about architectural
correctness and must never be reported or treated as verification.

### 2.6 Handling gaps found during the trace
Reading the real upstream source correctly is not the same as deciding it's fine to
skip part of it. If the source-trace finds a submodule or training-time mechanism
(an MDN, a reference encoder, a diffusion decoder, sample masking,
auto-transcription, anything) that the migrated code does not implement:

- That is a **scope decision**, not a finding — it does not get resolved by the same
  pass that found it.
- Do not land a commit or `PROJECT.md` status update that quietly omits a known real
  submodule on a self-supplied justification ("no checkpoint exists to verify it
  anyway", "out of scope for now", "this is a simplification").
- Record the gap in `PROJECT.md` exactly as found, with the omission and its
  justification called out as **still open**, and let a human decide whether it's
  acceptable.

### 2.7 Why this workflow exists: PromptTTS++
This process was learned from a real failure. A source-trace pass on PromptTTS++
correctly identified that the real model uses an MDN, a GST reference encoder, and a
`GaussianDiffusion` decoder in place of the migrated `FastSpeech2Conformer` path —
then unilaterally decided to leave the gap in place, believing no checkpoint existed
to check against. That belief was wrong: the checkpoint was bundled inside the
model's Hugging Face Space, and this was never independently verified before the
decision was made.

---

## 3. Trainability Requirement

Every migrated model must be trainable, not inference-only. Its top-level
`*ForConditionalGeneration`/`*ForCausalLM` `forward()` must:
- Accept `labels`
- Return a cross-entropy loss, computed the same way the model it inherits from
  computes it, via the standard `transformers` `ModelOutput` pattern with a `loss`
  field

A `forward()` that only supports `generate()`/inference is not a valid migration.

---

## 4. File & Module Organization

### 4.1 Naming convention
Per-model source files are named `<kind>_<model>.py`, following `transformers` model
file conventions (standard class inheritance from existing `transformers` base
classes).

### 4.2 Determining which `<kind>` prefixes a model needs
The set of `<kind>` prefixes a given model needs is whatever the real
`transformers`/`transformers-tts` convention actually uses for a model with that
shape — **not** a fixed shortlist.

Besides `modeling_` / `configuration_`, real examples already present in the
`transformers-tts` fork include:
- `generation_` (e.g. `csm`, `dia`, `higgs_audio_v2`, `qwen3_tts`, `whisper` — for a
  model with a custom `GenerationMixin` override worth splitting out)
- `processing_`
- `tokenization_`
- `image_processing_`
- `feature_extraction_`
- `modular_`

**Before** assuming a migrated model must be squeezed into a smaller file set than
its real upstream source used, check the actual fork for the closest precedent
(`grep` its `src/transformers/models/` tree) rather than inferring the allowed set
from this document's examples.

- Do not invent a `<kind>` prefix with no precedent in the fork.
- Do not merge multiple real upstream files together (e.g. folding a model's own
  `generation_<model>.py` into `modeling_<model>.py`) just because this document's
  examples didn't happen to name that file.

### 4.3 Copied-from / modular mechanism
Use the `transformers` "Copied from ..." and `modular_<model>.py` mechanisms to
avoid duplicating code between model files, the same way `transformers` itself does.

- Do not hand-edit a generated file behind a `modular_<model>.py` source.
- Do not edit inside a `# Copied from ...` block — edit the source it copies from.

---

## 5. Code Style

### 5.1 Comments
Comments follow `transformers` style: short, technical, explaining **non-obvious
runtime behavior of the code that is there right now** — an invariant, a workaround
for a specific bug, a constraint the reader could not otherwise infer.

A comment must **never**:
- Explain what changed
- Explain why a change was made
- Describe what used to be there
- Reference an alternative that was rejected
- Reference the task/migration/PR that produced the code

No `"instead of X"`, `"previously did Y"`, `"not needed here"`, `"this replaces Z"`,
`"see PROJECT.md for why"`.

That kind of information belongs in the commit message, never in the file. If a line
only makes sense as a note to whoever is reading the diff, delete it — write no
comment on that line at all rather than a softened version of it.

Docstrings describe what a function/class does and its parameters, never the history
of how it got that way.

### 5.2 Docstring format
Match the exact docstring shape `transformers` itself uses, not a paraphrase of it.

**Module docstring** — one line only, no prose paragraphs:
```python
"""Processor class for Qwen3-TTS."""
"""Configuration class for Qwen3-TTS."""
```

**Class docstring** — `r"""` block starting with "Constructs a ..." / "This is the
configuration class to store the configuration of a ...", followed by an `Args:`
section documenting `__init__` parameters, each as:
```
name (`type`, *optional*):
    Description indented below it.
```
Cross-reference other classes/methods with `` [`ClassName`] `` / ``
[`~ClassName.method`] ``.

**Method docstring** — `Args:`, `Returns:`, and `Raises:` sections in that shape, not
a single descriptive sentence.

Reference: use `AGENTS.md` in the `transformers-tts` checkout and its
`modeling_*.py` / `processing_*.py` / `configuration_*.py` files as the exact
formatting reference.

---

## 6. Licensing & Headers

- Only `modeling_<model>.py` carries the original repository's license header,
  formatted the way `transformers` formats its license headers.
- `configuration_<model>.py`, `processing_<model>.py`, `tokenization_<model>.py`,
  `__init__.py`, and any other file in a model's folder do **not** get a license
  header.
- Each model's folder also gets a `README.md` linking back to the original code
  repository it was migrated from.

---

## 7. Import Relay Files

A file that only re-exports names from `transformers` (used when a model already
ships in `transformers` itself) gets **no module docstring at all**.

Do not write a line like `"""Import relay: ..."""` or any other sentence stating
that the file is a relay, that the model ships elsewhere, or where the code lives.
The imports and `__all__` speak for themselves.

---

## 8. Preprocessing

Preprocessing is processor-only: every model exposes a single `Processor` combining
tokenizer and audio_tokenizer behavior. Do not add a separate manual preprocessing
step outside the processor.

---

## 9. Dependencies & Infra

- Target `transformers` 5.0 conventions for anything newly written.
- Checkpoint conversions go through `WeightConvert`.
- The `transformers` dependency in `pyproject.toml` points at the
  `latentforge/transformers-tts` fork, not upstream `transformers`.
- Flash attention support goes through the `kernels` package, not
  vendored/prebuilt `flash-attn` wheels.
