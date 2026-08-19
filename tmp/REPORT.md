# Voice Design TTS — Gen-Gen Speaker Consistency 실험 리포트

## 0. 목표 · 설정 · 제약

- **모델**: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (talker-only, ~1.9B). Voice Design = persona(instruct 텍스트)로부터 음성 생성.
- **목표 지표**: **Gen-Gen COS** — 같은 persona에서 **독립적으로 두 번 생성**한 음성 간 화자 유사도 (ECAPA-TDNN cosine, `speechbrain/spkrec-ecapa-voxceleb`). `sim_mean`으로 표기.
- **부가 지표**: WER (내용 정확도, 낮을수록 좋음), UTMOS (음질, 높을수록 좋음), FFE.
- **핵심 제약 (반드시 지킴)**:
  1. **정상 샘플링 유지** (temperature 0.9, top_k 50). 샘플링을 제한하면 Voice Cloning과 달라지므로 금지.
  2. **WER 보존** (base ≈ 0.124). COS를 올려도 WER이 깨지면 무효.
- **참고 상한**: Voice Cloning의 Gen-Gen COS는 ~0.60 (Ref-Gen 0.90과 다름). 즉 목표 천장은 ~0.60.
- **데이터셋**: LibriTTS-P (df1 annotator), train-clean-100 기반 29,679 발화, persona = `combined_prompt`.experiment_qwen_vqr_kablation
- **메커니즘 (baseline)**: **Continuation / Anchor** — persona로 고정 문장 앵커를 1회 생성(seed 고정) → 그 앵커를 ICL reference로 넣고 실제 문장을 연속 생성. 앵커가 인스턴스별 음색을 pin해 Gen-Gen 일관성을 만든다.

---

## 1. 결과 총괄표 (전 실험)

WER-safe 기준 = WER ≲ 0.14. **★ = WER-safe SOTA.**

| # | 실험 | COS(sim) | WER | UTMOS | WER-safe | 비고 |
|---|------|:---:|:---:|:---:|:---:|------|
| **베이스라인 / 메커니즘** |
| B1 | two-pass sequential (cached, voice prompt 없음) | 0.402 | ~0.12 | — | ✅ | 원본 순차 |
| B2 | two-pass batched (cached) | 0.346 | 0.122 | 3.47 | ✅ | 배치화로 하락 |
| B3 | two-pass batched (no-cache) | 0.269 | 0.122 | 3.46 | ✅ | fresh ref |
| B4 | single-pass (cached, voice prompt 보임) | 0.348 | 0.127 | 3.39 | ✅ | |
| B5 | single-pass (no-cache) | 0.266 | 0.127 | 3.40 | ✅ | |
| **B6** | **single-pass no-cache fixedanchor** | **0.394** | **0.124** | **3.34** | ✅ | **WER-safe 기준 베이스** |
| **Fixed Reference Text Prompt 템플릿** |
| F1 | desc_raw | 0.391 | 0.128 | 3.34 | ✅ | |
| F2 | natural | 0.405 | 0.123 | 3.42 | ✅ | 템플릿 중 최고 |
| F3 | read_aloud | 0.385 | 0.125 | 3.45 | ✅ | |
| F4 | recording | 0.398 | 0.129 | 3.48 | ✅ | |
| **앵커 축 (학습 없음)** |
| A1 | 앵커 길이 tiny(2tok) | 0.336 | 0.124 | 3.32 | ✅ | 너무 짧으면 하락 |
| A2 | 길이 short(11) | 0.353 | 0.122 | 3.43 | ✅ | |
| A3 | 길이 long(25) | 0.368 | 0.119 | 3.52 | ✅ | |
| A4 | 길이 xlong(47) | 0.352 | 0.117 | 3.47 | ✅ | 포화·미세 하락 |
| A5 | greedy anchor (seed+greedy) | 0.408 | 0.131 | 3.29 | ✅ | 앵커 결정화 소폭 이득 |
| A6 | multi-ref M=1 | 0.394 | 0.124 | 3.34 | ✅ | =베이스 |
| A7 | multi-ref M=4 | 0.345 | 0.121 | 3.44 | ✅ | 다중 앵커 오히려 하락 |
| A8 | multi-ref M=8 (short) | 0.348 | 0.156 | 3.39 | ⚠️ | |
| A9 | temperature 0.9 | 0.408 | 0.131 | 3.29 | ✅ | |
| A10 | temperature 0.6 | 0.396 | 0.123 | 3.35 | ✅ | |
| A11 | temperature 0.3 | **0.696** | 0.194 | 3.13 | ❌ **무효** | 샘플링 제한 → 제약 위반 |
| **Learnable Global Query (파라미터화)** |
| G1 | raw (k=4, codec-CE만) | 0.392 | 0.604 | 2.72 | ❌ | off-manifold, WER 붕괴 |
| G2 | raw k=4 + sub-talker loss | 0.404 | 0.422 | 3.39 | ❌ | WER 여전히 붕괴 |
| G3 | global residual 벡터 | 0.350 | 0.117 | 3.43 | ✅ | WER-safe but 이득 없음 |
| G4 | softmax(vocab 볼록결합) | 0.479 | 0.314 | 3.27 | ❌ | 높은 COS, WER 붕괴 |
| G5 | VQ (k=8, ref_text 불일치) | **0.512** | 0.408 | 3.02 | ❌ | 최고 COS지만 WER 붕괴 |
| G6 | linear | 0.299 | 0.394 | 2.53 | ❌ | 전부 나쁨 |
| **VQ Learnable Global Query in Reftext (k-ablation) — ref_text 일치** |
| V1 | vqr k=1 | 0.318 | 0.157 | 3.37 | ⚠️ | |
| V2 | vqr k=4 | 0.347 | 0.131 | 3.25 | ✅ | |
| V3 | vqr k=8 | 0.382 | 0.137 | 3.04 | ✅ | |
| V4 | vqr k=16 | 0.347 | 0.132 | 3.26 | ✅ | |
| **V5 ★** | **vqr k=32** | **0.423** | **0.135** | **3.20** | ✅ | **WER-safe SOTA** |
| **Learnable Per-sample Query / Speaker-Embed Generator** |
| P1 | persona→query generator (gen_k16) | — | — | — | ❌ | collapse |
| P2 | speaker-embed gen (MLP, CE) | collapse | — | — | ❌ | per-persona cos 0.96–1.0 |
| P3 | + ECAPA anti-collapse head | 미미 | — | — | ❌ | λ=10에도 약함 |
| P4 | MiniLM persona encoder 주입 | 0.217 | — | — | ❌ | collapse는 풀렸으나 주입 실패 |
| P5 | 단순 Linear (frozen backbone, CE) | collapse | — | — | ❌ | div≈1.0 (아래 §7) |
| **모델 Fine-tuning / SFT** |
| S1 | LoRA reconstruction (초기, 레이아웃 오류) | 0.377 | 0.967 | 0.67 | ❌ | garbage |
| S2 | reconstruction SFT (streaming, light 150step) | 0.364 | 0.122 | 3.33 | ✅ | WER 유지, COS 이득 없음(↓) |
| S3 | ICL 연속화 SFT (v1 aggressive / v2 gentle) | — | broke | — | ❌ | 텍스트 무시 |
| S4 | joint (backbone LoRA + Linear, CE) | collapse | broke | — | ❌ | div 0.9987 |
| S5 | joint + contrastive spread (LAM=5) | 0.242 | 0.266 | 2.00 | ❌ | distinct code지만 off-manifold |

---

## 2. 베이스라인 · 메커니즘 (B1–B6)

**구현.** Continuation 메커니즘. Stage-1: `encode_voice_design(FIXED_CONTENT, persona)` → `model.generate`로 앵커 오디오 codec 생성. Stage-2: `generate_icl_prompt`로 `[instruct][role][codec prefix][ICL: ref_text+target_text ⊕ (codec_bos+앵커codec)]` 구성 → `talker.generate`로 타깃 문장을 앵커 음색으로 연속 생성.
- **cached**: 같은 voice prompt는 앵커 1개 공유. **no-cache**: 발화마다 fresh 앵커. **fixedanchor**: no-cache 구조지만 stage-1 앞에서 seed를 매번 리셋 → voice prompt별 앵커가 동일 (stage-2는 stochastic 유지).

**결과.**
- 원본 순차 two-pass = COS **0.402**. **배치화하면 0.346으로 하락** (bf16 + left-padding으로 batch≠batch1, bit-identical 아님 — 정상 동작이며 버그 아님).
- voice prompt를 시퀀스에 **보이게** 하고 앵커를 seed-lock한 **fixedanchor(B6) = 0.394 / WER 0.124 / UTMOS 3.34**. 이후 모든 학습 실험의 WER-safe 기준선.

**문제.** 앵커 메커니즘만으로 COS ~0.39–0.40에서 포화. VC의 0.60까지 0.2 갭.

---

## 3. Fixed Reference Text Prompt 실험 (F1–F4)

**동기.** 앵커를 만드는 **고정 문장(reference text)**의 문체/내용이 COS에 영향을 주는지.

**구현.** stage-1 앵커 생성 시 FIXED_CONTENT를 4가지 템플릿으로 교체:
- `desc_raw`: persona 설명 자체를 읽기, `natural`: 자연스러운 대화체, `read_aloud`: "다음을 낭독하세요" 지시형, `recording`: "이것은 녹음입니다" 형.

**결과.** 전부 COS 0.385–0.405, WER 0.123–0.129로 **거의 동일**. `natural`이 0.405로 미세 최고. **템플릿 선택은 COS에 유의미한 영향 없음.**

**문제/결론.** **Content 축은 (최소 길이만 확보되면) COS와 거의 무관.** COS Gen-Gen 측정 구간은 최종 target content로 생성된 구간이고, reference text는 앵커의 음색을 만드는 촉매일 뿐 → 문체를 바꿔도 음색 pin 강도는 안 변함. 이 실험이 "content 축은 막다른 길"임을 확정.

---

## 4. 앵커 축 실험 (A1–A11)

**앵커 길이 (A1–A4).** reference 문장 길이 2→11→25→47 토큰. COS 0.336→0.353→0.368→0.352. **최소 길이(~25토큰)까지는 오르고 이후 포화/미세 하락.** 너무 짧으면 음색 정보 부족.

**앵커 결정화 (A5 greedy).** stage-1을 greedy+seed로 완전 결정화 → 0.408 (베이스 0.394 대비 소폭↑). 앵커가 더 안정적이면 약간 이득.

**Multi-reference (A6–A8).** 앵커를 M개 생성해 pin 강화 시도. M=1(0.394) → M=4(0.345) → M=8(0.348). **오히려 하락.** 여러 앵커가 서로 다른 음색을 섞어 pin이 흐려짐.

**Temperature (A9–A11).** stage-2 temperature 0.9→0.6→0.3. COS 0.408→0.396→**0.696**. **temp 0.3에서 COS 0.696으로 급등하지만 WER도 0.194로 상승 + 샘플링 제한 = 제약 위반으로 무효.** Voice Cloning과 달라지므로 이 방향은 사용 불가 (사용자 명시).

**결론.** 앵커 축에서 짜낼 수 있는 이득은 greedy 결정화의 +0.014 수준. Multi-ref·temperature는 각각 하락/무효.

---

## 5. Learnable Global Query — 파라미터화 실험 (G1–G6)

**핵심 아이디어.** 앵커를 만드는 stage-1 프롬프트에서 **고정 reference text를 K개의 학습 가능한 query 토큰으로 대체**. `[instruct][role][codec prefix][QUERY(K) + eos][bos]` → 앵커 생성. Query는 teacher-forced reconstruction(codec-0 CE + 0.3·sub-talker CE)으로 최적화 → 앵커가 음색을 더 잘 pin하도록.
- **Global** = 모든 persona가 **동일한 query 공유** (persona-독립).

**파라미터화 (on-manifold 가설 검증).**
- `raw` (G1/G2): query = 자유 (K,D) 벡터. off-manifold.
- `softmax` (G4): query = softmax(logits)·PROJ, vocab 임베딩의 볼록결합 (soft on-manifold).
- `vq` (G5): STE로 가장 가까운 실제 vocab 토큰에 양자화 (hard on-manifold).
- `linear` (G6): 고정 토큰에 학습 Linear 변환.

**결과.**
- **COS는 오름**: softmax 0.479, **vq 0.512** (전 실험 최고 COS). on-manifold일수록 음색 pin 강함.
- **그러나 WER 붕괴**: raw 0.604, softmax 0.314, vq 0.408, linear 0.394. residual(G3)만 WER 0.117 유지하나 COS 0.350 (이득 없음).

**문제 (근본).** query로 만든 앵커는 강하게 음색을 pin하지만(COS↑), stage-2에서 **ref_text가 앵커와 불일치**하면 WER이 깨진다. 이 실험군은 ref_text = 고정 문장 "This is a short reference recording."을 썼는데, 앵커는 학습 query로 만들어졌으므로 **ref_text↔ref_audio 불일치** → ICL이 혼란 → WER 붕괴. 즉 **COS와 WER이 이 구현에서는 상충.** → §6에서 해결.

---

## 6. VQ Learnable Global Query in Reftext (V1–V5) — WER-safe SOTA ★

**§5의 WER 문제를 해결한 결정판.** VQ(G5)는 query를 **실제 vocab 토큰**으로 양자화하므로, 그 토큰을 **텍스트로 디코드**할 수 있다. 이 디코드된 텍스트를 stage-2의 **ref_text로 그대로 사용** → ref_text↔ref_audio **일치** → WER 보존.

**구현.**
- 학습: `lq_train3.py QTYPE=vq`로 K개 VQ query 토큰 학습, `token_ids` 저장 (`ckpt/query_vq_k{K}.pt`).
- 추론: stage-1 앵커를 VQ query로 생성 **+** `REF_TEXT = tokenizer.decode(token_ids)`를 stage-2 ref_text로 사용 (§5의 핵심 차이).
- k-ablation: K ∈ {1,4,8,16,32}.

**결과.**
| k | COS | WER | UTMOS |
|---|---|---|---|
| 1 | 0.318 | 0.157 | 3.37 |
| 4 | 0.347 | 0.131 | 3.25 |
| 8 | 0.382 | 0.137 | 3.04 |
| 16 | 0.347 | 0.132 | 3.26 |
| **32** | **0.423** | **0.135** | **3.20** |

- **k=32에서 COS 0.423 / WER 0.135 / UTMOS 3.20 = 현재까지의 WER-safe SOTA.** 베이스 0.394 대비 +0.029.
- k가 커질수록 COS↑ (query 용량↑ → 앵커 음색 pin 강화), WER은 0.13대 유지.

**문제/한계.** k=32에서도 0.42 — VC의 0.60까지 여전히 갭. UTMOS가 k와 함께 소폭 하락(3.34→3.20). Global query라 persona-독립 → persona별 음색 다양성은 못 살림(일관성만 강화). 더 키우면 UTMOS 열화 우려.

---

## 7. Learnable Per-sample Query / Speaker-Embed Generator (P1–P5)

**아이디어.** Global(§5–6) 대신 **persona별로 다른** query/embedding을 생성. `persona → (Linear/MLP/MiniLM) → speaker_embed`를 codec의 speaker 슬롯 또는 query 위치에 주입. teacher-forced reconstruction으로 학습.

**구현 변형.**
- P1 `gen_k16`: persona→(MLP)→K개 VQ query. P2: persona→(MLP)→speaker_embed. P3: +ECAPA anti-collapse head. P4: 입력을 MiniLM 임베딩으로. P5: 단순 Linear (frozen backbone).

**결과 — 전부 collapse.** per-persona speaker_embed 쌍별 cosine ≈ 0.96–1.0 (모든 persona가 사실상 같은 임베딩). P4(MiniLM)만 임베딩 다양성은 확보(cos 0.80)했으나 주입해 평가 시 **COS 0.217** (베이스보다 낮음).

**문제 (근본, 3중).**
1. **입력 비구분성**: persona_vec = mean-pool projected instruct → persona끼리 cos 0.96–0.99. 단순 map은 없는 구분성을 못 만듦.
2. **로스가 화자 구분 미요구**: codec-0 CE는 내용/운율 위주라 텍스트만으로 낮출 수 있음 → speaker_embed는 상수로 수렴.
3. **one-to-many**: persona가 목소리를 유일하게 결정 못 함 → 결정적 map은 실제 다양한 화자 평균으로 붕괴. (이게 Voice Design COS가 낮은 근본 원인 그 자체.)

→ **persona에서 나온 결정적 임베딩은 collapse(효과 없음)이거나, 억지 구분 시 off-manifold(해로움).**

---

## 8. 모델 Fine-tuning / SFT (S1–S5)

**동기.** 작은 학습 컴포넌트가 아니라 백본(talker)을 직접 SFT.

- **S1 초기 LoRA reconstruction**: 레이아웃 오류(비-streaming pad, truncated audio, speaker_embed 간섭)로 **garbage** (UTMOS 0.67, WER 0.97).
- **핵심 수정**: 실제 생성은 **streaming interleave** — 매 프레임 `codec_summed(C[t]) + trailing_text[t]`로 타깃 텍스트가 오디오와 1:1 삽입됨 (`modeling_qwen3_tts.py:1690`). 이걸 맞춰야 함. harness는 base로 검증(WER 0.08).
- **S2 reconstruction SFT (streaming, light 150 step, attn-only r=4)**: WER 0.122 보존하나 **COS 0.364 (베이스 0.394보다 소폭↓)**. 재구성은 consistency를 겨냥하는 목적이 아님.
- **S3 ICL 연속화 SFT**: 앵커 오디오를 ICL ref로 넣고 same-speaker 타깃 재구성. 학습 세게 하면 **teacher-forcing 텍스트 shortcut**(ground-truth 오디오 음향 연속성만으로 CE 최소화 → 텍스트 무시)에 빠져 WER 붕괴.
- **S4 joint (backbone LoRA + Linear, CE)**: speaker_embed div 0.9987 (collapse) + WER 붕괴.
- **S5 joint + contrastive spread (LAM=5)**: contrastive로 **collapse는 깨짐** (div 0.989→−0.08, persona별 distinct code 확보) + 6-샘플 WER gate 통과(0.124). 하지만 전체 eval에서 **COS 0.242 / WER 0.266 / UTMOS 2.00** — distinct code가 **arbitrary(음색 무관)**라 모델을 manifold 밖으로 밀어 품질·일관성 모두 하락.

**문제 (핵심 발견).** **CE 재구성 계열은 — probe든 backbone SFT든 joint든 contrastive든 — WER-safe Gen-Gen COS를 못 올린다.** 두 실패 모드:
- (a) CE는 화자 일관성/구분을 보상하지 않음 → 조건 임베딩 collapse.
- (b) 음색을 바꿀 만큼 세게 학습하면 teacher-forcing 텍스트 shortcut으로 WER 붕괴.

---

## 9. 종합 결론

1. **WER-safe SOTA = VQ Learnable Global Query in Reftext, k=32: COS 0.423 / WER 0.135 / UTMOS 3.20** (§6). 핵심은 (i) 앵커를 학습 query로 강화 + (ii) VQ on-manifold라 **디코드해 stage-2 ref_text로 재사용** → WER 보존.
2. **Content(reference text 문체) 축은 COS와 무관** (§3). 앵커 축은 greedy 결정화 +0.014 정도만 (§4).
3. **On-manifold(VQ/softmax)일수록 COS↑**, 하지만 **ref_text 불일치 시 WER 붕괴** (§5). VQ+reftext 매칭이 유일한 WER-safe 경로.
4. **Per-sample(persona→embedding) 계열은 원리적으로 collapse** — persona 비구분성 + CE의 무-요구 + one-to-many (§7).
5. **CE 기반 SFT는 백본을 열어도 COS를 못 올림** — collapse 또는 텍스트 shortcut (§8). Temperature(§4)는 올리지만 제약 위반.

**남은 정공법.** 일관성 = within-persona 생성 분산 감소인데, CE·persona-embedding·contrastive 모두 이를 직접 겨냥하지 않는다. 유일하게 남은 방향은 **실제 생성 오디오의 Gen-Gen COS를 직접 reward로 최적화** (decode 비미분 → RL/정책경사, 또는 미분가능 화자 proxy). VQ Global Query(0.42)를 초기값으로 이 방향을 얹는 것이 0.42→0.60 갭을 좁힐 유력 후보.

---

## 부록 — 재현 파일

- 학습: `/tmp/lq_train2.py`(query/residual), `lq_train3.py`(raw/softmax/vq/linear), `lq_train4.py`(per-sample generator), `lq_train5.py`(speaker-embed), `lq_train_lin.py`(Linear), `sft_icl2.py`(recon/ICL SFT), `sft_joint.py`, `sft_contrast.py`.
- 추론/평가 notebook: `experiment_qwen_continuation_singlepass_nocache_fixedanchor.ipynb`(베이스), `experiment_qwen_continuation_fixedprompt_ablation.ipynb`, `experiment_qwen_len_ablation.ipynb`, `experiment_qwen_greedyanchor.ipynb`, `experiment_qwen_multiref.ipynb`, `experiment_qwen_temp_sweep.ipynb`, `experiment_qwen_qvariants_infer.ipynb`, `experiment_qwen_vqr_kablation.ipynb`(SOTA), `experiment_qwen_spkgen*_infer.ipynb`, `experiment_qwen_sft_*_infer.ipynb`.
- 지표 원본: `results/_summary/*.json`.
