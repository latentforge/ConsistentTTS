---
name: chaeunize
description: Write or revise an academic paper in this author's voice. Use whenever drafting, rewriting, or reviewing any part of a manuscript (title, abstract, introduction, related work, method, experiments, limitations, conclusion, figure captions), or when asked to make prose "sound like my papers" or match the lab's writing style. Encodes the argumentative spine, section templates, sentence constructions, and phrase bank extracted from the author's accepted papers.
---

# Chaeunize

이 저자의 논문 작성 방식을 재현 가능한 형태로 고정한 스킬이다. 문장을 새로 고안하기 전에
여기에 대응하는 패턴이 있는지 먼저 확인하고, 있으면 그것을 쓴다.

## 근거 자료

| 약칭 | 논문 | 특성 |
| --- | --- | --- |
| **G** | GITA: Input-level Test-time Adaptive Object Detection for Stable Long-horizon Adaptation under Continual Weather Domain Shifts. WACV 2027 Submission #1993, Algorithms Track | 국제 학회, 8p + 부록, 명명된 시스템, 방법 논문 |
| **K** | Analyzing and Improving Voice Consistency in Voice Design TTS. KCC 2026 | 국내 학회, 3p, 명명된 산출물 없음, 분석 + 완화 논문 |

인용된 영문은 전부 원문이다. `(G)` 또는 `(K)`로 출처를 표시한다.

---

# 제1부. 논증의 골격

개별 문장을 다듬기 전에 이 골격이 서 있는지 먼저 확인한다. 두 논문이 동일한 골격을 갖는다.

```
1  현실 배치 상황에서 문제가 발생한다
2  기존 계열이 그 문제를 다루고 있다
3  그러나 특정 조건에서 실패한다
4  그 실패는 우연이 아니라 구조적이다              <- 축
5  실패가 구조적이면 처방도 구조적이어야 한다
6  그래서 문제를 다른 공간 또는 다른 층위로 옮긴다   <- 기여
7  대가가 있으면 그 자리에서 말한다
8  옮긴 것이 최적이라 주장하지 않는다               <- 겸양
```

## 1.1 4번이 서명이다

문제를 개별 방법의 결함이 아니라 **그 방법 계열의 구조적 성질**로 격상시킨다. 이것이 처방의
필요성을 만든다. 이 격상이 없으면 논문은 "우리 방법이 더 좋다"로 축소된다.

> "This feedback loop reflects **not an incidental implementation detail but a structural
> property** of parameter-update adaptation" (G)

> "confirming that this drift is **structural rather than specific to any single method or
> architecture**." (G)

> "**We argue that the root cause is architectural.**" (K)

> "places Voice Design in a **structurally similar position** to pre-LM voice cloning approaches"
> (K)

**격상의 세 수순.**

1. 실패를 관측한다.
2. 그 실패가 계열의 어떤 성질에서 필연적으로 따라 나오는지 기제로 설명한다.
   `because [계열이 하는 일], [작은 오차]가 [큰 실패]로 누적된다`
3. 개별 방법의 문제가 아님을 실증하거나 유비로 보인다.
   - 실증: 여러 baseline이 같이 무너짐을 보인다 (G, Fig. 2)
   - 유비: 과거에 같은 구조를 가졌던 계열이 같은 문제를 겪었음을 든다 (K, pre-LM cloning)

## 1.2 4번을 세운 뒤에는 반드시 5번으로 간다

```
Regularization-based remedies manage this instability after it arises
rather than removing it at its source. We instead eliminate instability
by construction ... (G)
```

**증상 관리와 원인 제거를 대비시킨다.** 기존 처방을 "사후 관리"로, 우리 처방을 "설계상 보장"으로
위치시킨다.

> "This is **a structural guarantee** on the adaptation space, **not a soft preference layered on
> top**." (G)

## 1.3 8번을 빠뜨리지 않는다

주장을 세운 직후 같은 문단 안에서 범위를 좁힌다.

> "we view GITA as **a minimal demonstration of this idea rather than an optimal or exhaustive
> solution**." (G)

> "We view this as **a first step toward** principled consistency modeling in Voice Design TTS"
> (K)

---

# 제2부. 어투

## 2.1 대조 구성 (최우선 규칙)

**이 저자의 문장은 대조로 정의된다. 새 개념을 대조항 없이 도입하지 않는다.**

| 형태 | 원문 |
| --- | --- |
| `A rather than B` | "reframes instability as a problem to be addressed by design **rather than** managed through regularization" (G) |
| | "restricting adaptation to the low-dimensional, bounded input space **rather than** the detector's parameter space" (G) |
| | "conditioned synthesis on abstract speaker representations **rather than** raw audio context" (K) |
| `A rather than merely B` | "the optimization problem itself can be constrained **rather than merely** regularized" (G) |
| `not A but B` | "reflects **not** an incidental implementation detail **but** a structural property" (G) |
| `A, not B` | "a structural guarantee on the adaptation space, **not** a soft preference layered on top" (G) |
| `Unlike A, which ..., B ...` | "**Unlike** reference audio, **which** encodes a specific acoustic identity, textual descriptions such as "young female voice" covers a wide region of the acoustic space" (K) |
| `By comparison, ...` | "**By comparison,** Voice Design models infer speaker identity from text alone" (K) |
| `In contrast, ...` | "**In contrast,** our method operates purely in the input space" (G) |
| `While A, B` | "**While** Ref-Gen measures how well a synthesized sample matches a given reference recording, Voice Design operates without any canonical reference utterance." (K) |
| `X differ in where` | "These methods differ in **where** the adaptation acts, on an *internal* feature representation or directly on the *raw input*" (G) |
| `fundamentally different from` | "This problem is **fundamentally different from** the reference-to-generation (Ref-Gen) evaluation protocol" (K) |

**작성 규칙.** 개념을 도입하는 문장을 쓸 때마다 다음을 자문한다. *무엇과 대비되는가.* 답이 없으면
그 문장은 아직 이 저자의 문장이 아니다.

## 2.2 현상에 이름을 붙인다

관측한 현상에 이름을 주고 그 이후로 그 이름을 쓴다.

> "We refer to this phenomenon as ***generation-to-generation (Gen-Gen) inconsistency***." (K)

> "Building on this principle, **which we refer to as Input-level Test-time Adaptation**, we
> instantiate a simple and practical realization, GITA" (G)

형식은 `We refer to this [phenomenon | principle] as **X**` 이고, 이름은 굵게 또는 기울여 표시한다.

## 2.3 핵심 어휘

| 어휘 | 용법 |
| --- | --- |
| `structural`, `architectural` | 문제를 계열의 성질로 격상. 이 저자의 축 |
| `by construction` | 처방이 사후 관리가 아니라 설계상 보장임 |
| `at its source` | 원인 제거와 증상 관리의 대비 |
| `structurally precluded` | 문제 발생 여지 자체가 없음 |
| `inductive bias` | 구조가 자연히 부여하는 이점 (K) |
| `underspecified`, `underspecification` | 조건이 대상을 충분히 좁히지 못함 (K) |
| `bounded`, `low-dimensional` | 제약된 공간의 성질 |
| `agnostic` | 적용 범위의 넓음. `detector-agnostic`, `architecture-agnostic` |
| `instantiate` | 원리에서 구현체로 내려갈 때 |
| `anchor` | 무언가를 고정하는 역할 |
| `drift`, `collapse`, `accumulate` | 점진적 실패의 어휘 |
| `regime` | 관측된 구간의 구조. "indicate two regimes" (K) |
| `minimal demonstration`, `first step toward` | 겸양 |

## 2.4 강조 부사

문단에서 **가장 중요한 문장 하나**를 표시한다. 한 문단에 하나만 쓴다.

- `Crucially,` "**Crucially,** this instability is not merely hypothetical" (G)
- `Crucially,` "**Crucially,** $\gamma$ is the only learnable parameter in GITA" (G)
- `Notably,` "**Notably,** most baselines apply only to a subset of detectors" (G)
- `Moreover,` 추가 근거
- `Specifically,` 방금 말한 것의 구체화

## 2.5 전환 어구

| 위치 | 어구 |
| --- | --- |
| 선행의 한계 | `However,` / `Despite this flexibility,` (K) |
| 관측에서 설계로 | `These observations motivate ...` (G) / `Based on these observations, ...` (K) / `Motivated by this finding,` (G) |
| 원리에서 구현으로 | `Building on this principle, ...` (G) |
| 대응 착수 | `To address this, we introduce ...` (K) / `To mitigate this gap, we propose ...` (K) |
| 결과 제시 | `As shown in Table 1, ...` / `The result, shown in Fig. 2, is ...` (G) |
| 결과 해석 | `These results indicate that ...` / `These dynamics indicate two regimes.` (K) |
| 대비되는 결과 | `Clear differences emerge after ...` (G) |

## 2.6 인칭, 시제, 문장 길이

- **1인칭 복수를 적극적으로 쓴다.** `We consider`, `We argue that`, `We instantiate`,
  `We interpret`, `We observe`, `We hypothesize`, `We treat this not as ... but as ...`,
  `We note, however, that ...`, `We view this as ...`
- 수행한 일과 관측 결과는 현재형으로 쓴다.
- **긴 문장을 쓰되 핵심 주장은 짧게 끊는다.**
  > "This is a structural guarantee on the adaptation space, not a soft preference layered on top."
- 종속절로 조건과 기제를 한 문장 안에 담는다.
  > "As shown in Table 1, although Voice Design models remain competitive in naturalness, their
  > Gen-Gen cosine similarity is markedly lower than the voice cloning baseline, indicating severe
  > speaker drift across generations." (K)
  >
  > `As shown in [표], although [양보], [발견], indicating [함의].` 네 성분이 한 문장에 들어간다.

## 2.7 추론의 강도를 표시한다

검증되지 않은 것을 검증된 것처럼 쓰지 않는다.

| 강도 | 표지 |
| --- | --- |
| 관측 | `We observe that ...` (K) |
| 논증 | `We argue that ...` (G, K) |
| 가설 | `We hypothesize this dedicated sink reduces content leakage ...` (K) |
| 해석 | `We interpret this blending as an identity-preserving residual connection` (G) |
| 귀속 | `we attribute this to Enc-Dec models' architectural constraint` (K) |
| 시사 | `These results suggest that ...` (G) |

---

# 제3부. 제목

## 3.1 명명된 산출물이 있을 때

**`이름: 무엇 + 무엇을 위해 + 어떤 조건에서`**

> **GITA: Input-level Test-time Adaptive Object Detection for Stable Long-horizon Adaptation
> under Continual Weather Domain Shifts** (G)

| 성분 | 내용 |
| --- | --- |
| 이름 | `GITA` |
| 무엇인가 | `Input-level Test-time Adaptive Object Detection` |
| 무엇을 위해 | `for Stable Long-horizon Adaptation` |
| 어떤 조건에서 | `under Continual Weather Domain Shifts` |

## 3.2 명명된 산출물이 없을 때

**`동명사 쌍 + 대상 + 범위`**

> **Analyzing and Improving Voice Consistency in Voice Design TTS** (K)

`Analyzing and Improving`처럼 **진단과 처방을 동명사 쌍으로 병치**한다. 논문이 두 일을 한다는 것이
제목에서 드러난다.

## 3.3 공통 규칙

1. **명사구로 끝난다.** 주절에 동사를 두지 않는다.
2. **선언문을 제목으로 쓰지 않는다.** `X Is Y` 형태를 쓰지 않는다.
3. 조건절 `under ...` 또는 `in ...`으로 적용 범위를 한정한다.
4. 약어는 본문에서 한 번 확장한다. `GITA (Gamma-based Intensity Transformation for Adaptation)`

---

# 제4부. Abstract

## 4.1 공통 골격

두 논문이 같은 순서를 따른다. 문장 수만 다르다.

```
1  배치 상황과 최근 흐름
2  기존 계열 또는 새 패러다임이 무엇을 하는가
3  However + 실패 + 그 기제                     <- as 절 또는 종속절로 기제를 붙인다
4  그래서 우리가 제안하는 원리 또는 분석
5  그 원리의 구현체 또는 처방
6  결과
7  대가 또는 유지된 성질                          <- 반드시 인접 배치
8  겸손한 마무리 명사구
```

## 4.2 원문 대조

| 수 | GITA (6문장, 약 150어) | KCC (9문장, 약 200어) |
| --- | --- | --- |
| 1 | "Object detectors deployed in real-world environments often suffer from performance degradation under continual weather-induced domain shifts." | "Recent advances in text-to-speech (TTS) have been driven by language model (LM)-based architectures, enabling high-quality zero-shot voice cloning through continuation mechanisms where reference audio naturally guides generation." |
| 2 | "Continual Test-time Adaptation (CTTA) addresses this challenge by adapting models online using an unlabeled target stream." | "From this foundation, Voice Design TTS has emerged, replacing reference audio with natural language descriptions to enable speech generation without any audio prompt." |
| 3 | "**However**, existing CTTA approaches can degrade sharply or even collapse under long-horizon adaptation, **as** continually updating a high-dimensional parameter space based on noisy, unlabeled feedback allows small misalignments to accumulate into optimization drift over time." | "**However**, despite this flexibility, Voice Design models **lack the continuation-based inductive bias** that naturally facilitates consistent voice modeling, leaving generation coherence largely unaddressed." |
| 4 | "**Motivated by this finding, we propose** Input-level Adaptation as a design principle that reframes instability as a problem to be addressed by design rather than managed through regularization, by restricting adaptation to the low-dimensional, bounded input space rather than the detector's parameter space." | "**In this paper, we analyze** this limitation through a generation-to-generation (Gen-Gen) evaluation protocol **and show that** Voice Design systems exhibit substantially lower speaker consistency than zero-shot cloning models." |
| 5 | "**We instantiate this principle with** GITA (Gamma-based Intensity Transformation for Adaptation), a lightweight learnable input transformation that combines histogram stretching and gamma correction for detector-agnostic adaptation, while the entire detector remains frozen." | "**To mitigate this gap, we propose** a data reconstruction approach at the voice profile level, with a dedicated consistency token to stabilize attention." |
| 6 | "Experiments on SHIFT and CityScapes-Corrupted demonstrate consistent performance improvements across multiple detectors under both discrete and continuous domain shifts," | "Our approach improves Gen-Gen cosine similarity **from 0.22 to 0.40** in the target setting, corresponding to an **82% relative gain**." |
| 7 | "**while maintaining** long-horizon stability and computational efficiency." | "**At the same time, the method reveals a clear trade-off** with content fidelity, reflected in higher word error rate (WER)." |
| 8 | (없음) | "These findings provide **an initial quantitative foundation** for studying speaker consistency in Voice Design TTS." |

## 4.3 3번 문장이 초록의 무게중심이다

**실패를 보고하는 데 그치지 않고 그 자리에서 기제를 붙인다.**

- 기제형: `as [주체]가 [조건]에서 [행위]하면 [작은 오차]가 [큰 실패]로 누적된다` (G)
- 결여형: `lack the [X] that naturally facilitates [Y], leaving [Z] unaddressed` (K)

두 형태 중 하나를 쓴다. 실패만 서술하고 넘어가지 않는다.

## 4.4 수치를 넣을지 결정하는 기준

**GITA는 넣지 않고 KCC는 넣는다. 규칙은 다음과 같다.**

| 조건 | 처리 |
| --- | --- |
| 주장이 **정성적 성질**(안정성, 일관성 유지, 범용성)인 경우 | 수치를 넣지 않고 `consistent improvements`, `while maintaining ...`으로 쓴다 |
| 주장이 **개선의 크기 자체**인 경우 | 헤드라인 수치를 넣는다. `from 0.22 to 0.40`, `an 82% relative gain` |

**수치를 넣었으면 반드시 대가를 인접 문장에 쓴다.** KCC가 개선 수치 바로 다음 문장에서
`At the same time, the method reveals a clear trade-off with content fidelity`로 대가를 밝힌다.

## 4.5 마무리 명사구

두 논문 모두 `These [results | findings] [suggest | provide] [겸손한 명사구]` 로 닫는다.

> "These results **suggest that** constraining the adaptation space is **a simple and practical
> path toward** reliable source-free perception in changing environments." (G)

> "These findings **provide an initial quantitative foundation for** studying speaker consistency
> in Voice Design TTS." (K)

**겸손한 명사구 목록**: `a simple and practical path toward`, `an initial quantitative foundation
for`, `a first step toward`, `a minimal demonstration of`

---

# 제5부. Introduction

## 5.1 다섯 문단 구성

| 문단 | 역할 |
| --- | --- |
| 1 | 깔때기. 일반 상황에서 이 논문이 서는 지점까지 좁힌다 |
| 2 | 관측된 문제. 현상을 서술하고 이름을 붙인다 |
| 3 | 인접 개념과의 구별. 왜 기존 틀로는 다룰 수 없는가 |
| 4 | 구조적 격상과 전환. 우리 원리를 세운다 |
| 5 | 기여 |

긴 논문(G)에서는 2번과 3번이 각각 선행 계열 하나씩을 다루는 형태로 확장된다.

## 5.2 문단 1. 깔때기

```
일반적 문제 또는 최근 성과 [refs]
  -> 특정 조건에서 더 심각해지거나, 새 패러다임이 등장한다 [refs]
  -> 실제 응용이 그 조건을 예시한다
  -> 따라서 어떤 능력이 필수적이다
  -> Consequently, [기존 계열]이 등장했다 [refs]
```

**구체적 예시를 인용부호로 넣는다.**

> "produces speech directly from natural-language descriptions such as **"a calm middle-aged
> female voice"** or **"an energetic male narrator"**, without requiring reference audio" (K)

## 5.3 문단 2. 관측과 명명

```
Despite this [장점], we observe that [문제].
When [조건이 바뀌면], [현상].
We refer to this phenomenon as **[이름]**.
```

## 5.4 문단 3. 인접 개념과의 구별

```
This problem is fundamentally different from [인접 개념].
While [인접 개념]은 [무엇]을 한다, [우리 상황]은 [그것이 성립하지 않는다].
The more relevant question is therefore whether [진짜 질문].
```

**기울임으로 핵심어를 강조한다.**

> "whether the model can repeatedly instantiate the *same* latent speaker identity from the
> *same* textual voice description." (K)

## 5.5 문단 4. 구조적 격상과 전환

**논문에서 수사의문문이 허용되는 유일한 자리다. 한 편에 한 번만 쓴다.**

> "These observations motivate a different design question: rather than searching for a better
> way to update detector parameters online, **can we instead adapt in a space where instability
> is structurally precluded by design?**" (G)

이어서 답하고, 원리에 이름을 붙이고, 구현체를 소개하고, 중심 주장을 명시한 뒤 겸양으로 닫는다.

> "**Our central claim is that** constraining adaptation to an input space is, by construction,
> more resistant to long-horizon drift than adapting in parameter space, **and we view GITA as a
> minimal demonstration of this idea rather than an optimal or exhaustive solution**." (G)

짧은 논문에서는 수사의문문 없이 단정으로 간다.

> "**We argue that the root cause is architectural.**" (K)

## 5.6 문단 5. 기여

`Our contributions are summarized as follows:` (G) 또는 `Based on these observations, our
contributions are:` (K) 다음 불릿 목록으로 쓴다.

**구성은 [진단] / [제안] / [검증 또는 추가 분석] 을 축으로 한다.** 논문이 그 이상을 담으면 항목이
늘어난다. 항목 수를 맞추기 위해 성격이 다른 기여를 한 항목에 묶지 않는다.

| 순서 | GITA | KCC |
| --- | --- | --- |
| 1 진단 | "**We identify**, through a long-horizon evaluation protocol, a structural drift problem in parameter-update CTTA, where adapting a high-dimensional parameter space from noisy, unlabeled feedback accumulates into optimization drift and eventual collapse as online adaptation proceeds." | "**We analyze** the architectural reason why Voice Design TTS is inherently more vulnerable to speaker drift, grounded in the absence of continuation mechanisms." |
| 2 제안 | "**We propose**, to the best of our knowledge, the first *Input-level Adaptation* for CTTA, a design principle that is stable by construction, restricting adaptation to a low-dimensional, bounded transformation of the raw input rather than the detector's parameters." | "**We propose** a mitigation based on voice profile-level training data reconstruction and attention sink stabilization." |
| 3 검증 | "**Extensive experiments** in a continual setting on SHIFT and CityScapes-Corrupted demonstrate consistent performance improvements under discrete and continuous weather shifts across one-stage and two-stage detectors with both CNN and Transformer backbones, enabling architecture-agnostic adaptation while maintaining long-horizon stability and computational efficiency." | "**We analyze** the trade-off between voice consistency and attention efficiency via training dynamics behind this approach." |

**주목.** KCC의 3번은 실험이 아니라 **트레이드오프 분석**이다. 대가를 기여로 내건다. 개선만
기여로 삼지 않는 것이 이 저자의 특징이다.

`to the best of our knowledge, the first` 는 2번에만, 그리고 실제로 처음일 때만 쓴다.

---

# 제6부. Related Work

## 6.1 구성

주제별 하위 절 두세 개. 각 절은 서술하고 **`However,` 로 한계를 지적하며 닫는다.**

| GITA | KCC |
| --- | --- |
| 2.1 Test-time Adaptation (TTA) | 2.1 Evolution of Zero-shot TTS Architectures |
| 2.2 Test-time Adaptive Object Detection | 2.2 Voice Design TTS |
| 2.3 Adaptation at Different Representational Levels: Input-level vs Internal-representation | 2.3 Speaker Consistency Evaluation |

**마지막 절이 본 논문의 위치를 결정하는 절이다.** 여기에 가장 가까운 선행을 배치한다.

- G의 2.3은 `A vs B` 형식으로 우리와 선행이 갈리는 축 자체를 제목에 넣는다.
- K의 2.3은 평가 관행을 다루고 `yet explicit speaker consistency modeling in Voice Design remains
  unexplored.` 로 공백을 확정한다.

## 6.2 절을 닫는 문장

> "However, these approaches rely on architecture-specific internal representations, ... These
> limitations motivate the development of stable, efficient, and architecture-agnostic
> frameworks." (G)

> "However, existing work has focused primarily on achieving target voice characteristics in
> single utterances, measured through Ref-Gen similarity on annotated test sets. Comparatively
> little attention has been paid to whether the same description leads to a stable speaker
> identity across multiple utterances." (K)

`Comparatively little attention has been paid to ...` 는 공백을 지적하는 표준 어구다.

## 6.3 우리 위치의 확정

마지막 절 끝에서 우리가 무엇을 다르게 하는지 한 문장으로 밝힌다.

> "**In contrast, we aim to** adapt the raw input only, fully online and source-free, without
> target domain knowledge." (G)

---

# 제7부. Method

## 7.1 절 구성

| GITA | KCC |
| --- | --- |
| 4.1 Problem Formulation | 3.1 Problem: The Absence of Continuation Mechanism |
| 4.2 Design Principle: Stability by Construction | 3.2 Data Reconstruction for Explicit Voice Modeling |
| 4.3 Overview of GITA: A Simple Instantiation | 3.3 Attention Sink Stabilization |
| 4.4 Input Transformation | |
| 4.5 Stability-preserving Residual Blending | |
| 4.6 Feature Statistics Alignment | |

## 7.2 규칙

1. **첫 절은 문제다.** 표기를 도입하는 `Problem Formulation`이거나, 문제를 명명하는
   `Problem: [명사구]` 다.
2. **둘째 절은 설계 원리다.** `Design Principle: [한 구절 요약]` 형식. 여기서 원리를 세우고
   기존 처방과 대비시킨다. 구현 세부는 넣지 않는다.
3. **셋째 절은 개요다.** `Overview of [이름]: A Simple Instantiation`. 전체 파이프라인을 서술하고
   그림과 알고리즘을 참조한 뒤 세부 절로 넘어간다.
4. 세부 절 제목은 **하는 일 + 그 일의 성질**로 짓는다. `Stability-preserving Residual Blending`
5. 짧은 논문에서는 1과 2를 합치고 세부 절을 둘로 줄인다 (K).

## 7.3 절 제목의 `[역할]: [명사구]` 형식

- `Design Principle: Stability by Construction` (G)
- `Overview of GITA: A Simple Instantiation` (G)
- `Problem: The Absence of Continuation Mechanism` (K)
- `Adaptation at Different Representational Levels: Input-level vs Internal-representation` (G)

콜론 앞이 역할, 뒤가 그 절의 내용을 압축한 명사구다.

## 7.4 굵은 머리말

세부 항목은 별도 절을 만들지 않고 굵은 머리말로 문단을 시작한다. 두 형태가 있다.

| 형태 | 용도 | 원문 |
| --- | --- | --- |
| 용어를 주어로 문장에 녹임 | 개념 정의 | "**Histogram Stretching** is a linear intensity transformation that rescales image pixel values to utilize a desired dynamic range" (G) |
| 마침표로 끊고 새 문장 | 절차 항목 | "**Target feature statistics.** Given an input feature, each backbone block $b$ produces ..." (G) |

## 7.5 절차의 서술

`We achieve this through a two-step process. First, ... Second, ...` (K) 형태로 본문 안에서
번호를 매긴다. 불릿을 쓰지 않는다.

## 7.6 설계 선택의 정당화

각 설계 요소마다 **왜 그것이어야 하는지**를 붙인다.

> "**Crucially,** $\gamma$ is the only learnable parameter in GITA, which keeps the adaptation
> space minimal and bounded." (G)

> "We interpret this blending as an identity-preserving residual connection on the input: a fixed
> fraction $(1-\lambda)$ of the original image is always retained, regardless of how the
> transformation behaves. **This provides a bounded safety guarantee**, ..." (G)

## 7.7 수식

- 번호를 붙이고 본문에서 참조한다.
- 기호를 말로 먼저 정의한다. "where $\phi$ denotes the operations preceding the normalization
  layer" (G)
- `Formally, [대상] is denoted as [표기], where ...` 로 형식화를 도입한다.

---

# 제8부. Experiments

## 8.1 절 구성

```
5.1 Datasets and Evaluation Scenarios     (또는 Experimental Setup)
5.2 Experimental Setup
5.3 Main Results
5.4 Further Analysis
```

**주 결과 절의 이름은 `Main Results` 다.**

## 8.2 굵은 머리말

- `**Datasets and Models.**` `**Evaluation Metrics.**` (K)
- `**Experiment on Discrete.**` `**Experiment on Continuous.**` `**Ablation Study.**`
  `**Extreme Long-horizon Stability.**` `**Cross-dataset Generalization.**` (G)

## 8.3 결과 서술의 순서

```
[표 참조] -> [전반적 경향] -> [예외 또는 조건] -> [해석]
```

> "In Round 1, most adaptation methods improve over the Direct-Test baseline, and our method
> achieves the best or highly competitive performance across detectors (e.g., +3.62 mAP on
> SHIFT-D. with Faster R-CNN (ResNet-50)), reducing distribution discrepancy without modifying
> detector parameters. **The exception is YOLO11, whose gains remain marginal; see Appendix D**
> for a detailed analysis." (G)

**예외를 같은 문단에서 자진 보고한다.**

## 8.4 발견의 구조에 이름을 붙인다

> "These dynamics **indicate two regimes**. In the early stage, the model improves voice
> consistency without severely compromising linguistic content. In the later stage, competition
> for attention becomes stronger, and intelligibility degrades more rapidly." (K)

관측을 나열하지 않고 그 관측이 이루는 구조를 명명한다.

## 8.5 자기 방법의 대가를 결과 절에서 말한다

> "**These results reveal a clear trade-off:** our approach improves voice consistency across
> generations, but this gain is accompanied by a substantial degradation in content fidelity." (K)

> "**However, this ultimately introduces a new sink** that competes with content modeling for
> attention capacity, and as it absorbs an increasing share of the attention budget, the model's
> ability to represent linguistic content degrades." (K)

## 8.6 경쟁 방법을 공정하게 처우한다

우리가 우위인 지점에서 상대 방법의 이점이 어디서 오는지 설명하고 그것이 우리와 직교함을 밝힌다.

> "the high throughput of WHW-Skip instead comes from skipping adaptation steps, **an orthogonal
> mechanism that other update-based methods could adopt as well**." (G)

## 8.7 부록으로 미루기

> "**Due to space constraints, we summarize the key findings of each analysis below and defer all
> experimental details, tables, and discussions to the Appendix.**" (G)

이후 각 분석을 굵은 머리말 한 문단으로 요약하고 `Appendix B.1` 형태로 지시한다.

---

# 제9부. Limitations

## 9.1 형식

**`This work has several limitations. First, ... Second, ... Third, ...`** (K)

각 항목은 진술로 끝나지 않고 **원인 또는 귀속을 함께 쓴다.**

> "Second, the improvement is more pronounced in the Dec-Only architecture than in the Enc-Dec
> models; **we attribute this to** Enc-Dec models' architectural constraint: speaker conditioning
> is mediated through cross-attention in the encoder, limiting the influence of a decoder-level
> token on the upstream speaker representation." (K)

## 9.2 해결되지 않은 것을 명시한다

> "Third, the observed COS/WER trade-off **remains unresolved**." (K)

## 9.3 인라인 겸양

별도 절과 별개로, 본문 각 절에서 주장한 자리마다 범위를 좁힌다.

| 유형 | 원문 |
| --- | --- |
| 해석의 한정 | "**We treat this not as a claim that** domain shift is inherently low-level, **but only as** an instrumental signal: the specific intensity statistics that our method measures and corrects move consistently under these weather transitions." (G) |
| 한계의 자진 인정 | "**We note, however, that** this low-level view does not capture every weather-induced change, since real scenes can also undergo genuine semantic changes." (G) |
| 적용 범위 명시 | "This pilot observation is conducted purely at the raw-pixel level; the alignment objective used by our method instead operates on feature-level statistics as described in Sec. 4." (G) |

## 9.4 Future work

한계에서 곧바로 이어 쓴다. 무엇을 하면 되는지를 구체적으로 적는다.

> "Future work may explore self-supervised objectives that directly optimize for cross-utterance
> speaker invariance, or approaches that align textual voice descriptions with audio
> representations, **recovering the continuation-based inductive bias that exists in Voice Cloning
> TTS**." (K)

---

# 제10부. Conclusion

## 10.1 구성

**단일 문단. 네 문장에서 여섯 문장.**

```
[논문이 무엇을 했는가 또는 무엇을 규명했는가]
[핵심 기제]
[유지되는 성질 또는 결과]
[겸손한 자리매김]
```

> "This paper **identifies a fundamental problem** in Voice Design TTS: unlike LM-based voice
> cloning, it cannot rely on acoustic continuation to stabilize speaker identity across
> generations. We propose a training recipe based on voice profile-level data reconstruction and
> attention sink stabilization. **We view this as a first step toward** principled consistency
> modeling in Voice Design TTS, **and hope this work motivates further research on** stable and
> intelligible text-driven speech synthesis." (K)

> "We propose GITA, an input-level CTTA framework for adaptive object detection under continual
> weather shifts. GITA addresses long-horizon instability by moving adaptation from detector
> parameters to the input space. It keeps the detector frozen and learns only a lightweight
> gamma-based intensity transformation. ... **These results suggest that** constraining the
> adaptation space is **a simple and practical path toward** reliable source-free perception in
> changing environments." (G)

## 10.2 하위 절

| 조건 | 형태 |
| --- | --- |
| 긴 논문 (8p 이상) | `Conclusion` 단독. 한계는 본문 각 절에 인라인 분산 |
| 짧은 논문 (4p 이하) | `Discussion and Conclusion` 아래 `Limitations and Future Directions` 와 `Conclusion` 두 하위 절 (K) |

---

# 제11부. 그림과 표

## 11.1 그림 캡션

무엇을 보이는지 서술하고 **무엇을 읽어야 하는지까지 쓴다.** 대비가 핵심이면 캡션 안에서 대비를
완결한다.

> "Figure 2. Detection performance across ten repeated adaptation rounds using Faster R-CNN
> (ResNet-50). Several parameter-update baselines that are competitive in early rounds degrade
> sharply or collapse as the same weather-domain cycle repeats, **whereas our input-level
> adaptation maintains stable performance throughout.** Additional results are in Appendix C." (G)

다중 패널은 `(A)` `(B)`로 나누고 각 패널에 완결된 설명을 준다.

> "Figure 1: (A) **Unlike** Voice Cloning, which leverages audio continuation as a natural
> learning objective, Voice Design relies solely on a textual prompt, making continuation
> impossible and resulting in an unstable generation process. (B) **Without** the consistency
> token, content information leaks into the voice representation, corrupting speaker identity.
> **With** the token inserted, it serves as a dedicated information anchor that absorbs
> content-related attention, yielding a clean and stable voice representation." (K)

`Without ... With ...` 대비를 캡션 안에 넣는다.

## 11.2 표

- 최고값 **bold**, 차선값 밑줄. 캡션에 규약을 명시한다.
  > "where **bold** and underlined values denote the best and second-best results, respectively."
- 지표 방향을 화살표로 표시한다. `UTMOS↑`, `WER↓`, `COS_GG↑` (K)

## 11.3 참조 형식

| 대상 | 형식 |
| --- | --- |
| 그림 | `Fig. 1` (G) / `Figure 1` (K) |
| 표 | `Table 1` |
| 절 | `Sec. 4`, `Sec. 3.2` |
| 부록 | `Appendix A`, `Appendix B.1` |
| 알고리즘 | `Algorithm 1` |

---

# 제12부. 금지 사항

1. **새 개념을 대조항 없이 도입하지 않는다.** 이것이 첫 번째 규칙이다.
2. **선언문 제목을 쓰지 않는다.** `X Is Y` 형태를 쓰지 않는다.
3. **수사의문문은 한 편에 한 번, Introduction 전환 문단에서만 쓴다.**
4. **강조 부사를 한 문단에 두 번 쓰지 않는다.**
5. **개선을 주장한 뒤 대가를 말하지 않고 넘어가지 않는다.**
6. **주장한 뒤 범위를 좁히지 않고 넘어가지 않는다.**
7. **실패를 서술만 하고 기제를 붙이지 않은 채 넘어가지 않는다.**
8. 불릿 목록은 기여 목록 외에는 쓰지 않는다.
9. 검증되지 않은 추론에 강도 표지(`We hypothesize`, `We argue`)를 붙이지 않고 단정하지 않는다.

---

# 제13부. 점검표

## 골격

| 항목 | 확인 |
| --- | --- |
| 문제를 계열의 구조적 성질로 격상시킨 문장이 있는가 | |
| 그 격상이 기제 설명 또는 유비로 뒷받침되는가 | |
| 증상 관리와 원인 제거를 대비시켰는가 | |
| `Our central claim is that ...` 또는 그에 준하는 단정이 있는가 | |
| 중심 주장 직후에 범위를 좁혔는가 | |

## Abstract

| 항목 | 확인 |
| --- | --- |
| 3번 문장에 실패의 기제가 종속절로 붙어 있는가 | |
| 원리 제안과 구현체 소개가 분리되어 있는가 | |
| 수치를 넣었다면 대가를 인접 문장에 썼는가 | |
| 겸손한 명사구로 닫았는가 | |

## Introduction

| 항목 | 확인 |
| --- | --- |
| 문단 1이 깔때기 형태인가 | |
| 관측한 현상에 이름을 붙였는가 | |
| 인접 개념과의 구별을 명시했는가 | |
| 수사의문문이 정확히 한 번, 전환 문단에만 있는가 | |
| 기여가 진단 제안 검증을 축으로 구성되어 있는가 | |

## 본문

| 항목 | 확인 |
| --- | --- |
| Related Work 각 절이 `However`로 한계를 지적하며 닫는가 | |
| Method 첫 절이 문제인가 | |
| 설계 원리 절이 `Design Principle: [요약]` 형식인가 | |
| 개요 절에서 전체를 서술한 뒤 세부로 내려가는가 | |
| 세부 항목이 굵은 머리말로 처리되었는가 | |
| 실험 주 결과 절이 `Main Results`인가 | |
| 예외를 같은 문단에서 자진 보고했는가 | |
| 발견의 구조에 이름을 붙였는가 | |
| 자기 방법의 대가를 결과 절에서 말했는가 | |
| 경쟁 방법의 이점을 공정하게 설명했는가 | |

## 문장

| 항목 | 확인 |
| --- | --- |
| 새로 도입한 개념마다 대조항이 있는가 | |
| 추론의 강도가 표지되어 있는가 | |
| 핵심 주장 문장이 짧게 끊겨 있는가 | |
| 그림 캡션이 무엇을 읽어야 하는지까지 쓰고 있는가 | |
| Conclusion이 단일 문단이고 겸손한 자리매김으로 닫는가 | |

---

# 부록. 어구 사전

바로 꺼내 쓸 수 있는 형태로 정리한다.

## 구조적 격상

- `reflects not an incidental implementation detail but a structural property of ...`
- `confirming that this ... is structural rather than specific to any single method or architecture`
- `We argue that the root cause is architectural.`
- `places [X] in a structurally similar position to [과거 계열], which also [같은 성질] and suffered from analogous [문제]`
- `Crucially, this ... is not merely hypothetical:`

## 대비

- `A rather than B` / `A rather than merely B`
- `not A but B` / `A, not B`
- `Unlike A, which ..., B ...`
- `While A ..., B ...`
- `By comparison, ...` / `In contrast, ...`
- `X differ in where ...`
- `This problem is fundamentally different from ...`
- `[기존]은 [문제]를 after it arises rather than removing it at its source`

## 전환

- `These observations motivate a different design question: rather than ..., can we instead ...?`
- `Building on this principle, which we refer to as X, we instantiate ...`
- `Motivated by this finding, we propose ...`
- `To address this, we introduce ...`
- `To mitigate this gap, we propose ...`
- `Based on these observations, our contributions are:`

## 공백 지적

- `Comparatively little attention has been paid to whether ...`
- `..., yet explicit [X] in [Y] remains unexplored.`
- `These limitations motivate the development of ...`
- `In contrast, we aim to ...`

## 겸양

- `we view [X] as a minimal demonstration of this idea rather than an optimal or exhaustive solution`
- `We view this as a first step toward ...`
- `We treat this not as a claim that ..., but only as an instrumental signal:`
- `We note, however, that ...`
- `The exception is [X], whose gains remain marginal; see Appendix [Y].`
- `..., an orthogonal mechanism that other [계열] could adopt as well.`

## 결과와 해석

- `As shown in Table 1, although [양보], [발견], indicating [함의].`
- `These results reveal a clear trade-off: [이득], but [대가].`
- `These dynamics indicate two regimes. In the early stage, ... In the later stage, ...`
- `Clear differences emerge after ...`
- `These results indicate that ...`

## 마무리

- `These results suggest that [X] is a simple and practical path toward [Y].`
- `These findings provide an initial quantitative foundation for [X].`
- `We view this as a first step toward [X], and hope this work motivates further research on [Y].`
