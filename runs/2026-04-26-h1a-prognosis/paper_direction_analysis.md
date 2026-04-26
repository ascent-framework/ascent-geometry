# H1a 실패 시 논문 방향 분석

**작성일**: 2026-04-26  
**참조**: `ascent-framework/docs/preregistration/v1.3.md` §7~§9

---

## 핵심 결론

**H1a 실패 = 논문 실패가 아니다.**

preregistration v1.3 §9에 Recovery Path가 이미 명시되어 있다.  
모든 실험 결과가 §7 해석 테이블에 의해 허용 클레임으로 매핑된다.

---

## §7 해석 테이블 (v1.3 원문 기준)

| H1a | H1b | H2 | 허용되는 주장 |
|-----|-----|-----|-------------|
| Fail | N/A | Any | "No shared adaptive structure detected" |
| Inconclusive | N/A | Any | "Weak evidence; inconclusive" |
| Marginal pass | Fail | Any | "Possible shared update manifold only" |
| Marginal pass | Pass | Fail | "Suggestive capability-like structure, no transfer" |
| Marginal pass | Pass | Pass | "Suggestive capability-like decomposition with transfer" |
| Strong pass | Pass | Fail | "Capability-like decomposition, limited transfer" |
| Strong pass | Pass | Pass | **"Capability-like decomposition with transfer"** (목표) |

H1a fail이더라도 논문 제출 불가가 아니라, **다른 클레임 레이어로 이동**한다.

---

## 시나리오별 논문 방향

### 시나리오 1: H1a fail + H2 pass

```
공통 부분공간 없음 (H1a fail)
    BUT
태스크별 방향이 다른 모델 변형체로 전이됨 (H2 pass)
```

이것이 오히려 더 강한 독창적 발견일 수 있다.

- 각 태스크가 고유한 방향을 갖는데도 그 방향이 모델 패밀리에 걸쳐 보존된다
- 모델 구조에 task-specific stable direction이 내재되어 있다는 뜻
- "공통 부분공간 없이도 전이가 일어난다" → 기존 transfer learning 가정과 충돌하는 새로운 발견

**허용 클레임**: "Task-specific adaptation directions are geometrically stable across model variants, without requiring a shared subspace."

---

### 시나리오 2: H1a fail + H2 fail (Hard Falsification)

§8 기준: H1a fail AND H2 fail 둘 다 충족 시 Hard Falsification.

§9 Recovery Path 활성화:

> **Alternative hypothesis:**  
> "Tiny updates may work through decision-boundary smoothing, implicit regularization,  
> or output distribution reshaping, rather than structured adaptive subspaces."

**Alternative paper title** (v1.3 §9 원문):  
*"Understanding TinyLoRA Without Shared Adaptive Subspaces"*

조사 방향:
- 13개 파라미터 업데이트 이후 loss landscape는 어떻게 바뀌는가?
- output probability distribution이 구조적으로 이동하는가?
- implicit L2 regularization이 작은 업데이트 크기를 설명하는가?

→ 여전히 **publishable empirical contribution**으로 v1.3에 명시됨.

---

### 시나리오 3: H1a marginal pass (낙관, 아직 가능)

현재 4-task 데이터는 MCQ 3개(CommonsenseQA, ARC-Challenge, HellaSwag) + GSM8K로,  
도메인 다양성이 부족한 상태다.

수학 군집(GSM8K, AMC, MATH500) + 코드 군집(HumanEval, MBPP)이 추가되면  
singular value drop이 생겨 r90이 낮아질 가능성이 있다.

r90 ≤ 4이면 marginal_pass → §7 테이블 상위권 클레임으로 이동 가능.

---

## 논문 방향을 지금 바꾸지 않아야 하는 이유

1. **preregistration이 모든 outcome을 커버한다.**  
   §7 테이블이 결과에 따른 클레임을 이미 매핑했다.

2. **H2를 아직 실행하지 않았다.**  
   Hard Falsification은 H1a fail AND H2 fail 둘 다 필요하다.  
   H2 없이 논문 방향을 바꾸는 것은 시기상조다.

3. **10-task 풀런이 아직 완료되지 않았다.**  
   수학/코드 도메인이 추가되면 r90이 달라질 수 있다.

4. **사후 방향 변경은 신뢰도를 낮춘다.**  
   결과를 보고 가설을 조정하는 것은 HARKing에 해당한다.  
   현재 preregistration 안에서 결과를 있는 그대로 보고하는 것이 맞다.

---

## 현재 데이터가 이미 보여주는 것

| 지표 | 값 | 해석 |
|------|-----|------|
| H1b mean\|cos\| (1000-step) | 0.00037~0.00045 | 태스크별 방향이 매우 뚜렷하게 구분됨 |
| H1a r90 (4-task) | 4/4 | 공통 저차원 구조 없음 |
| 벡터 norm (1000-step) | 23.1~23.2 | 태스크 간 적응 크기는 균일 |

H1b pass 자체는 강한 결과다.  
"태스크별 적응 방향은 고차원 공간에서 서로 직교하게 구분된다"는 것은  
task arithmetic, 모델 편집, catastrophic forgetting 연구와 직접 연결된다.

---

## 권고 액션

```
1. 10-task 수집 완료 (계획대로 진행)
2. registered H1a/H1b 실행 → 실제 r90 확인
3. H2 전이 실험 진행 (Qwen2.5-1.5B-Instruct → Qwen2.5-1.5B)
4. H1a + H2 결과를 보고 §7 테이블에서 클레임 레이어 결정
5. 결정된 클레임에 맞게 Introduction / Contribution 문장 수정
```

---

## 논문 가치에 대한 핵심 관점

논문의 핵심 가치는 "공통 부분공간이 존재한다"는 주장이 아니라,  
**"적응 기하학을 preregistered 방식으로 측정해서 어떤 결과가 나왔는지 보고한다"** 는 것이다.

결과가 부정적이어도 그 자체가 기여다.  
preregistration + 부정적 결과의 정직한 보고는 재현 가능한 과학의 핵심이며,  
현재 ML 연구 커뮤니티에서 점점 더 높이 평가받는 방향이다.
