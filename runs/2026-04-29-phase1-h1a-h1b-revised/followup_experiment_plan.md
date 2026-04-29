# Phase 1 후속 실험 계획 — Subspace 구조 심화 분석

**작성일**: 2026-04-29  
**배경**: 10-task revised exploratory H1a 결과 — r90=9, ρ=0.90, inconclusive (weak fail)

---

## 1. 현재 결과 해석

### 무엇을 발견했나

| 지표 | 값 | 해석 |
|------|----|------|
| H1a r90 | 9/10 | 10개 벡터가 거의 전 차원 점유 |
| H1a ρ | 0.90 | 공통 저차원 subspace 없음 |
| H1b mean\|cos\| | 0.074 | 벡터들 서로 거의 수직 |
| H1b 판정 | pass | 태스크별 방향성 독립 확인 |

### H1a 실패의 과학적 의미

H1a 실패 = "다양한 태스크를 학습한 LoRA 벡터들은 공통의 저차원 기하 구조를 갖지 않는다"

이것은 다음을 시사한다:
- 태스크마다 모델을 수정하는 방향이 완전히 다름
- LoRA 업데이트는 태스크 특화적 (task-specific), 범용적 (task-general) 구조 없음
- 이 자체가 논문의 핵심 발견이 될 수 있음

**중요**: H1a 실패는 "실험 실패"가 아니라 "가설 기각"이다. 기각도 기여다.

---

## 2. 후속 실험 계획

### Plan A — Attention / MLP 기능 분리 분석

#### 근거

Transformer 내부 기능 분업:
- **Attention (q/k/v/o_proj)**: 토큰 간 관계, 맥락 집중 패턴
- **MLP (gate/up/down_proj)**: 사실 지식 저장, 비선형 변환

Attention은 태스크와 무관한 공통 패턴(어디에 집중할지)을 학습할 가능성이 있다.
MLP는 태스크별 지식 인코딩이 지배적이라 독립적일 가능성이 높다.

#### 구현

벡터 구조 (provenance 기준):
- 총 196개 항목 (28 레이어 × 7 모듈)
- Attention 인덱스: 112개 범위 (23.6%, 2,179,072 elements)
- MLP 인덱스: 84개 범위 (76.4%, 7,053,312 elements)

```python
# provenance를 이용한 슬라이싱
attn_idx = []  # attention 파라미터 인덱스 목록
mlp_idx  = []  # MLP 파라미터 인덱스 목록

for entry in provenance['layers']:
    mod = entry['name'].split('.')[-1]
    start, length = entry['offset'], entry['a_numel'] + entry['b_numel']
    if mod in {'q_proj', 'k_proj', 'v_proj', 'o_proj'}:
        attn_idx.extend(range(start, start + length))
    else:
        mlp_idx.extend(range(start, start + length))

v_attn = full_vector[attn_idx]
v_mlp  = full_vector[mlp_idx]
```

#### 예상 결과

| 가설 | 근거 |
|------|------|
| Attention r90 < MLP r90 | Attention이 태스크 공통 패턴을 공유할 가능성 |
| MLP r90 ≈ 10 | 태스크별 지식은 완전 독립 방향 |

→ Attention에서 r90 ≤ 4라면 **새로운 기여**:  
"LoRA update geometry는 functional component에 따라 다르다. Attention은 공유 구조, MLP는 태스크 독립적."

---

### Plan B — 태스크 군집별 Subspace 분석

#### 근거

10개 혼합 태스크에서 r90=9인 것은 당연할 수 있다. 유사 태스크들만 모으면 공유 구조가 드러날 수 있다.

#### 군집 구성

| 군집 | 태스크 | 공통점 |
|------|--------|--------|
| **MCQ** | CommonsenseQA, ARC-Challenge, ARC-Easy, OpenbookQA | 4지선다 MCQ |
| **코드** | HumanEval, MBPP | 코드 생성 |
| **수학** | GSM8K, SVAMP | word problem, 숫자 정답 |
| **언어추론** | HellaSwag, WinoGrande | 자연어 완성/이진 선택 |

#### 예상 결과

MCQ 4개 군집: r90 ≤ 2라면 "동일 포맷 태스크는 공통 subspace를 공유한다"는 강력한 발견.

---

### Plan C — 레이어 깊이별 분석

#### 근거

초기 레이어(0~9): 일반적 언어 이해 → 태스크 공통 구조 가능
중간 레이어(10~18): 추론 처리 → 혼합
후기 레이어(19~27): 태스크 특화 출력 → 독립

#### 구현

28개 레이어를 Early/Mid/Late로 3분할 후 각각 H1a 실행.

---

## 3. 새로운 돌파구 — 기여 재구성

### 원래 가설 vs 실제 발견

| | 원가설 (H1a pass 가정) | 실제 발견 |
|--|----------------------|----------|
| 핵심 주장 | 태스크 벡터가 저차원 공통 구조 공유 | 태스크 벡터가 완전 독립 방향 |
| 시사점 | 범용 fine-tuning geometry 존재 | 태스크 특화성이 지배적 |
| 활용 | 공통 방향으로 효율적 학습 | task arithmetic 간섭 최소 |

### 새로운 기여 방향 3가지

#### 기여 1: Functional Component별 기하 구조 차이 (Plan A로 검증)

"LoRA update geometry는 기능별로 다르다. Attention은 태스크 간 공유 구조를 가질 수 있으나 MLP는 태스크 독립적이다. 이는 selective merging의 이론적 근거가 된다."

→ **실용적 함의**: MLP는 태스크별로 유지하고 Attention만 선택적으로 병합하는 모델 편집 전략

#### 기여 2: Task Arithmetic의 기하적 근거 (현재 결과로 즉시 주장 가능)

H1b pass (mean |cos| = 0.074) = 태스크 벡터들이 거의 수직

"서로 다른 태스크 LoRA 벡터들이 거의 직교한다는 것은 task vector arithmetic (덧셈/뺄셈)에서 태스크 간 간섭이 최소화됨을 기하적으로 보장한다. 이것이 task arithmetic이 실제로 동작하는 이유다."

→ Ilharco et al. (2023) Task Arithmetic 논문의 경험적 관찰을 기하학적으로 설명하는 기여

#### 기여 3: Domain Cluster Geometry (Plan B로 검증)

"유사 도메인 태스크들은 공통 subspace를 공유하지만, 이종 도메인 간에는 subspace가 분리된다. 이는 continual learning에서의 catastrophic forgetting이 기하적으로 왜 발생하는지 설명한다."

---

## 4. 구현 우선순위 및 GPU 비용

| 실험 | GPU 필요 | 구현 비용 | 우선순위 |
|------|----------|-----------|----------|
| Plan A (Attention/MLP 분리) | **없음** (벡터 재슬라이싱) | 낮음 | **1순위** |
| Plan B (군집별 H1a) | **없음** (기존 벡터 재사용) | 낮음 | **2순위** |
| Plan C (레이어별) | **없음** (벡터 재슬라이싱) | 중간 | **3순위** |

**세 가지 모두 추가 GPU 없이 현재 벡터만으로 실행 가능.**

---

## 5. 다음 단계

1. `analysis/` 하위에 `h1a_functional_split.py` 구현 (Plan A)
2. `analysis/` 하위에 `h1a_cluster_analysis.py` 구현 (Plan B)
3. Plan A 실행 → Attention r90 vs MLP r90 비교
4. Plan B 실행 → MCQ 군집 r90 확인
5. 결과에 따라 논문 기여 방향 재설정

---

## 6. 참고: 벡터 인덱스 구조

```
총 벡터 길이: 9,232,384
레이어 수: 28 (Qwen2.5-1.5B)
모듈 수: 7 (q/k/v/o/gate/up/down)
LoRA rank: 8

Attention 파라미터: 2,179,072 (23.6%)
  - q_proj: [1536→8, 8→1536] × 28 = 688,128
  - k_proj: [1536→8, 8→256]  × 28 = 401,408
  - v_proj: [1536→8, 8→256]  × 28 = 401,408
  - o_proj: [1536→8, 8→1536] × 28 = 688,128

MLP 파라미터: 7,053,312 (76.4%)
  - gate_proj: [1536→8, 8→8960] × 28 = 2,351,104
  - up_proj:   [1536→8, 8→8960] × 28 = 2,351,104
  - down_proj: [8960→8, 8→1536] × 28 = 2,351,104

인덱스는 레이어 순서로 interleaved (연속 블록 아님)
→ provenance.json 기반 fancy indexing 필요
```
