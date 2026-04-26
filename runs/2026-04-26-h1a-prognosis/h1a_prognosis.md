# H1a 통과 가능성 분석 및 대응 방향

**작성일**: 2026-04-26  
**기준 데이터**: 1000-step 4-task 완료 시점 (CommonsenseQA, ARC-Challenge, HellaSwag, GSM8K)

---

## 1. H1a 판정 기준

H1a는 task matrix(정규화된 task update vector를 열로 쌓은 행렬)의 SVD를 통해 공통 저차원 구조 존재를 검증한다.

```
rho = r90 / min(N, sqrt(dim))
    = r90 / min(10, sqrt(9,232,384))
    = r90 / 10          (N=10이 denominator)
```

| rho 범위 | r90 (N=10) | 판정 |
|----------|-----------|------|
| rho < 0.30 & CI_high < 0.35 | r90 ≤ 2 | strong_pass |
| 0.30 ≤ rho < 0.50 | r90 = 3~4 | marginal_pass |
| 0.50 ≤ rho < 0.70 | r90 = 5~6 | inconclusive |
| rho ≥ 0.70 & CI_low ≥ 0.65 | r90 ≥ 8 | fail |

---

## 2. 현재까지 수집된 데이터

### 실제 측정값

| 실험 | N | r90 | rho | H1a 판정 | H1b 판정 |
|------|---|-----|-----|---------|---------|
| 50-step 10-task (파일럿) | 10 | 9/10 | 0.900 | inconclusive (CI=[0.4,0.7]) | pass (mean\|cos\|=0.086) |
| 1000-step 3-task | 3 | 3/3 | 1.000 | inconclusive | pass (mean\|cos\|=0.000369) |
| 1000-step 4-task | 4 | 4/4 | 1.000 | — | pass (mean\|cos\|=0.000450) |

### 핵심 관측

- 1000-step에서도 multi-task matrix의 singular values가 near-isotropic  
  (`[1.0002, 1.0000, 0.9998]` — 50-step과 구조 동일)
- 태스크 수가 늘어날수록 r90 ≈ N 패턴이 유지됨
- H1b mean|cos|이 1000-step에서 50-step 대비 200배 감소 (0.086 → 0.00037)

---

## 3. 핵심 역설

```
H1b mean|cos| = 0.00037   → 벡터들이 거의 직교 (태스크별 고유 방향 ✓)
                             ↕ 구조적 충돌
H1a pass 조건             → 벡터들이 저차원 공통 부분공간에 집중되어야 함

직교에 가까운 벡터들 = near-equal singular values = r90 ≈ N → H1a 통과 불가
```

H1b 통과 자체가 H1a에 불리한 증거다. 9.2M 차원 공간에서 10개 벡터가 서로 직교한다는 것은 공통 저차원 구조가 없다는 뜻이다.

---

## 4. 10-task 1000-step 예측 시나리오

| 시나리오 | 조건 | 예상 r90 | rho | H1a 판정 |
|----------|------|---------|-----|---------|
| **낙관** | 수학 군집(4개) + 코드 군집(2개) + MCQ 군집(4개)이 명확히 분리 | 3~4 | 0.3~0.4 | marginal_pass |
| **중간** | 약한 군집 — singular value drop 불분명 | 5~6 | 0.5~0.6 | inconclusive |
| **비관** | 현재 패턴 그대로 (벡터 직교 → isotropic) | 9~10 | 0.9~1.0 | fail |

**현재 데이터 기준 낙관 시나리오 가능성은 낮다.** 3개의 동질적인 MCQ 태스크 (CommonsenseQA, ARC-Challenge, HellaSwag)로 구성된 3-task 분석에서도 singular value drop이 없었으므로, 도메인 군집 효과를 기대하기 어렵다.

---

## 5. 선택지

### A. 결과 그대로 보고 (권고)

10-task 수집 완료 → registered H1a/H1b 그대로 실행 → 결과를 정직하게 기록.

- preregistration v1.3 준수
- inconclusive/fail 자체가 유의미한 발견: "태스크별 적응은 저차원 공통 부분공간보다 태스크별 고유 방향으로 일어난다"
- H2 (전이 실험)는 H1a와 독립적으로 진행 가능

### B. 레이어별 / 모듈별 SVD 추가 (exploratory)

현재 분석은 `concat(ΔW_A, ΔW_B)` 전체 9.2M 벡터 기준이다. full vector가 near-orthogonal이더라도 레이어별 혹은 모듈 타입별로는 군집이 존재할 수 있다.

예시:
```bash
# q_proj 레이어만 쌓은 task matrix → SVD
# attention 계열(q/k/v/o_proj) vs MLP 계열(gate/up/down_proj) 분리
```

수학 태스크들이 attention에서 비슷한 방향을 공유하고, MCQ 태스크들이 MLP에서 비슷할 가능성이 있다. full vector에 합쳐지면 구조가 묻힌다.

**라벨링 주의**: 이 분석은 registered H1a가 아닌 exploratory로 명시해야 한다.

### C. preregistration amendment — 권고하지 않음

판정 기준(rho 임계값, 정규화 방식)을 사후에 변경하면 publication 신뢰도가 낮아진다. 결과가 나온 후 기준을 완화하는 것은 HARKing(Hypothesizing After Results are Known)에 해당한다.

---

## 6. 권고 액션

```
1. 10-task 수집 마저 완료
   (HumanEval → MBPP → AMC → MATH500 → ARC-Easy → WinoGrande)

2. registered H1a/H1b 그대로 실행 → 결과 기록

3. exploratory 추가: 레이어별 / 모듈별 SVD
   (analysis/h1a_h1b_task_matrix.py를 per-layer 버전으로 확장)

4. 결론 프레이밍:
   "H1b 강하게 통과 — 태스크별 적응 방향은 뚜렷하게 구분된다.
    H1a 관찰되지 않음 — 공통 저차원 부분공간은 full-vector 수준에서 존재하지 않는다.
    레이어별 분석에서 부분적 군집 구조가 관찰될 가능성은 열려 있다."
```

---

## 7. 해석 참고

H1a가 "통과 못 했다"는 것이 실험 실패가 아니다.  
"각 태스크가 독립적인 방향으로 모델을 업데이트한다"는 것 자체가 발견이며, H2 전이 실험(서로 다른 모델 간 벡터 전이)의 동기와 결합하면 일관된 서사를 형성할 수 있다.
