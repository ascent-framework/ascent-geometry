# Revised Exploratory H1a/H1b Analysis (2026-04-29)

## Status: COMPLETED

## Input
- Tasks: 10 (CommonsenseQA, ARC-Challenge, HellaSwag, GSM8K, OpenbookQA, ARC-Easy, WinoGrande, SVAMP, HumanEval, MBPP)
- Vector dimension: 9,232,384
- Bootstrap samples: 1000, seed: 42

## H1a Results (공통 저차원 subspace 존재 여부)
- r90: 9
- ρ (rho): 0.9000
- ρ CI₉₅: [0.4000, 0.7000]
- **Decision: inconclusive**

해석: r90=9는 10개 벡터가 거의 전 차원을 사용한다는 뜻. 공통 저차원 subspace가 없음을 시사하지만,
bootstrap CI가 [0.40, 0.70]으로 넓어 fail 확정 불가 → inconclusive 판정.
사실상 weak fail에 가까운 inconclusive.

## H1b Results (벡터 간 쌍별 유사도)
- mean |cos|: 0.07375
- max |cos|: 0.33535
- pair count: 45
- **Decision: pass**

해석: 벡터들이 r90 subspace 내에서 서로 거의 수직. 태스크별 방향성은 분명히 다름.

## 종합 판정
- H1a: **inconclusive (weak fail)**
- H1b: **pass**

태스크 특화 벡터들은 서로 다른 방향을 향하고 있으나 (H1b pass),
공통 저차원 기하 구조는 관찰되지 않음 (H1a inconclusive).
H1a를 지지하려면 r90 ≤ 3~4 수준이 필요했음.

## 이전 interim check와 비교

| 단계 | tasks | mean|cos| | r90 | rho |
|------|-------|-----------|-----|-----|
| 4-task preview | 4 | 0.00045 | 4 | 1.00 |
| 5-task precheck | 5 | 0.00131 | 5 | 1.00 |
| **10-task final** | **10** | **0.07375** | **9** | **0.90** |

10개 전체에서 r90이 9로 증가 — 수학/코드 군집 추가로도 공통 구조 없음 확인.

## 다음 단계
- H2 전이 실험 설계 (벡터 합산/보간으로 cross-task transfer 테스트)
