# ASCENT-G 현황 문서

**작성일**: 2026-04-26 (revised)
**모델**: `Qwen/Qwen2.5-1.5B-Instruct`

---

## 전체 진행 상황

| 단계 | 상태 | 비고 |
|------|------|------|
| Phase 0 파이프라인 검증 (GSM8K) | ✅ 완료 | 2026-04-21, T4 GPU |
| Phase 1 태스크 10개 50-step 파일럿 수집 | ✅ 완료 | 2026-04-22~24 |
| H1a/H1b 파일럿 분석 | ✅ 완료 (Inconclusive) | 2026-04-25 |
| **개정 10-task 1000-step 수집** | 🔄 진행 중 | 4/10 완료, exploratory plan |
| H2 전이 실험 | ⏳ 대기 | H1a/H1b 이후 |

---

## 1000-step 수집 현황

### 완료

| Task | Norm | 소요시간 | 실행값 | 권장값 | 비고 |
|------|------|---------|--------|--------|------|
| CommonsenseQA | 23.16 | 5955s | 256 | 64 | ✅ 완료, 재실행 필요 없음 |
| ARC-Challenge | 23.17 | 7470s | 256 | 64 | ✅ 완료, 재실행 필요 없음 |
| HellaSwag | 23.11 | 12638s | 256 | 64~96 | ✅ 완료, 재실행 필요 없음 |
| GSM8K | 23.21 | 26032s | 256 | 256 | ✅ 완료, `step 460` 조기중단, best reward `0.9125 @ step 280` |

### 미실행 (개정 exploratory 계획)

| Task | 전략 | 계획값 | 권장값 | Kaggle T4 예상 시간 |
|------|------|--------|--------|----------------------|
| HumanEval | 원안 유지 | 256 | 256 | ~17.7h |
| MBPP | 원안 유지 | 256 | 256 | ~16.1h |
| AMC | 64토큰으로 테스트 | 64 (deviation) | 64, 실패 시 96 재검토 | ~5~7h |
| MATH500 | 64토큰으로 테스트 | 64 (deviation) | 96 권장, 속도 우선이면 64 | ~5~7h at 64, ~7~9h at 96 |
| ~~MATH~~ → **ARC-Easy** | 대체 | 64 | 64 | ~1.5~2.5h |
| ~~AIME~~ → **WinoGrande** | 대체 | 64 | 64 | ~2~4h |

권장 원칙:
- 코드 생성 태스크(`HumanEval`, `MBPP`)는 `256` 유지
- 수학 최종답 태스크는 `GSM8K`만 `256` 유지, `AMC`/`MATH500`는 exploratory로 `64~96` 검토
- MCQ / binary-choice 태스크는 `64` 중심으로 운영

---

## 태스크 변경 이력

### MATH → ARC-Easy (대체)
- **사유**: `competition_math`는 느리고 학습 신호가 약함. step 100~190 구간에서 reward가 대부분 `0.0000`이었고, step 190 시점 elapsed `202.1m`, eta `861.4m`로 1000-step 완주 예상 시간이 약 17시간 수준이었음.
- **해석**: 등록 태스크 유지보다는 실행 가능한 대체 태스크를 찾는 exploratory 운영 결정으로 판단
- **대체**: `allenai/ai2_arc`의 `ARC-Easy` split — 짧은 과학 상식 MCQ, 기존 `ARC-Challenge`와 구조가 유사해 구현 비용이 낮음
- **구현 상태**: CLI 경로(`config/task_registry.json`, `train_grpo_task.py`)에는 반영됨. Kaggle 노트북은 별도 생성 필요

### AIME → WinoGrande (대체)
- **사유**: AIME는 운영상 매우 느릴 가능성이 높고, 1.5B 모델에서 안정적 reward를 기대하기 어려운 고난도 태스크로 판단
- **대체**: `allenai/winogrande` — 이진 선택형 상식 추론으로 출력이 짧고 reward 경로를 단순하게 구성할 수 있음
- **구현 상태**: CLI 경로(`config/task_registry.json`, `train_grpo_task.py`)에는 반영됨. Kaggle 노트북은 별도 생성 필요

### max_completion_length 조정 (AMC, MATH500)
- 256 → 64 토큰
- 등록 `v1.3` 기본값과는 다른 exploratory deviation
- 런 노트와 최종 리포트에 편차(deviation)로 명시 기록

---

## 개정 exploratory 10개 태스크 후보

| # | Task | Domain |
|---|------|--------|
| 1 | CommonsenseQA | 상식 추론 ✅ |
| 2 | ARC-Challenge | 과학 상식 ✅ |
| 3 | HellaSwag | 자연어 추론 ✅ |
| 4 | GSM8K | 초등 수학 |
| 5 | HumanEval | 코드 생성 |
| 6 | MBPP | 코드 생성 |
| 7 | AMC | 수학 경시 (64tok) |
| 8 | MATH500 | 수학 경시 (64tok) |
| 9 | ARC-Easy | 과학 상식 |
| 10 | WinoGrande | 언어/상식 추론 |

이 목록은 등록 태스크 세트 대체안이 아니라, 운영 제약을 반영한 exploratory 후보 목록이다.
등록 태스크 세트는 여전히 `README.md`의 `v1.3` 목록을 따른다.

---

## H1a/H1b 파일럿 분석 결과 (50-step, 참고용)

**판정: Inconclusive** — 초기화 노이즈가 학습 신호를 압도

| 지표 | 50-step | 1000-step (3-task 미리보기) |
|------|---------|--------------------------|
| H1b mean\|cos\| | 0.086 | 0.00037 |
| H1a r_90 | 9/10 | TBD |

1000-step에서 H1b 신호가 200배 감소 → 태스크별 방향성 확인됨.

---

## 2026-04-26 업데이트

### GSM8K 1000-step 풀런 결과

- Kaggle T4에서 `max_steps=1000`으로 실행했고, 실제 종료는 `step 460`
- 종료 이유: `180 step` 동안 reward 최고값 갱신 없음
- 최고 reward: `0.9125 @ step 280`
- 마지막 reward: `0.6750 @ step 460`
- 평균 reward: `0.7231`
- 실제 step 시간: `56.59s/step` (`26032.3s / 460 step`)

해석:
- 이 런은 실패가 아니라 `수렴 후 정체`에 따른 조기중단에 가깝다.
- reward가 `0.55`에서 시작해 `0.9125`까지 올라갔고 후반 평균도 유지돼, 학습 신호는 분명하다.
- `adapter`, `checkpoint-450`, `run_report.json`, `update_vector.npy`, `update_vector_provenance.json`이 모두 생성돼 후속 분석용 아티팩트로 사용 가능하다.

### H1a/H1b 중간 해석

- `50-step 10태스크` 파일럿: `H1a inconclusive`, `H1b pass`
- `1000-step 3태스크` preview: `H1b pass`, mean `|cos| = 0.00037`
- `GSM8K`를 포함한 `1000-step 4태스크` quick check:
  - mean `|cos| = 0.00045`
  - max `|cos| = 0.00113`
  - `r90 = 4`, `rho = 1.0`

현재 판단:
- `H1b`는 우호적이다. 풀런 벡터들은 서로 매우 다른 방향을 보인다.
- `H1a`는 아직 우호적이라고 보기 어렵다. 다만 반박으로 결론낼 단계도 아니다.
- 특히 수학 군집(`MATH`, `AIME`, `AMC`, `MATH500`)과 코드 군집(`HumanEval`, `MBPP`)이 추가돼야 공통 저차원 구조 존재 여부를 제대로 볼 수 있다.

### 운영/계측 수정

- 모든 `phase0-*.ipynb`에서 진행 로그를 `stdout`과 `train_progress.log`에 동시에 기록하도록 수정
- 모든 노트북에 동일한 조기중단 로직 적용
- `per_step_time_sec` 계산을 `max_steps`가 아니라 `actual_steps` 기준으로 수정
- 이후 생성되는 `run_report.json`에는 `training.actual_steps`가 함께 기록됨

---

## 해석 주의

- 이 문서의 개정 계획은 등록 Phase 1 primary analysis가 아니라 exploratory 실행 계획이다.
- `MATH`, `AIME` 교체나 `max_completion_length=64` 사용은 모두 등록 경로 밖의 운영상 수정이다.
- 따라서 이후 10-task 결과는 `registered H1a/H1b`가 아니라 `revised exploratory H1a/H1b`로 라벨링해야 한다.

---

## 아티팩트 위치

| 아티팩트 | 위치 |
|----------|------|
| 50-step 파일럿 벡터 | Kaggle kernel outputs |
| 1000-step 벡터 (완료분) | Kaggle kernel outputs |
| 런 리포트 | `runs/2026-04-25-phase1-{task}-qwen2.5-1.5b/` |
| H1a/H1b 파일럿 리포트 | `runs/2026-04-25-phase1-h1a-h1b-pilot/` |
| 상세 실행 계획 | `runs/2026-04-25-phase1-h1a-h1b-pilot/full_run_plan.md` |

---

## 다음 액션 (우선순위 순)

1. HumanEval, MBPP 노트북 → Kaggle T4에서 실행 (max_steps=1000)
2. AMC, MATH500 노트북 → max_completion_length=64으로 실행
3. ARC-Easy, WinoGrande 노트북 신규 생성 → 실행
4. 10개 벡터 수집 완료 후 `h1a_h1b_task_matrix.py` 실행 → revised exploratory H1a/H1b 판정
