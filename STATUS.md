# ASCENT-G 현황 문서

**작성일**: 2026-04-29 (revised 8)
**모델**: `Qwen/Qwen2.5-1.5B-Instruct`

---

## 전체 진행 상황

| 단계 | 상태 | 비고 |
|------|------|------|
| Phase 0 파이프라인 검증 (GSM8K) | ✅ 완료 | 2026-04-21, T4 GPU |
| Phase 1 태스크 10개 50-step 파일럿 수집 | ✅ 완료 | 2026-04-22~24 |
| H1a/H1b 파일럿 분석 | ✅ 완료 (Inconclusive) | 2026-04-25 |
| **개정 10-task 1000-step 수집** | ✅ 완료 | 10/10 완료 (2026-04-29) |
| **revised H1a/H1b 분석 (10-task)** | ✅ 완료 | H1a inconclusive, H1b pass (2026-04-29) |
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
| OpenbookQA | 23.14 | 2919s | 64 | 64 | ✅ 완료, `step 460` 조기중단, best reward `0.9250 @ step 280` |
| ARC-Easy | 23.12 | 2594s | 64 | 64 | ✅ 완료, `step 350` 조기중단, best reward `1.0000 @ step 170` |
| WinoGrande | 23.01 | 2068s | 64 | 64 | ✅ 완료, `step 270` 조기중단, best reward `0.6500 @ step 90` |
| SVAMP | 23.11 | 13446s | 256 | 256 | ✅ 완료, `step 290` runtime cap, best reward `0.9500 @ step 220` |
| HumanEval | 23.17 | 13774s | 256 | 256 | ✅ 완료, `step 230` runtime cap, best reward `0.7125 @ step 200` |
| MBPP | 23.19 | 13450s | 256 | 256 | ✅ 완료, `step 250` runtime cap, best reward `0.6750 @ step 160` |

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

### AMC → OpenbookQA (대체, 2026-04-27)
- **사유**: AMC 64-token(v5), 96-token(v7) 모두 step 190 조기중단. best reward 0.1125~0.1250 @ step 10, 이후 개선 없음. `kaggle-aimo/amc_filtered`가 Qwen2.5-1.5B-Instruct 수준에서 reward 신호를 생성하지 못함. preregistration §4.5 exclusion criterion 2 충족.
- **대체**: `allenai/openbookqa` — 과학 상식 MCQ, 출력 짧음, ARC-Challenge와 유사한 reward 구조

### AMC + MATH500 → SVAMP (2개 → 1개 수학 태스크로 통합, 2026-04-27)
- **사유**: MATH500(`HuggingFaceH4/MATH-500`)은 competition math 하위셋으로 AMC와 동일한 실패 패턴 예상. MATH(1000-step)에서도 step 100~190 reward 대부분 0.0000 관측.
- **수학 군집 유지 필요**: AMC/MATH500 둘 다 제외 시 수학 태스크가 GSM8K 1개뿐이 됨 → H1a 군집 분석에 불리.
- **대체**: `ChilleD/SVAMP` — GSM8K와 동일한 word problem 구조, `final_number_exact_match` reward 재사용 가능, 구현 비용 최소.

### max_completion_length 조정 기록
- AMC 64-token (v5): near-zero reward → 실패
- AMC 96-token (v7): near-zero reward → 실패 (토큰 길이가 아닌 태스크 난이도 문제 확인)

---

## 개정 exploratory 10개 태스크 목록 (v2 — 2026-04-27)

| # | Task | Domain | 상태 | max_completion |
|---|------|--------|------|----------------|
| 1 | CommonsenseQA | 상식 추론 | ✅ 완료 | 256 |
| 2 | ARC-Challenge | 과학 MCQ | ✅ 완료 | 256 |
| 3 | HellaSwag | 자연어 추론 | ✅ 완료 | 256 |
| 4 | GSM8K | 수학 word problem | ✅ 완료 | 256 |
| 5 | HumanEval | 코드 생성 | ✅ 완료 | 256 |
| 6 | MBPP | 코드 생성 | ✅ 완료 | 256 |
| 7 | **SVAMP** | 수학 word problem | ✅ 완료 | 256 |
| 8 | **OpenbookQA** | 과학 상식 MCQ | ✅ 완료 | 64 |
| 9 | ARC-Easy | 과학 MCQ | ✅ 완료 | 64 |
| 10 | WinoGrande | 언어/상식 추론 | ✅ 완료 | 64 |

제외 기록:
- ~~MATH~~ → ARC-Easy (2026-04-25, competition math reward 신호 없음)
- ~~AIME~~ → WinoGrande (2026-04-25, 고난도 + 긴 런타임)
- ~~AMC~~ → OpenbookQA (2026-04-27, 64/96-token 모두 step 190 조기중단)
- ~~MATH500~~ → SVAMP (2026-04-27, competition math 동일 패턴 예상, 수학 군집 유지 위해 SVAMP로 대체)

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
- `완료 5태스크` precheck:
  - mean `|cos| = 0.00131`
  - max `|cos| = 0.00393`
  - `r90 = 5`, `rho = 1.0`

현재 판단:
- `H1b`는 우호적이다. 풀런 벡터들은 서로 매우 다른 방향을 보인다.
- `H1a`는 아직 우호적이라고 보기 어렵다. 다만 반박으로 결론낼 단계도 아니다.
- 5개만 봐도 공통 저차원 subspace는 아직 뚜렷하지 않다.
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
| 이번 대화 요약 | `runs/2026-04-26-h1a-prognosis/conversation_summary.md` |

---

## 2026-04-27 업데이트

### AMC 실험 결과
- 64-token (v5): best reward 0.1125 @ step 10, step 190 조기중단
- 96-token (v7): best reward 0.1250 @ step 10, step 190 조기중단
- 판정: **제외 확정** — reward 신호 부재 (preregistration §4.5 criterion 2)
- run record: `runs/2026-04-27-phase1-amc-qwen2.5-1.5b/`

### 태스크 목록 개정 (v2)
- AMC → **OpenbookQA** 대체
- MATH500 → **SVAMP** 대체 (수학 군집 유지 목적)
- 수학 도메인: GSM8K + SVAMP 2개 유지

### OpenbookQA 1000-step 풀런 결과 (2026-04-27)
- Kaggle T4에서 `max_steps=1000`으로 실행, 실제 종료는 `step 460`
- 종료 이유: `180 step` 동안 reward 최고값 갱신 없음
- 최고 reward: `0.9250 @ step 280`
- 마지막 reward: `0.5125 @ step 460`
- 평균 reward: `0.6916`
- 실제 step 시간: `6.34s/step` (2918.5s / 460 step) — 최단 기록
- norm: `23.14` — 다른 완료 태스크들과 일치
- run record: `runs/2026-04-27-phase1-openbookqa-qwen2.5-1.5b/`

### WinoGrande 1000-step 풀런 결과 (2026-04-27)
- Kaggle T4에서 `max_steps=1000`으로 실행, 실제 종료는 `step 270`
- 종료 이유: `180 step` 동안 reward 최고값 갱신 없음
- 최고 reward: `0.6500 @ step 90`
- 마지막 reward: `0.4000 @ step 270`
- 평균 reward: `0.5005`
- 실제 step 시간: `7.66s/step` (2067.7s / 270 step)
- norm: `23.01` — 타 완료 태스크(23.11~23.21) 대비 소폭 낮음, 사용 가능
- 해석: binary-choice(1/2) 태스크, random baseline=0.5. 초기 상승 후 수렴 실패 → GRPO 신호 약함. 벡터 자체는 non-degenerate.
- run record: `runs/2026-04-27-phase1-winogrande-qwen2.5-1.5b/`

### 4h Kaggle 할당량 제약 대응
- HumanEval, MBPP, SVAMP: `MAX_RUNTIME_MINUTES=220` 적용 → 3h40m 훈련 후 clean stop
- 짧은 태스크(ARC-Easy, WinoGrande)는 4h 내 완주 가능

---

## 2026-04-28 업데이트

### SVAMP 1000-step 풀런 결과 (2026-04-28)
- Kaggle T4에서 `max_steps=1000`으로 실행, 실제 종료는 `step 290`
- 종료 이유: runtime 224.1m가 MAX_RUNTIME_MINUTES=220 초과
- 최고 reward: `0.9500 @ step 220` — 전체 태스크 중 최고
- 마지막 reward: `0.7875 @ step 290`
- 평균 reward: `0.8103` — 전체 태스크 중 최고
- 실제 step 시간: `46.37s/step` (13446.1s / 290 step)
- norm: `23.11` — 정상 범위
- 해석: GSM8K와 동일한 reward 함수. step 60에 이미 0.8375 달성, 빠른 수렴. reward 여전히 활성 상태에서 runtime cap으로 종료.
- run record: `runs/2026-04-28-phase1-svamp-qwen2.5-1.5b/`

---

## 2026-04-29 업데이트

### HumanEval 1000-step 풀런 결과 (2026-04-29)
- Kaggle T4에서 `max_steps=1000`으로 실행, 실제 종료는 `step 230`
- 종료 이유: runtime 229.6m가 MAX_RUNTIME_MINUTES=220 초과
- 최고 reward: `0.7125 @ step 200`
- 마지막 reward: `0.5375 @ step 230`
- 평균 reward: `0.5190`
- 실제 step 시간: `59.89s/step` (13774.1s / 230 step)
- norm: `23.17` — 정상 범위
- 해석: 코드 생성 태스크, pass@1 reward. 1.5B 모델 치고 양호한 수렴. reward 노이즈하게 진행(0.31~0.71). 벡터 non-degenerate.
- run record: `runs/2026-04-29-phase1-humaneval-qwen2.5-1.5b/`

### MBPP 1000-step 풀런 결과 (2026-04-29)
- Kaggle T4에서 `max_steps=1000`으로 실행, 실제 종료는 `step 250`
- 종료 이유: runtime 224.2m가 MAX_RUNTIME_MINUTES=220 초과
- 최고 reward: `0.6750 @ step 160`
- 마지막 reward: `0.4375 @ step 250`
- 평균 reward: `0.4675`
- 실제 step 시간: `53.80s/step` (13450.3s / 250 step)
- norm: `23.19` — 전 태스크 중 최고, 정상 범위
- 해석: 코드 생성 태스크, pass@1 reward. HumanEval과 동일한 노이즈 패턴(0.31~0.68). 벡터 non-degenerate.
- run record: `runs/2026-04-29-phase1-mbpp-qwen2.5-1.5b/`

### 🎉 10/10 수집 완료 (2026-04-29)

| 태스크 | norm | best reward |
|--------|------|-------------|
| CommonsenseQA | 23.16 | — |
| ARC-Challenge | 23.17 | — |
| HellaSwag | 23.11 | — |
| GSM8K | 23.21 | 0.9125 |
| OpenbookQA | 23.14 | 0.9250 |
| ARC-Easy | 23.12 | 1.0000 |
| WinoGrande | 23.01 | 0.6500 |
| SVAMP | 23.11 | 0.9500 |
| HumanEval | 23.17 | 0.7125 |
| MBPP | 23.19 | 0.6750 |

---

## 2026-04-29 H1a/H1b 분석 결과 (10-task revised exploratory)

| 지표 | 값 |
|------|-----|
| num_tasks | 10 |
| r90 | 9 |
| ρ | 0.9000 |
| ρ CI₉₅ | [0.40, 0.70] |
| **H1a 판정** | **inconclusive (weak fail)** |
| H1b mean\|cos\| | 0.07375 |
| H1b max\|cos\| | 0.33535 |
| **H1b 판정** | **pass** |

해석:
- r90=9 → 10개 벡터가 거의 전 차원을 점유. 공통 저차원 subspace 없음.
- H1b pass → 벡터들 서로 거의 수직. 태스크 특화성 확인.
- 결론: 공통 기하 구조(H1a) 없이 태스크별 독립 방향(H1b pass). H1a는 사실상 weak fail.
- run record: `runs/2026-04-29-phase1-h1a-h1b-revised/`

---

## 다음 액션 (우선순위 순)

1. H2 전이 실험 설계 및 실행
