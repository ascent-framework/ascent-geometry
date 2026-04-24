# ASCENT-G 현황 문서

**작성일**: 2026-04-25  
**브랜치**: `codex/prepare-h1a-h1b-analysis`  
**모델**: `Qwen/Qwen2.5-1.5B-Instruct`

---

## 전체 진행 상황

| 단계 | 상태 | 비고 |
|------|------|------|
| Phase 0 파이프라인 검증 (GSM8K) | ✅ 완료 | 2026-04-21, T4 GPU |
| Phase 1 태스크 10개 50-step 파일럿 수집 | ✅ 완료 | 2026-04-22~24 |
| H1a/H1b 파일럿 분석 | ✅ 완료 (Inconclusive) | 2026-04-25 |
| **H1a/H1b 등록 분석 (1000-step)** | ⏳ 미완 | 현재 블로커 |
| H2 전이 실험 | ⏳ 대기 | H1a/H1b 이후 |

---

## Phase 0 파이프라인 검증

- **결과**: `train → extract → SVD` 파이프라인 전체 정상 작동 확인
- **실행 환경**: Kaggle, Tesla T4 GPU, bf16
- **근거**: `runs/2026-04-22-phase0-gsm8k-qwen2.5-1.5b/`

---

## Phase 1 — 50-step 파일럿 수집 결과

모든 10개 태스크를 50스텝으로 실행. 결과 요약:

| Task | Steps | Vector Norm | sec/step | 1000step 예상 |
|------|------:|------------:|---------:|--------------:|
| GSM8K | 50 | 22.89 | 58.2s | ~16h |
| CommonsenseQA | 50 | — | 4.7s | ~1.3h |
| AIME | 50 | 22.87 | 72.1s | ~20h |
| ARC-Challenge | 50 | 22.89 | 6.5s | ~1.8h |
| HellaSwag | 50 | 22.88 | 13.5s | ~3.7h |
| MATH | 50 | — | 45.9s | ~12.8h |
| MBPP | 50 | 22.89 | 57.9s | ~16.1h |
| AMC | 50 | 22.88 | 65.7s | ~18.3h |
| HumanEval | 50 | 22.89 | 63.8s | ~17.7h |
| MATH500 | 50 | 22.89 | 65.6s | ~18.2h |

**총 1000-step 예상 GPU 시간**: ~126 Kaggle GPU-hours

### 관찰된 패턴 (50-step 기준)
- 모든 태스크 벡터 norm이 22.87~22.89로 거의 동일 → **초기화 노이즈 지배**
- CommonsenseQA / ARC-Challenge는 데이터셋이 작아 스텝당 시간이 매우 짧음 (5~7s)
- AIME / AMC / MATH500 수학 태스크는 생성 길이가 길어 스텝당 시간이 김 (65~72s)

---

## H1a/H1b 파일럿 분석 결과

**판정: Inconclusive** — 50-step 벡터로는 등록 가설 판정 불가

| 지표 | 값 | 해석 |
|------|-----|------|
| `r_90` | 9 / 10 | 거의 등방성 — 태스크 구분 안 됨 |
| `rho` | 0.90 | 표면상 높지만 의미 없음 |
| Bootstrap 95% CI | [0.4, 0.7] | 실제 신뢰구간은 낮음 |
| H1b mean `|cos|` | 0.086 | 태스크 방향은 서로 다름 (pass) |
| H1b max `|cos|` | 0.314 | — |

### 왜 Inconclusive인가?

LoRA 초기화 구조: W_A는 랜덤, W_B는 0. 50스텝 후에도 초기화 랜덤 노이즈가
태스크별 학습 신호보다 크다. 따라서 10개 태스크 벡터가 구조적으로 거의 동일하게
보이며 SVD 스펙트럼이 평탄(uniform ~1.0)해진다.

**필요한 것**: 최소 200~300스텝 이상의 충분한 학습 → 태스크별 신호가 초기화 노이즈를 압도

근거 파일: `runs/2026-04-25-phase1-h1a-h1b-pilot/`

---

## 현재 블로커: 1000-step Full Run

### 등록 기준 (v1.3)
- 스텝 수: **1000** (현재 50 — 미충족)
- 방법: GRPO, AdamW, lr=1e-4, LoRA r=8 alpha=16

### 옵션 비교

| 옵션 | 스텝 | 등록 여부 | 총 GPU 시간 | Kaggle 무료 주수 |
|------|------|-----------|------------|----------------|
| 현행 유지 (50-step) | 50 | ❌ pilot-only | 완료 | — |
| 탐색용 중간 run | 200~300 | ❌ exploratory | ~40~60h | 2~3주 |
| **등록 full run** | **1000** | **✅** | **~126h** | **~4~5주** |

### Kaggle 무료 할당량
- 주당 30 GPU-hours
- 126h ÷ 30h/주 ≈ **약 4~5주** (순차 실행 시)
- 병렬 실행 시 (계정 분산 또는 Pro): ~16~20시간 wall-clock

### 태스크별 우선순위 (시간 기준)
빠른 태스크부터 실행해 부분 결과라도 조기 확보:

1. CommonsenseQA (~1.3h) — 가장 빠름
2. ARC-Challenge (~1.8h)
3. HellaSwag (~3.7h)
4. MATH (~12.8h)
5. MBPP (~16.1h)
6. GSM8K (~16.2h)
7. HumanEval (~17.7h)
8. MATH500 (~18.2h)
9. AMC (~18.3h)
10. AIME (~20h)

---

## 아티팩트 위치

| 아티팩트 | 위치 |
|----------|------|
| 50-step 파일럿 벡터 | Kaggle kernel outputs (각 `chson0316/ascent-g-phase-0-pilot-{task}-qwen2-5-1-5b`) |
| 로컬 다운로드 (임시) | `/tmp/vectors/{task}/` (비영구) |
| 어댑터 체크포인트 | Kaggle kernel outputs |
| 런 리포트 | `runs/2026-04-{date}-phase0-{task}-qwen2.5-1.5b/` |
| H1a/H1b 파일럿 리포트 | `runs/2026-04-25-phase1-h1a-h1b-pilot/` |

---

## 다음 액션

1. **단기**: 노트북 `max_steps=50 → 1000` 수정
2. **단기**: CommonsenseQA, ARC-Challenge부터 Kaggle full run 시작 (빠른 것 먼저)
3. **중기**: 10개 태스크 full run 완료 후 `h1a_h1b_task_matrix.py` 재실행
4. **장기**: H1a/H1b 등록 판정 → H2 전이 실험으로 이동
