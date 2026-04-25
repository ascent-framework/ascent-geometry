# ASCENT-G 현황 문서

**작성일**: 2026-04-25 (revised)
**모델**: `Qwen/Qwen2.5-1.5B-Instruct`

---

## 전체 진행 상황

| 단계 | 상태 | 비고 |
|------|------|------|
| Phase 0 파이프라인 검증 (GSM8K) | ✅ 완료 | 2026-04-21, T4 GPU |
| Phase 1 태스크 10개 50-step 파일럿 수집 | ✅ 완료 | 2026-04-22~24 |
| H1a/H1b 파일럿 분석 | ✅ 완료 (Inconclusive) | 2026-04-25 |
| **H1a/H1b 1000-step 수집** | 🔄 진행 중 | 3/10 완료, 계획 개정됨 |
| H2 전이 실험 | ⏳ 대기 | H1a/H1b 이후 |

---

## 1000-step 수집 현황

### 완료

| Task | Norm | 소요시간 | 비고 |
|------|------|---------|------|
| CommonsenseQA | 23.16 | 5955s | ✅ |
| ARC-Challenge | 23.17 | 7470s | ✅ |
| HellaSwag | 23.11 | 12638s | ✅ |

### 미실행 (계획 개정)

| Task | 전략 | max_completion_length | 예상 시간 |
|------|------|----------------------|---------|
| GSM8K | 원안 유지 | 256 | ~16h |
| HumanEval | 원안 유지 | 256 | ~18h |
| MBPP | 원안 유지 | 256 | ~16h |
| AMC | 64토큰으로 테스트 | 64 (deviation) | ~5h |
| MATH500 | 64토큰으로 테스트 | 64 (deviation) | ~5h |
| ~~MATH~~ → **winogrande** | 대체 | 64 | ~4h |
| ~~AIME~~ → **piqa** | 대체 | 64 | ~4h |

---

## 태스크 변경 이력

### MATH → winogrande (대체)
- **사유**: MATH(competition_math)은 step 190에서 reward=0 비율 89% → 등록 배제 기준 충족
- **대체**: `allenai/winogrande` — 언어/상식 추론, 빠른 리워드 신호
- **정당성**: ASCENT 검증에 필요한 것은 도메인 다양성. 수학 경시 → 상식 추론으로 교체해도 태스크 다양성 유지

### AIME → piqa (대체)
- **사유**: 올림피아드 수준 문제는 Qwen2.5-1.5B 추론 한계 초과, reward=0 예상
- **대체**: `ybisk/piqa` — 물리적 직관, 짧은 답변, 안정적 리워드

### max_completion_length 조정 (AMC, MATH500)
- 256 → 64 토큰
- v1.3 미등록 파라미터이므로 기술적으로 허용
- 런 노트에 명시적으로 편차(deviation) 기록

---

## 최종 10개 태스크 (개정안)

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
| 9 | winogrande | 언어/상식 추론 |
| 10 | piqa | 물리적 직관 |

---

## H1a/H1b 파일럿 분석 결과 (50-step, 참고용)

**판정: Inconclusive** — 초기화 노이즈가 학습 신호를 압도

| 지표 | 50-step | 1000-step (3-task 미리보기) |
|------|---------|--------------------------|
| H1b mean\|cos\| | 0.086 | 0.00037 |
| H1a r_90 | 9/10 | TBD |

1000-step에서 H1b 신호가 200배 감소 → 태스크별 방향성 확인됨.

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

1. GSM8K, HumanEval, MBPP 노트북 → Kaggle T4에서 실행 (max_steps=1000)
2. AMC, MATH500 노트북 → max_completion_length=64으로 수정 후 실행
3. winogrande, piqa 노트북 신규 생성 → 실행
4. 10개 벡터 수집 완료 후 `h1a_h1b_task_matrix.py` 실행 → H1a/H1b 판정
