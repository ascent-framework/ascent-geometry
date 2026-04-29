# Codex Adversarial Review — 한국어 정리 (2026-04-29)

## 판정: needs-attention

---

## 핵심 문제

**현재 H1a/H1b 분석 결과는 사전등록(preregistered) 증거로 신뢰할 수 없다.**

`analysis/h1a_h1b_task_matrix.py`는 입력 벡터 수가 10개이기만 하면 자동으로 `scope: "registered"`, `registered_ready: true`를 보고서에 기재한다. 실제로 어떤 태스크가 입력됐는지 검증하지 않는다.

---

## 구체적 결함 [high]

**파일**: `analysis/h1a_h1b_task_matrix.py`, line 191–212

이번 실행에서 입력한 태스크:
- OpenbookQA, ARC-Easy, WinoGrande, SVAMP ← **대체 태스크** (exploratory)

사전등록된 원래 태스크:
- MATH, AIME, AMC, MATH500 ← **등록 태스크** (registered)

그러나 생성된 보고서(`runs/2026-04-29-phase1-h1a-h1b-revised/h1a_h1b_report.json`)에는 `scope: "registered"`로 표기돼 있다.

이 구조적 취약점은 downstream 요약, 자동화 파이프라인, 논문 작성 과정에서 exploratory 증거를 사전등록 1차 분석으로 오인할 수 있다.

---

## 조치 사항

| 우선순위 | 조치 |
|----------|------|
| 즉시 | `h1a_h1b_report.json`의 `scope`를 `"exploratory"`로 재생성 또는 수동 수정 |
| 단기 | 분석 스크립트에 등록 태스크 명단 검증 로직 추가 — 이름이 불일치하면 `scope="exploratory"`로 강제 전환 |
| 중기 | SHA/provenance 검증을 registered 경로의 필수 입력으로 설정 |

---

## 현재 보고서의 올바른 해석

이번 `runs/2026-04-29-phase1-h1a-h1b-revised/` 결과는:

- **라벨**: `revised exploratory H1a/H1b` (등록 경로 아님)
- **입력**: 운영 제약으로 대체된 10개 태스크
- **유효성**: exploratory 참고 증거로는 유효하나, preregistered primary analysis로 인용 불가

run_note.md와 STATUS.md에 이미 "등록 태스크 세트 대체안이 아니라 운영 제약을 반영한 exploratory 후보 목록"이라고 명시돼 있다. 문서 수준에서는 올바르게 라벨링됐으나, **보고서 JSON 파일 자체가 `scope: registered`를 emit하는 것이 문제**다.

---

## 결론

실험 설계와 운영 기록 자체의 문제가 아니라, 분석 스크립트의 `registered_ready` 판단 로직이 태스크 수만 보고 태스크 명단을 확인하지 않는 구조적 결함이다. 분석 스크립트 수정 후 보고서를 재생성해야 한다.
