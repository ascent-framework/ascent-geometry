# AMC — Phase 1 Full Run (2026-04-27)

## Status: EARLY-STOPPED — 제외 후보

## 실행 이력

| 버전 | max_completion_length | actual_steps | best reward | 조기중단 이유 |
|------|----------------------|-------------|------------|-------------|
| v5 (2026-04-26) | 64 | 190 | 0.1125 @ step 10 | 180 step 개선 없음 |
| v7 (2026-04-27) | 96 | 190 | 0.1250 @ step 10 | 180 step 개선 없음 |

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training (v7 기준)
- Requested steps: 1000
- Actual completed steps: 190
- Total time: 5,227.4s (1.45h)
- Per-step time: 27.51s
- Best reward: 0.1250 @ step 10
- Early-stop: no reward improvement for 180 steps

## Update Vector (v7 기준)
- Shape: [9,232,384]
- Norm: 23.0622
- SHA-256: ca1428f774791108222216cd0071877a9e2c92d0a91b464f9e1d7e841e16c73b
- r90 mean: 6.61, range: 5–7

## 관찰

reward가 step 10에서 0.1250을 찍은 뒤 전혀 개선되지 않고 0~0.08 사이를 랜덤하게 오갔다.
64→96 토큰으로 늘려도 패턴이 동일했으므로 **truncation 문제가 아니라 태스크 난이도 문제**다.

`kaggle-aimo/amc_filtered`는 AMC 경시 문제로 Qwen2.5-1.5B-Instruct 수준에서
안정적인 reward 신호를 생성하기 어렵다.

## Preregistration §4.5 exclusion 검토

> "Reward signal is absent: reward = 0 for > 80% of rollouts at step 200"

step 190 로그 기준으로 대부분의 step에서 reward가 0.00~0.06 수준.
best reward도 step 10 이후 갱신 없음 → exclusion criterion 2 충족에 가깝다.

벡터 자체는 non-degenerate하게 추출됐으나, reward 신호가 없으므로
이 벡터가 의미있는 태스크 적응을 반영하는지 불명확하다.

## 다음 액션 후보

1. **AMC 제외 확정** — exclusion note 작성, WinoGrande 또는 ARC-Easy로 대체
2. **데이터셋 교체 재시도** — 다른 AMC 소스 또는 더 쉬운 수학 경시 문제 사용

## Artifact 위치

- run_report.json: `/tmp/amc-phase1-output/amc-qwen2.5-1.5b-phase1/run_report.json`
- update_vector.npy: `/tmp/amc-phase1-output/amc-qwen2.5-1.5b-phase1/update_vector.npy`
- source kernel: `chson0316/ascent-g-phase-0-pilot-amc-qwen2-5-1-5b` (v7)
