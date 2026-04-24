# Full Run Plan — 1000-step H1a/H1b Collection

## Objective

Collect registered update vectors (1000 gradient steps) for all 10 tasks
to enable a valid H1a/H1b decision.

## Registered Settings (v1.3)

- Steps: 1000 per task
- Method: GRPO, AdamW, lr=1e-4
- LoRA: r=8, alpha=16
- Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Hardware: T4 GPU (Kaggle), bf16

## Estimated Time

- Per task: ~1000 × 58s = ~16 hours on T4
- Total: 10 tasks × 16h = ~160 Kaggle GPU-hours
- Parallelism: run all 10 kernels simultaneously → wall-clock ~16 hours

## Task Queue

| Task | Kernel | Est. Hours | Status |
|------|--------|-----------|--------|
| GSM8K | chson0316/ascent-g-phase-0-pilot-gsm8k-qwen2-5-1-5b | ~16h | pending |
| MATH | chson0316/ascent-g-phase-0-pilot-math-qwen2-5-1-5b | ~16h | pending |
| AIME | chson0316/ascent-g-phase-0-pilot-aime-qwen2-5-1-5b | ~16h | pending |
| AMC | chson0316/ascent-g-phase-0-pilot-amc-qwen2-5-1-5b | ~16h | pending |
| MATH500 | chson0316/ascent-g-phase-0-pilot-math500-qwen2-5-1-5b | ~16h | pending |
| HumanEval | chson0316/ascent-g-phase-0-pilot-humaneval-qwen2-5-1-5b | ~16h | pending |
| MBPP | chson0316/ascent-g-phase-0-pilot-mbpp-qwen2-5-1-5b | ~16h | pending |
| CommonsenseQA | chson0316/ascent-g-phase-0-pilot-commonsenseqa-qwen2-5-1-5b | ~16h | pending |
| HellaSwag | chson0316/ascent-g-phase-0-pilot-hellaswag-qwen2-5-1-5b | ~16h | pending |
| ARC-Challenge | chson0316/ascent-g-phase-0-pilot-arc-challenge-qwen2-5-1-5b | ~16h | pending |

## Notebook Change Required

Update `max_steps=50` → `max_steps=1000` in the GRPOConfig cell of each
task notebook before pushing to Kaggle.

## Kaggle Quota Note

Kaggle free tier: 30 GPU hours/week per account. 160 hours requires either:
- Multiple accounts / Kaggle Pro, OR
- Staggered schedule: ~5 tasks per week × 2 weeks

## Post-Collection Steps

1. Download all 10 `update_vector.npy` from Kaggle outputs
2. Verify SHA-256 against provenance files
3. Run: `python3 analysis/h1a_h1b_task_matrix.py --vector ...`
4. Save registered H1a/H1b report to `runs/`
