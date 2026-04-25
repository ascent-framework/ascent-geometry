# Full Run Plan — 1000-step H1a/H1b Collection (Revised 2026-04-25)

## Objective

Collect 1000-step update vectors for 10 tasks to enable a valid H1a/H1b decision.
Priority: getting real signal that tests whether ASCENT holds, not strict OSF compliance.

## Core Settings (unchanged from v1.3)

- Steps: 1000 per task
- Method: GRPO, AdamW, lr=1e-4
- LoRA: r=8, alpha=16
- Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Hardware: T4 GPU (Kaggle), bf16

## Task Plan

### Group A — Confirmed working (3 tasks complete)

| Task | Run dir | Norm | Time |
|------|---------|------|------|
| CommonsenseQA | 2026-04-25-phase1-commonsenseqa-qwen2.5-1.5b | 23.16 | 5955s ✅ |
| ARC-Challenge | 2026-04-25-phase1-arc-challenge-qwen2.5-1.5b | 23.17 | 7470s ✅ |
| HellaSwag | 2026-04-25-phase1-hellaswag-qwen2.5-1.5b | 23.11 | 12638s ✅ |

### Group B — Run as-is, expected to work (3 tasks)

| Task | Domain | Notes |
|------|--------|-------|
| GSM8K | Math reasoning | 50-step reward=0.043, short answers |
| HumanEval | Code | Short completions, clear reward signal |
| MBPP | Code | Short completions, clear reward signal |

### Group C — max_completion_length=64 test (2 tasks)

These tasks have long generation times (~65s/step). Reducing max_completion_length
256→64 is NOT specified in v1.3, so it is a deviation, but it does not change
the training method or LoRA config. Logged explicitly below.

| Task | Concern | Strategy |
|------|---------|----------|
| AMC | 65.7s/step at 256 tokens, reward uncertainty | Run with max_completion_length=64, monitor reward at step 50 |
| MATH500 | 65.6s/step, competition_math subset too hard | Run with max_completion_length=64, monitor reward at step 50 |

**Deviation note**: `max_completion_length=64` for AMC and MATH500.
Rationale: parameter not registered in v1.3; speeds up T4 run from ~18h to ~5h per task.
If reward=0 at step 50, apply task exclusion criterion and replace with Group D.

### Group D — Replacement tasks (replacing MATH and AIME)

MATH: excluded at step 190 (reward=0 for 89% of rollouts, exceeds 80% criterion).
AIME: expected to fail — competition olympiad problems exceed 1.5B reasoning capacity.

Replacements chosen for domain diversity and confirmed trainability:

| Original | Replacement | Dataset | Domain | Rationale |
|----------|-------------|---------|--------|-----------|
| MATH | winogrande | allenai/winogrande, winogrande_xl split | Language/commonsense reasoning | Fast reward signal, diverse from other tasks |
| AIME | piqa | ybisk/piqa | Physical intuition | Short answers, reliable reward signal |

**Replacement justification**: ASCENT tests whether task adaptation produces
geometrically distinct update vectors. Domain diversity is what matters —
replacing two math competition tasks with two commonsense/reasoning tasks
preserves the diversity requirement while enabling actual training signal.

## Estimated Time (revised)

| Task | est. h | max_completion_length |
|------|-------|-----------------------|
| CommonsenseQA | ✅ done | 256 |
| ARC-Challenge | ✅ done | 256 |
| HellaSwag | ✅ done | 256 |
| GSM8K | ~16h | 256 |
| HumanEval | ~17.7h | 256 |
| MBPP | ~16h | 256 |
| AMC | ~5h | 64 (deviation) |
| MATH500 | ~5h | 64 (deviation) |
| winogrande | ~4h (est.) | 64 |
| piqa | ~4h (est.) | 64 |

Total remaining: ~68 GPU-hours (~2.3 weeks on Kaggle free tier)

## Fallback — if AMC/MATH500 fail at 64 tokens

If reward=0 at step 50 with max_completion_length=64, replace with:
- AMC → `openbookqa` (science QA, short answers)
- MATH500 → `social_i_qa` (social reasoning, short answers)

## Post-Collection Steps

1. Download all 10 `update_vector.npy` from Kaggle outputs
2. Run: `python3 analysis/h1a_h1b_task_matrix.py --vector ...`
3. Save registered H1a/H1b report to `runs/`
