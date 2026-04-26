# Full Run Plan — Revised Exploratory 10-Task Collection (2026-04-25)

## Objective

Collect 1000-step update vectors for a revised 10-task set and run an exploratory
H1a/H1b-style analysis on the normalized matrix.

This plan is not the preregistered v1.3 primary analysis. It is an operational
revision motivated by runtime limits and weak reward signal on some original tasks.

## Core Settings Carried Over From v1.3

- Steps: 1000 per task
- Method: GRPO, AdamW, lr=1e-4
- LoRA: r=8, alpha=16
- Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Hardware: T4 GPU (Kaggle), bf16

## Task Plan

### Group A — Confirmed working (3 tasks complete)

| Task | Run dir | Norm | Time | Executed | Recommended |
|------|---------|------|------|----------|-------------|
| CommonsenseQA | 2026-04-25-phase1-commonsenseqa-qwen2.5-1.5b | 23.16 | 5955s ✅ | 256 | 64 |
| ARC-Challenge | 2026-04-25-phase1-arc-challenge-qwen2.5-1.5b | 23.17 | 7470s ✅ | 256 | 64 |
| HellaSwag | 2026-04-25-phase1-hellaswag-qwen2.5-1.5b | 23.11 | 12638s ✅ | 256 | 64~96 |

### Group B — Run as-is, expected to work (3 tasks)

| Task | Domain | Recommended max_completion_length | Notes |
|------|--------|-----------------------------------|-------|
| GSM8K | Math reasoning | 256 | Final-number reward needs enough room for reasoning |
| HumanEval | Code | 256 | Shorter caps risk truncating valid code solutions |
| MBPP | Code | 256 | Test-execution reward path benefits from full function bodies |

### Group C — max_completion_length=64 test (2 tasks)

These tasks have long generation times (~65s/step). Reducing max_completion_length
256→64 is an exploratory deviation from the registered setup. It does not change
the optimizer or LoRA config, but it does change generation behavior and must be
labeled as such in all downstream notes and reports.

| Task | Concern | Strategy |
|------|---------|----------|
| AMC | 65.7s/step at 256 tokens, reward uncertainty | Run with max_completion_length=64, monitor reward at step 50, raise to 96 only if clearly truncated |
| MATH500 | 65.6s/step, competition_math subset too hard | Prefer max_completion_length=96; use 64 only if runtime pressure dominates |

**Deviation note**: `max_completion_length=64` for AMC and MATH500.
Rationale: speeds up T4 run from ~18h to ~5h per task.
Decision rule: if reward remains effectively zero through the early run window,
do not use the task in the revised exploratory set.

### Group D — Replacement tasks (replacing MATH and AIME)

MATH: runtime and reward behavior were poor in the observed run log.
From step 100 to 190, reward was mostly `0.0000`, with only sparse `0.0500`
events, and step 190 already reached elapsed `202.1m` with eta `861.4m`.

AIME: not yet excluded by an equivalent observed run log in this repository.
It is treated here as a high-risk candidate for weak reward and long runtime,
so replacement remains an exploratory planning choice rather than a preregistered
exclusion.

Replacements are chosen for domain diversity and likely trainability. They are
now implemented in the CLI path, though task-specific Kaggle notebooks still
need to be created:

| Original | Replacement | Dataset | Domain | Recommended max_completion_length | Rationale |
|----------|-------------|---------|--------|-----------------------------------|-----------|
| MATH | ARC-Easy | allenai/ai2_arc, ARC-Easy split | Science commonsense QA | 64 | Existing ARC-style formatter can be reused with minimal changes |
| AIME | WinoGrande | allenai/winogrande | Language/commonsense reasoning | 64 | Binary-choice format, short outputs, likely faster reward signal |

**Replacement justification**: if the goal is to test whether task adaptation
produces geometrically distinct update vectors under a practical compute budget,
domain-diverse tasks with short answers are a reasonable exploratory substitute.
They should not be described as the preregistered primary set without a separate
amendment record.

## Estimated Time (revised)

| Task | est. h on T4 | Planned max_completion_length | Recommended max_completion_length |
|------|---------------|------------------------------|----------------------------------|
| CommonsenseQA | ✅ done (1.65h) | 256 | 64 |
| ARC-Challenge | ✅ done (2.07h) | 256 | 64 |
| HellaSwag | ✅ done (3.51h) | 256 | 64~96 |
| GSM8K | ~16h | 256 | 256 |
| HumanEval | ~17.7h | 256 | 256 |
| MBPP | ~16.1h | 256 | 256 |
| AMC | ~5~7h | 64 (deviation) | 64, fallback 96 |
| MATH500 | ~5~7h at 64 / ~7~9h at 96 | 64 (deviation) | 96 preferred |
| ARC-Easy | ~1.5~2.5h (est.) | 64 | 64 |
| WinoGrande | ~2~4h (est.) | 64 | 64 |

Total remaining: ~68 GPU-hours (~2.3 weeks on Kaggle free tier)

## Fallback — if AMC/MATH500 fail at 64 tokens

If reward=0 at step 50 with max_completion_length=64, replace with:
- AMC → `openbookqa` (science QA, short answers)
- MATH500 → `piqa` (physical commonsense, short binary-choice answers)

These are also unimplemented candidates in the current repository and should be
treated as backlog items, not ready-to-run tasks.

## Post-Collection Steps

1. Download all 10 `update_vector.npy` from Kaggle outputs
2. Run: `python3 analysis/h1a_h1b_task_matrix.py --vector ...`
3. Save revised exploratory H1a/H1b report to `runs/`
