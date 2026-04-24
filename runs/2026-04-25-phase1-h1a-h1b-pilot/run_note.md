# H1a/H1b Pilot Analysis — 2026-04-25

## Status: INCONCLUSIVE (pilot-only, not registered evidence)

## Scope

- Input: 10 registered tasks, 50-step pilot vectors
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Method: SVD on normalized concat(ΔW_A, ΔW_B) vectors

## Results

### H1a
- `r_90 = 9`, `rho = 0.90`
- Bootstrap 95% CI: [0.4, 0.7]
- Bootstrap r_90 mean: 5.52
- **Decision: inconclusive**

### H1b
- Mean |cos| = 0.086, Max |cos| = 0.314 (45 pairs)
- **Decision: pass** (tasks are directionally distinct)

## Why Inconclusive

All 10 task vectors have nearly identical norms (22.87–22.92). Singular values
are uniformly ~1.0, meaning the normalized task matrix is near-isotropic.

Root cause: 50-step pilots produce vectors dominated by random LoRA
initialization (W_A is random, W_B starts at zero). After only 50 gradient
steps the task-specific signal is weak relative to the shared initialization
structure. The vectors are not meaningfully separated in direction space.

## Required Fix

Rerun all 10 tasks with **1000 gradient steps** (registered full run length
per v1.3). At 58 sec/step on T4, each task requires ~16 hours of Kaggle GPU
time. Full collection = 10 × 16h ≈ 160 Kaggle GPU-hours across 10 kernels.

## Artifact Paths

- Report: `runs/2026-04-25-phase1-h1a-h1b-pilot/h1a_h1b_report.json`
- Input vectors: `/tmp/vectors/{task}/` (not committed — re-download from Kaggle)

## Next Step

Schedule 1000-step full runs on Kaggle for all 10 registered tasks and
re-run `analysis/h1a_h1b_task_matrix.py` on the resulting vectors.
