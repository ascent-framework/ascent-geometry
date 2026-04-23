# ascent-geometry

ASCENT-G: Adaptation Geometry experiments (Paper 1)

Theory and pre-registration: [ascent-framework](https://github.com/ascent-framework/ascent-framework)

This repository should track the current ASCENT framework registration and
execution docs in `../ascent-framework`, especially:

- `docs/preregistration/v1.3.md`
- `docs/phase0/execution-plan.md`

## Repository structure

| Directory | Purpose |
|-----------|---------|
| `training/` | GRPO training scripts for registered tasks |
| `extraction/` | Update vector extraction from trained adapters |
| `analysis/` | SVD geometry analysis (`r_90`, spectra) |
| `statistics/` | Hypothesis tests — Phase 1+ only |
| `notebooks/` | Kaggle-ready pilot notebooks |
| `runs/` | Run logs and small derived outputs (no checkpoints) |

## Phase 0 target

One complete pilot: `train → extract → SVD` on `Qwen2.5-1.5B-Instruct` / `GSM8K`.

See [`ascent-framework/docs/phase0/execution-plan.md`](https://github.com/ascent-framework/ascent-framework/blob/main/docs/phase0/execution-plan.md) for acceptance criteria.

## Current status

- One complete Phase 0 pilot run has been completed on Kaggle for
  `Qwen2.5-1.5B-Instruct` / `GSM8K`, with imported run records under
  `runs/2026-04-22-phase0-gsm8k-qwen2.5-1.5b/`.
- Additional pilot runs have been completed for `CommonsenseQA` and `MATH`,
  and the task-aware training path now covers `HellaSwag` as well.
- Four valid pilot vectors now exist across `GSM8K`, `CommonsenseQA`, `MATH`,
  and `HellaSwag`; the imported run records live under `runs/`.
- The task-aware training path now also covers `ARC-Challenge`, with launcher
  notebooks and task-specific Kaggle metadata prepared.
- The task-aware training path now also covers `AIME`, using the `AIME 2025`
  dataset with a numeric final-answer reward.
- `MBPP` is now wired through the same task-aware code path, and its Kaggle
  pilot run has completed successfully after aligning the notebook with the
  dataset's `text`/`code` schema.
- `MBPP` is retained as a logged pilot capture, but it is excluded from the
  geometry analysis set because the reward remained flat and the SVD diagnostic
  was degenerate.
- Five analyzed pilot vectors now remain across `GSM8K`, `CommonsenseQA`,
  `MATH`, `HellaSwag`, and `ARC-Challenge`; the imported run records live
  under `runs/`.
- Initial reusable CLI entry points now exist in `training/`, `extraction/`,
  and `analysis/` for the Phase 0 path.
- The imported pilot analysis indicates a stronger adaptation signal in
  `mlp.gate_proj` and `mlp.up_proj` than in attention `k_proj` or `v_proj`,
  but this remains diagnostic rather than hypothesis evidence.
- `statistics/` remains intentionally empty until multi-task Phase 1 outputs
  exist.

## Setup

Install the current Python dependencies from `requirements.txt` before running
the scripts outside Kaggle.

Reproducible local environment setup:

- `SETUP.md`
- `scripts/bootstrap_venv.sh`
- `scripts/verify_env.py`
- `requirements-notes.md`

Execution metadata:

- `config/registry.json`
- `config/task_registry.json`
- `runs/phase0_run_note_template.md`

## Current entry points

- `training/phase0_gsm8k_grpo.py`
- `extraction/extract_registered_update_vector.py`
- `analysis/pilot_svd_diagnostic.py`
- `analysis/h1a_h1b_task_matrix.py`

## Execution checklist

- `EXPERIMENT_CHECKLIST.md`

## Experiment plan

1. Phase 0: validate one complete `train → extract → SVD` pipeline on
   `Qwen2.5-1.5B-Instruct` and `GSM8K`.
2. Phase 1: collect registered update vectors for at least 10 tasks, then run
   H1a and H1b on the normalized task matrix.
3. Phase 2: run H2 transfer experiments across the registered model pairs.
4. Exploratory analyses: run H1c, H3, and related follow-up analyses only after
   the registered primary path is complete.

## Registered models (`v1.3`)

- Primary: `Qwen2.5-1.5B-Instruct`
- Secondary A: `Qwen2.5-1.5B`
- Secondary B: `Qwen2.5-3B`
- Secondary C: `google/gemma-2-2b-it`

## Registered tasks (`v1.3`, minimum set for Phase 1)

GSM8K, MATH, AIME, AMC, MATH500, HumanEval, MBPP, CommonsenseQA,
HellaSwag, ARC-Challenge

## Registration notes

- The registered primary update object is `concat(ΔW_A, ΔW_B)` across all LoRA
  adapter layers.
- Any dense effective delta such as `scaling * (B @ A)` should be labeled
  exploratory or pilot-only unless the registration is explicitly amended.
- H1a and H1b are registered only for normalized multi-task update matrices.
  A single-task pilot SVD is a pipeline diagnostic, not hypothesis evidence.
