# ascent-geometry

ASCENT-G: Adaptation Geometry experiments (Paper 1)

Theory and pre-registration: [ascent-framework](https://github.com/ascent-framework/ascent-framework)

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

## Registered models

- Primary: `Qwen2.5-1.5B-Instruct`
- Secondary: `Gemma-2-2B-IT`

## Registered tasks (Phase 1)

GSM8K, MATH-500, ARC-Challenge, HellaSwag, WinoGrande
