# runs

Run logs and metadata. No model checkpoints here.

Canonical stage-report schema: `runs/report.schema.json`

## What belongs here

- Short run reports (what worked, what failed, what to fix)
- Run configuration snapshots (hyperparams, environment info)
- Small derived outputs (r_90 values, summary CSVs)
- Registered H1a/H1b analysis reports after multi-task vectors exist
- Required Phase 0 runtime metadata: GPU model, precision mode, per-step timing
- `run_phase0_pipeline.py` for creating a dated Phase 0 run directory and
  aggregating stage reports

## What does not belong here

- Adapter checkpoints → Kaggle Dataset or Google Drive
- Raw update vectors (large) → Kaggle Dataset or Google Drive

## Naming convention

`{date}-{phase}-{task}-{model}/`

Example: `2026-04-20-phase0-gsm8k-qwen2.5-1.5b/`

Imported Phase 0 pilot example:

- `runs/2026-04-22-phase0-gsm8k-qwen2.5-1.5b/`

## Current entry point

- `run_phase0_pipeline.py`

## Report shape

Every stage report should use the same top-level keys:

- `schema_version`
- `generated_at`
- `stage`
- `phase`
- `task`
- `model`
- `method`
- `scope`
- `summary`
- `config`
- `metrics`
- `artifacts`
- `runtime`
- `validation`
- `notes`
