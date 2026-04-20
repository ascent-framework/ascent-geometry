# runs

Run logs and metadata. No model checkpoints here.

## What belongs here

- Short run reports (what worked, what failed, what to fix)
- Run configuration snapshots (hyperparams, environment info)
- Small derived outputs (r_90 values, summary CSVs)

## What does not belong here

- Adapter checkpoints → Kaggle Dataset or Google Drive
- Raw update vectors (large) → Kaggle Dataset or Google Drive

## Naming convention

`{date}-{phase}-{task}-{model}/`

Example: `2026-04-20-phase0-gsm8k-qwen2.5-1.5b/`
