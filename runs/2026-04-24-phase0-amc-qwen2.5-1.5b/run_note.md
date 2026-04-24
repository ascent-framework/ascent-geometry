# Phase 0 Run Note

## Summary

- Date: 2026-04-24 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/add-amc-task-path`
- Commit: `d3ac9ed`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `AMC`
- Method: `GRPO`

## Environment

- Python version: Kaggle Python 3 notebook runtime
- Torch version: `2.10.0+cu128`
- GPU model: `Tesla T4`
- VRAM: `15.6 GB`
- Precision: `bf16`
- Environment matched `requirements.txt` exactly: not verified in imported report

## Outputs

- Training report: `training_report.json`
- Extraction report: `extraction_report.json`
- Analysis report: `analysis_report.json`
- Analysis summary: `analysis_summary.md`
- Run manifest: `run_manifest.json`
- Raw run report: `run_report.json`
- Adapter artifact location: `/kaggle/working/amc-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/amc-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- The rerun completed end-to-end on `T4` after fixing numeric-answer reward handling.
- Training, extraction, and pilot SVD completed without runtime errors.
- The resulting vector is non-degenerate and matches the expected dimensionality.

## What Failed

- Earlier attempts failed due to session GPU mismatch and reward helper type handling.

## Failure History

- Attempt 1: session ran on `P100` and failed the hard GPU guard (`need T4`).
- Attempt 2: `AttributeError: 'int' object has no attribute 'replace'` in reward parsing for numeric AMC answers.
- Attempt 3: after converting reward parsing input to string, the rerun completed successfully.

## What To Fix Next

- Keep T4 pinned before launch.
- Preserve numeric-safe reward parsing for future numeric-answer tasks.

## Notes

- See `kaggle_execution.md` in this directory for the exact run command.
- This run is `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- Source kernel: `chson0316/ascent-g-phase-0-pilot-amc-qwen2-5-1-5b`
- Training runtime: `3285.5s` total, `65.71s` per step over `50` steps.
- Update vector SHA-256: `ba5b982fe3f54cb5b939657e557c75062608935570b21957036411286a3c1844`
