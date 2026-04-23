# Phase 0 Run Note

## Summary

- Date: 2026-04-23 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/add-mbpp-task-path`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `MBPP`
- Method: `GRPO`

## Environment

- Python version: Kaggle Python 3 notebook runtime
- GPU model: `Tesla T4`
- Precision: expected `bf16`
- Environment matched `requirements.txt` exactly: not yet verified

## Outputs

- Training report: `training_report.json`
- Extraction report: `extraction_report.json`
- Analysis report: `analysis_report.json`
- Analysis summary: `analysis_summary.md`
- Run manifest: `run_manifest.json`
- Adapter artifact location: `/kaggle/working/2026-04-23-phase0-mbpp-qwen2.5-1.5b/adapter`
- Update vector artifact location: `/kaggle/working/2026-04-23-phase0-mbpp-qwen2.5-1.5b/update_vector.npy`

## Acceptance Criteria

- [ ] Model loaded successfully
- [ ] Training reached stopping point
- [ ] Adapter saved and reloaded
- [ ] Registered update vector extracted
- [ ] Update vector non-degenerate
- [ ] Pilot SVD diagnostic ran successfully

## What Worked

- Pending Kaggle execution.

## What Failed

- Pending Kaggle execution.

## What To Fix Next

- Import the completed Kaggle outputs if the run finishes successfully.
- Keep launcher outputs mirrored into `artifacts_root`.
- Start the next registered task collection run after MBPP.

## Notes

- See `kaggle_execution.md` in this directory for the exact run command.
- This run is planned as `pilot_only` and should not be interpreted as
  registered H1a/H1b/H2 evidence.
