# Phase 0 Run Note

## Summary

- Date: 2026-04-23 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/add-math-task-path`
- Commit: `563efd9`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `MATH`
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
- Adapter artifact location: `/kaggle/working/2026-04-23-phase0-math-qwen2.5-1.5b/adapter`
- Update vector artifact location: `/kaggle/working/2026-04-23-phase0-math-qwen2.5-1.5b/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- Kaggle `T4` environment completed the full `train -> extract -> SVD` pilot pipeline for `MATH`.
- Training, registered update-vector extraction, and pilot SVD all completed on the same adapter.
- The extracted vector was non-zero, covered all registered target modules, and matched the expected dimensionality.

## What Failed

- The first attempt failed on `P100`; rerunning on `T4` resolved the hardware issue.
- The pre-fix launcher archived `training` and `extraction` outputs directly under `artifacts_root`, but did not preserve `analysis_report.json` or `run_manifest.json` there.
- The missing `analysis` and `manifest` JSONs were recovered from the Kaggle execution log for this run.

## What To Fix Next

- Persist durable artifact URIs after the run completes.
- Keep launcher outputs mirrored into `artifacts_root` so `analysis_report.json` and `run_manifest.json` are always downloadable.
- Start the next registered task collection run now that a third valid task vector exists.

## Notes

- See `kaggle_execution.md` in this directory for the exact run command.
- This run is planned as `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- Source kernel: `chson0316/ascent-g-phase-0-pilot-math-qwen2-5-1-5b`
- Training runtime: `2296.7s` total, `45.93s` per step over `50` steps
- Update vector SHA-256: `dbbbae000afc6e2f037dd71c8a23c8e40bf8e9846092fd71ee2f1e8facf47fb5`
