# Phase 0 Run Note

## Summary

- Date: 2026-04-23 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/add-arc-challenge-task-path`
- Commit: `6645f7a`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `ARC-Challenge`
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
- Adapter artifact location: `/kaggle/working/arc-challenge-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/arc-challenge-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- The T4 rerun completed the full `train -> extract -> SVD` pilot pipeline for `ARC-Challenge`.
- Training, registered update-vector extraction, and pilot SVD all completed on the same adapter.
- The extracted vector was non-zero, covered all registered target modules, and matched the expected dimensionality.

## What Failed

- The first attempt required switching the Kaggle session back to `T4`.

## What To Fix Next

- Keep the Kaggle accelerator pinned to `T4` before launching future pilots.
- Start the next registered task collection run after this ARC-Challenge pilot.

## Notes

- See `kaggle_execution.md` in this directory for the exact run command.
- This run is planned as `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- Source kernel: `chson0316/ascent-g-phase-0-pilot-arc-challenge-qwen2-5-1-5b`
- Training runtime: `325.2s` total, `6.50s` per step over `50` steps
- Update vector SHA-256: `cf94440107893191e1fe5cf65cfa63987f1676e0902f55303319df4f4f4a048f`

