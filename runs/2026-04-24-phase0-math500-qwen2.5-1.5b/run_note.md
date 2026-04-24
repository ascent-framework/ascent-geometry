# Phase 0 Run Note

## Summary

- Date: 2026-04-24 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/add-math500-task-path`
- Commit: `18f7f29`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `MATH500`
- Method: `GRPO`

## Environment

- Python version: Kaggle Python 3 notebook runtime
- Torch version: `2.10.0+cu128`
- GPU model: `Tesla T4`
- VRAM: `15.6 GB`
- Precision: `bf16`
- Runtime matched the expected Kaggle GPU path: yes
- Environment matched `requirements.txt` exactly: not verified in imported report

## Outputs

- Training report: `training_report.json`
- Extraction report: `extraction_report.json`
- Analysis report: `analysis_report.json`
- Analysis summary: `analysis_summary.md`
- Run manifest: `run_manifest.json`
- Raw run report: `run_report.json`
- Kaggle execution log: `kaggle.log`
- Adapter artifact location: `/kaggle/working/math500-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/math500-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- T4 rerun completed end-to-end without runtime errors.
- Training, registered extraction, and pilot SVD all completed in one session.
- The update vector is non-degenerate with expected dimension and full target coverage.

## What Failed

- Earlier run observation mismatch occurred while the kernel was still running in UI.

## Failure History

- No execution failure in the final T4 session.

## What To Fix Next

- Keep `T4` pinned before launching to avoid guard-trigger reruns.
- Continue keeping all run-local reports mirrored and committed with run note updates.

## Notes

- Source kernel: `chson0316/ascent-g-phase-0-pilot-math500-qwen2-5-1-5b`
- Training runtime: `3280.1s` total, `65.6s` per step over `50` steps.
- Update vector shape/norm: `[9232384]`, `22.8853`
- Update vector SHA-256: `85f7ad82ab0089e93d6445a708612e16d93d87199b947c7b4d0bb30a473b7d89`
- This run is `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
