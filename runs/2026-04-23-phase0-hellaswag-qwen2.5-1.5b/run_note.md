# Phase 0 Run Note

## Summary

- Date: 2026-04-23 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/add-hellaswag-task-path`
- Commit: `4acd368`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `HellaSwag`
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
- Adapter artifact location: `/kaggle/working/hellaswag-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/hellaswag-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- The T4 rerun completed the full `train -> extract -> SVD` pilot pipeline for `HellaSwag`.
- Training, registered update-vector extraction, and pilot SVD all completed on the same adapter.
- The extracted vector was non-zero, covered all registered target modules, and matched the expected dimensionality.

## What Failed

- The first attempt failed because the Kaggle session was still on `Tesla P100-PCIE-16GB` instead of `T4`.
- That was a session hardware mismatch, not a code failure.

## What To Fix Next

- Keep the Kaggle accelerator pinned to `T4` before launching future pilots.
- Start the next registered task collection run after this HellaSwag pilot.

## Notes

- See `kaggle_execution.md` in this directory for the exact run command.
- This run is planned as `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- Source kernel: `chson0316/ascent-g-phase-0-pilot-hellaswag-qwen2-5-1-5b`
- Training runtime: `674.4s` total, `13.49s` per step over `50` steps
- Update vector SHA-256: `411b5696d868c764f131b34613a65f20b9f1e2331074c933147918ddf3ec5a70`

