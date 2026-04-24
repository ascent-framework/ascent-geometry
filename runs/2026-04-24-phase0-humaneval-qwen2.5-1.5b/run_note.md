# Phase 0 Run Note

## Summary

- Date: 2026-04-24 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/implement-humaneval-task-path`
- Commit: `33f44cf`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `HumanEval`
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
- Adapter artifact location: `/kaggle/working/humaneval-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/humaneval-qwen2.5-1.5b-phase0/update_vector.npy`

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

- No critical runtime errors in the final run.

## Failure History

- No execution failure in the final T4 session.

## What To Fix Next

- Re-run MBPP with the same indentation-preserving code normalization patch.
- Keep `T4` pinned before launch to avoid guard-trigger reruns.

## Notes

- Source kernel: `chson0316/ascent-g-phase-0-pilot-humaneval-qwen2-5-1-5b`
- Training runtime: `3191.6s` total, `63.83s` per step over `50` steps.
- Update vector shape/norm: `[9232384]`, `22.8905`
- Update vector SHA-256: `be63373d4f174d390ea130c37bb537c77754a02ffda01ef034502b77be9fe46b`
- This run is `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
