# Phase 0 Run Note

## Summary

- Date: 2026-04-24 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/add-aime-task-path`
- Commit: `a3925d8`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `AIME`
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
- Adapter artifact location: `/kaggle/working/aime-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/aime-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- The rerun completed end-to-end on `T4` with the corrected AIME field mapping (`problem`/`answer`).
- Training, extraction, and pilot SVD all completed without runtime errors.
- The resulting vector is non-degenerate and matches the expected dimensionality.

## What Failed

- Earlier attempts failed before this successful run due to notebook/data mismatches and GPU session drift.

## Failure History

- Attempt 1: `P100` session triggered the hard GPU guard (`need T4`).
- Attempt 2: `KeyError: 'answer'` after mapping expected a missing field.
- Attempt 3: `SyntaxError: unterminated string literal` in the final print cell.
- Attempt 4: `KeyError: 'extracted_solution'` after switching to AIME data rows containing `answer`.
- Attempt 5: AIME notebook had a stale dataset override path during debugging; removed so only `math-ai/aime25` is loaded.

## What To Fix Next

- Keep AIME mapping pinned to `answer` unless dataset schema changes are revalidated.
- Continue with next task collection run and monitor reward sparsity.

## Notes

- See `kaggle_execution.md` in this directory for the exact run command.
- This run is `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- Source kernel: `chson0316/ascent-g-phase-0-pilot-aime-qwen2-5-1-5b`
- Training runtime: `3605.5s` total, `72.11s` per step over `50` steps.
- Update vector SHA-256: `fed5ffd2789a049157a0c8a60bb4be21edcfc7fba993f05cb3aa6c51e3598cb8`
