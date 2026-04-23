# Phase 0 Run Note

## Summary

- Date: 2026-04-22 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/start-phase1-commonsenseqa`
- Commit: `59c414c`

## Target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `CommonsenseQA`
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
- Adapter artifact location: `/kaggle/working/2026-04-22-phase0-commonsenseqa-qwen2.5-1.5b/adapter`
- Update vector artifact location: `/kaggle/working/2026-04-22-phase0-commonsenseqa-qwen2.5-1.5b/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- Kaggle `T4` environment completed the full `train -> extract -> SVD` pilot pipeline for `CommonsenseQA`.
- Training, registered update-vector extraction, and pilot SVD all produced outputs for the same adapter.
- The extracted vector was non-zero, covered all registered target modules, and matched the expected dimensionality.

## What Failed

- Earlier reruns hit two code-path bugs before the successful `T4` completion:
- `train_grpo_task.py` used outdated TRL argument names.
- `run_phase0_pipeline.py` uppercased task names and broke `CommonsenseQA` registry lookup.
- The raw Kaggle extraction and analysis reports also hardcoded `GSM8K` in their task metadata; the imported local copies were corrected.

## What To Fix Next

- Persist durable artifact URIs after the run completes instead of relying on `/kaggle/working`.
- Keep task metadata parameterized across all stage scripts before Phase 1 collection expands further.
- Start the next registered task collection run now that a second valid task vector exists.

## Notes

- See `kaggle_execution.md` in this directory for the exact run command.
- This run is planned as `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- Source kernel: `chson0316/ascent-g-phase-0-pilot-commonsenseqa-qwen2-5-1-5b`
- Training runtime: `233.9s` total, `4.68s` per step over `50` steps
- Update vector SHA-256: `172ebc3af7f0af4881ff9089f01946627649c3d1b4eb64af180257b18e9ff708`
