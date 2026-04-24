# MBPP Kaggle Execution

This note records the successful `pilot_only` Kaggle rerun for `MBPP` on
`Qwen/Qwen2.5-1.5B-Instruct` after the reward-path fixes landed.

## Goal

Run one end-to-end `train -> extract -> SVD` pilot for `MBPP` using the
task-aware CLI path, then import the outputs into this dated run directory.

## Actual Runtime

- Kaggle notebook session
- GPU: `Tesla T4`
- Internet enabled
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `MBPP`
- Scope label: `pilot_only`

## Fixes Included

- Code normalization now preserves Python indentation.
- The MBPP prompt includes the first unit test assertion.
- Reward execution uses `subprocess.run(..., timeout=3.0)` instead of notebook-local multiprocessing.
- Live reward progress is printed as `[MBPP-REWARD] ...` for Kaggle log inspection.
- Extraction now records `b_norm_total` and `effective_delta_non_zero`.

## Observed Runtime Signals

- Final logged reward summary: `seen=400 passed=158 pass_rate=0.395 timeouts=0`
- Total training time: `2896.5s`
- Per-step time: `57.93s`
- Update vector norm: `22.8866`
- `b_norm_total`: `1.1453`
- `effective_delta_non_zero`: `true`

## Imported Outputs

Under `runs/2026-04-24-phase0-mbpp-qwen2.5-1.5b/`:

- `training_report.json`
- `extraction_report.json`
- `analysis_report.json`
- `analysis_summary.md`
- `run_manifest.json`
- `run_note.md`
- `run_report.json`
- `update_vector_provenance.json`
- `kaggle.log`

## Analysis Outcome

This rerun supersedes the failed `2026-04-23` MBPP analysis exclusion for
flat reward and degenerate SVD. The updated MBPP pilot is eligible for the
Phase 0 analyzed-vector set.
