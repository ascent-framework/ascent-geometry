# Phase 0 Run Note

> Superseded for analysis inclusion by the successful rerun in
> `runs/2026-04-24-phase0-mbpp-qwen2.5-1.5b/`. This `2026-04-23` record is
> retained as the failure history for the earlier flat-reward / degenerate-SVD
> MBPP pilot.

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

- GPU model: `Tesla T4`
- Precision: `bf16`
- Runtime matched the expected Kaggle GPU path.

## Outputs

- Training report: `training_report.json`
- Extraction report: `extraction_report.json`
- Analysis report: `analysis_report.json`
- Run manifest: `run_manifest.json`
- Umbrella report: `run_report.json`
- Analysis summary: `analysis_summary.md`
- Adapter artifact location: `/kaggle/working/mbpp-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/mbpp-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- End-to-end Kaggle execution completed on `Tesla T4`.
- The task-aware MBPP path loaded the dataset and trained for 50 steps.
- The adapter saved and reloaded successfully.
- The registered update vector was extracted and had non-zero norm.
- The final T4 run completed without session failure.

## What Failed

- The training signal was still degenerate: `reward = 0.0`, `loss = 0.0`, and
  `grad_norm = 0.0` throughout the run.
- The SVD summary in `run_report.json` was degenerate (`s_max = 0.0` across layers), so the analysis is not very informative.

## Failure History

- Initial notebook version used `example["prompt"]` for MBPP, but the dataset
  exposes `text` and `code`, which caused a `KeyError: 'prompt'`.
- The first MBPP reward design used exact string equality, which left
  `reward = 0.0`, `loss = 0.0`, and `grad_norm = 0.0` across training.
- A later rerun failed immediately because the Kaggle session was still on
  `Tesla P100-PCIE-16GB` while the notebook required `T4`.
- The final T4 rerun completed, but the reward signal still never turned
  positive, so the effective LoRA delta remained spectrally flat in the SVD
  diagnostic.

## What To Fix Next

- Keep the task-aware MBPP path in the shared pipeline.
- Relax or redesign the MBPP reward if the goal is to elicit non-zero updates.
- Treat the MBPP SVD result as a diagnostic edge case and exclude MBPP from
  the geometry analysis set.
- Move on to the next registered task vector.

## Analysis Status

- Excluded from geometry analysis due to flat reward, zero gradient norms, and
  degenerate SVD output in this specific run only.

## Notes

- This run is `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- The MBPP execution required aligning the notebook and registry with the dataset's `text` and `code` fields.
