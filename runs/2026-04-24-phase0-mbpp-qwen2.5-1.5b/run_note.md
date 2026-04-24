# Phase 0 Run Note

## Summary

- Date: 2026-04-24 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/mbpp-rerun-indent-fix`

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
- Kaggle execution log: `kaggle.log`
- Adapter artifact location: `/kaggle/working/mbpp-qwen2.5-1.5b-phase0/adapter`
- Update vector artifact location: `/kaggle/working/mbpp-qwen2.5-1.5b-phase0/update_vector.npy`

## Acceptance Criteria

- [x] Model loaded successfully
- [x] Training reached stopping point
- [x] Adapter saved and reloaded
- [x] Registered update vector extracted
- [x] Update vector non-degenerate
- [x] Effective delta confirmed non-zero
- [x] Pilot SVD diagnostic ran successfully

## What Worked

- End-to-end Kaggle execution completed on `Tesla T4`.
- The task-aware MBPP path trained for 50 steps with live reward progress logs.
- The reward path remained active throughout the run (`pass_rate ~= 0.395`, `timeouts = 0`).
- The adapter saved and reloaded successfully.
- The registered update vector was extracted, had non-zero norm, and passed the `effective_delta_non_zero` gate.
- The SVD summary was non-degenerate across all 196 captured layers.

## Failure History

- The original notebook used `example["prompt"]` even though MBPP exposes `text` and `code`, causing a `KeyError`.
- The first reward path flattened Python indentation and compared raw strings, which left `reward = 0.0`, `loss = 0.0`, and `grad_norm = 0.0` throughout training.
- A later rerun used notebook-local multiprocessing and failed on Kaggle because `spawn` could not resolve the notebook-defined worker entry point.
- Another rerun failed with `NameError: json is not defined` in the subprocess payload builder.
- The final rerun replaced notebook-local multiprocessing with `subprocess.run(..., timeout=3.0)`, restored imports, and completed successfully.

## What Changed

- Code normalization now preserves indentation for generated Python functions.
- The MBPP prompt includes the first test assertion so reward is aligned with the visible task contract.
- Reward execution now runs in a timed subprocess and logs `[MBPP-REWARD]` progress to the Kaggle output.
- Extraction now records `b_norm_total` and `effective_delta_non_zero` so false-pass non-degeneracy from LoRA initialization alone is avoided.

## Analysis Status

- Included in geometry analysis for the pilot vector set.

## Notes

- This run is `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- This run supersedes the exclusion-only MBPP note preserved under `runs/2026-04-23-phase0-mbpp-qwen2.5-1.5b/`.
