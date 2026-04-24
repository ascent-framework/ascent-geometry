# Phase 0 Run Note

## Summary

- Date: 2026-04-24 KST
- Operator: son
- Environment: Kaggle notebook
- Branch: `codex/retry-mbpp-pilot`
- Commit: `fa1ced0`

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

- End-to-end Kaggle execution completed on `Tesla T4` with the repaired MBPP notebook path.
- The task-aware MBPP path loaded dataset fields correctly and ran all pipeline stages.
- Adapter save/reload and registered vector extraction were successful.

## What Failed

- The training signal remained degenerate: `reward = 0.0`, `loss = 0.0`, and `grad_norm = 0.0` at every logged step.
- `svd_results` remained flat (`s_max = 0.0` layerwise), so geometry analysis value is still absent.

## Failure History

- Initial notebook version used `example["prompt"]`, but the MBPP dataset exposes `text` and `code` (`KeyError: 'prompt'`).
- Early reward design (`code_exact_match`) produced fully flat learning signals.
- Multiple runs failed on `P100` due to the hard GPU guard requiring `T4`.
- A notebook structure issue could drop MBPP test fields in formatting; this was fixed before the latest rerun.
- Latest rerun (`2026-04-24`) confirms the path is operational, but still spectrally degenerate.

## What To Fix Next

- Keep MBPP in the run ledger but excluded from geometry analysis.
- If MBPP must be analyzable, redesign reward shaping or sampling to avoid all-zero rewards.
- Continue collection on other tasks for analysis-grade vectors.

## Analysis Status

- Excluded from geometry analysis due to flat reward/gradient and degenerate SVD output.

## Notes

- This run is `pilot_only` and should not be interpreted as registered H1a/H1b/H2 evidence.
- Source kernel: `chson0316/ascent-g-phase-0-pilot-mbpp-qwen2-5-1-5b`
- Training runtime: `3002.0s` total, `60.04s` per step over `50` steps.
- Update vector SHA-256: `c7fab727def0044a72b5fafea2acc0858503ed78ed0a90146fad91f8837796dd`
