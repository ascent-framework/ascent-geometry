# MBPP Phase 0 Analysis Summary

The latest MBPP pilot rerun completed end-to-end on `Tesla T4`, but it remains excluded from the geometry analysis set because the diagnostic signal is flat.

Key observed outputs from `run_report.json`:

- update vector shape: `9,232,384`
- update vector norm: `22.8688`
- SHA-256: `c7fab727def0044a72b5fafea2acc0858503ed78ed0a90146fad91f8837796dd`
- layers captured: `196`
- accepted criteria: all `true`

The analysis side remains degenerate:

- `svd_results` contains `196` layer entries, with `s_max = 0.0` across layers
- module-wise `s_max` averages are all `0.0`
- training logs still show `reward = 0.0`, `loss = 0.0`, and `grad_norm = 0.0` at all logged steps

Interpretation:

- The pipeline itself is valid: training, adapter reload, extraction, and report generation all completed.
- MBPP still does not yield an informative spectral geometry signal under the current reward/setup.
- MBPP should remain logged for operational traceability but excluded from geometry-analysis aggregation.
