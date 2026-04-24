# MBPP Phase 0 Analysis Summary

> Historical failure summary only. The successful rerun that restores MBPP to
> the analyzed pilot set lives under
> `runs/2026-04-24-phase0-mbpp-qwen2.5-1.5b/`.

The latest MBPP pilot completed end-to-end on `Tesla T4`, but it is excluded
from the geometry analysis set because the diagnostic signal is flat.

Key observed outputs from `run_report.json`:

- update vector shape: `9,232,384`
- update vector norm: `22.8695`
- SHA-256: `869f1efe01b46a2cb5656109e6fbc042dd21284d3c68f8c0489f4945612b4056`
- layers captured: `196`
- accepted criteria: all `true`

The analysis side is the anomaly:

- `svd_results` contains 196 layer entries, but `r90 = 1` for every layer
- `s_max = 0.0` for every layer
- module-wise averages are all zero
- the training logs still show `reward = 0.0`, `loss = 0.0`, and `grad_norm = 0.0`

Interpretation:

- The pipeline itself is valid: training, adapter reload, extraction, and report generation all completed.
- The MBPP SVD diagnostic did not surface a meaningful spectral pattern in this run.
- The reward signal remained flat, so the update object is structurally valid
  but not informative for geometry analysis.
- This should be treated as a diagnostic limitation, not as evidence of a null adaptation signal.

Operationally, MBPP is still a successful pilot vector capture.
The next step is to either relax the MBPP reward for future experiments or
keep MBPP as an exclusion note, then continue collecting additional task
vectors and compare them with the other analyzed pilot runs.
