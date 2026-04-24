# MBPP Phase 0 Analysis Summary

The `2026-04-24` MBPP rerun completed end-to-end on `Tesla T4` and is now
included in the geometry analysis set because the reward signal and SVD
diagnostic are both non-degenerate.

Key observed outputs from `run_report.json`:

- update vector shape: `9,232,384`
- update vector norm: `22.8866`
- SHA-256: `2413b99e418edc4f065e81a2d91df5bbdd53f1c4125316bf9b58c4c50c789fbc`
- layers captured: `196`
- `b_norm_total`: `1.1453`
- accepted criteria: all `true`

The analysis side is now healthy:

- `svd_results` contains 196 layer entries with `s_max > 0` for every layer
- mean `r90 = 6.57`, with range `4` to `7`
- mean `s_max = 0.0430`, max `s_max = 0.1244`
- module-wise mean `s_max` is strongest in `up_proj` and `gate_proj`
- Kaggle reward logs remained live throughout the run with final
  `pass_rate = 0.395` and `timeouts = 0`

Interpretation:

- The earlier MBPP failure mode was a reward-path bug, not a stable property of the task.
- Preserving indentation, injecting the first test assertion, and moving test execution to a timed subprocess restored a usable training signal.
- The MBPP update object is now informative enough for pilot geometry diagnostics.
- This remains pilot-only evidence and should not be interpreted as registered H1a/H1b/H2 evidence by itself.
