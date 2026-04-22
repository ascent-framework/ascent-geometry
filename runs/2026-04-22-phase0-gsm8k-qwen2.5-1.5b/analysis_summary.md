# Phase 0 Pilot Analysis Summary

This note summarizes the `svd_results` imported from the Kaggle Phase 0 pilot
run for `Qwen/Qwen2.5-1.5B-Instruct` on `GSM8K`.

## Scope

- This is a `pilot_only` analysis.
- It validates end-to-end pipeline behavior only.
- It is not registered evidence for H1a, H1b, or H2.

## Data Source

- Source kernel: `chson0316/ascent-g-phase-0-pilot-gsm8k-qwen2-5-1-5b`
- Source report: `/tmp/kaggle-kernel-output/gsm8k-qwen2.5-1.5b-phase0/run_report.json`
- Source timestamp: `2026-04-21T15:47:24.149191Z`

## Pipeline Outcome

- Full `train -> extract -> SVD` pilot completed successfully on `Tesla T4`
- Training reached `50` steps in `2907.9s`
- Adapter save and reload succeeded
- Registered update vector extraction succeeded
- All registered target modules were covered
- The update vector was non-degenerate

## Vector Summary

- Registered object: `concat(Delta W_A, Delta W_B)`
- Vector shape: `9,232,384`
- Vector norm: `22.8922`
- Layers captured: `196`
- SHA-256: `8bb6b927c8b0c39f46016f9d6aaa5eead753faf8a0805e5beff6252970a78420`

## SVD Pattern Summary

- `r90` range: `5` to `7`
- Mean `r90`: `6.57`
- Mean `s_max`: `0.0430`

Per-module mean `s_max`:

- `gate_proj`: `0.0863`
- `up_proj`: `0.0853`
- `q_proj`: `0.0343`
- `o_proj`: `0.0341`
- `down_proj`: `0.0314`
- `v_proj`: `0.0152`
- `k_proj`: `0.0144`

Per-module mean `r90`:

- `down_proj`: `6.89`
- `gate_proj`: `6.79`
- `o_proj`: `6.71`
- `up_proj`: `6.64`
- `q_proj`: `6.61`
- `k_proj`: `6.36`
- `v_proj`: `6.00`

## Interpretation

- The pilot update energy was concentrated much more strongly in `MLP`
  `gate_proj` and `up_proj` modules than in attention `k_proj` or `v_proj`
  modules.
- `q_proj` and `o_proj` showed moderate responses, clearly below `gate_proj`
  and `up_proj` but above `k_proj` and `v_proj`.
- The strongest individual responses were all in `mlp.gate_proj` or
  `mlp.up_proj`, with the top value at
  `base_model.model.model.layers.3.mlp.gate_proj` (`s_max = 0.1079`).
- Mean `s_max` was nearly flat across depth buckets, so the current pilot does
  not show a strong early-vs-late layer trend.
- Taken together, this pilot suggests that the single-task adaptation signal is
  more dominated by MLP feature transformation than by attention-key/value
  reconfiguration.

## Limits

- Single-task pilot results cannot support shared-subspace claims.
- No H1a/H1b decision should be inferred from this note.
- Artifact paths still point to downloaded `/tmp` locations and should be
  replaced with durable storage references.

## Next Step

Collect the same registered update object for the remaining registered tasks
and test whether the MLP-heavy pattern persists in the normalized multi-task
matrix.
