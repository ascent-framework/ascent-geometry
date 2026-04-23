# Phase 0 Pilot Analysis Summary

This note summarizes the completed `pilot_only` Kaggle run for
`CommonsenseQA` on `Qwen/Qwen2.5-1.5B-Instruct`.

## Scope

- This is a `pilot_only` single-task diagnostic.
- It validates the Phase 0 pipeline and produces one task vector.
- It is not registered evidence for H1a, H1b, or H2.

## Pipeline Outcome

- Full `train -> extract -> SVD` pilot completed successfully on `Tesla T4`
- Training reached `50` steps in `233.9s`
- Per-step time was `4.68s`
- Adapter save and downstream reuse succeeded
- Registered update vector extraction succeeded
- Pilot SVD diagnostic ran across `196` LoRA layers

## Vector Summary

- Registered object: `concat(Delta W_A, Delta W_B)`
- Vector shape: `9,232,384`
- Vector norm: `22.9150`
- SHA-256: `172ebc3af7f0af4881ff9089f01946627649c3d1b4eb64af180257b18e9ff708`
- Registered targets covered: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`

## SVD Pattern Summary

- Layers analyzed: `196`
- `r90` range: `4` to `7`
- Mean `r90`: `5.93`
- Mean `s_max`: `0.0475`

Per-module mean `s_max`:

- `up_proj`: `0.0952`
- `gate_proj`: `0.0899`
- `q_proj`: `0.0400`
- `o_proj`: `0.0392`
- `down_proj`: `0.0361`
- `v_proj`: `0.0167`
- `k_proj`: `0.0154`

Per-module mean `r90`:

- `gate_proj`: `6.29`
- `down_proj`: `6.18`
- `up_proj`: `5.96`
- `o_proj`: `5.89`
- `k_proj`: `5.86`
- `q_proj`: `5.79`
- `v_proj`: `5.57`

Top `s_max` layers:

- `layers.1.mlp.gate_proj`: `0.1226`
- `layers.7.mlp.up_proj`: `0.1150`
- `layers.2.mlp.up_proj`: `0.1139`
- `layers.17.mlp.up_proj`: `0.1134`
- `layers.3.mlp.up_proj`: `0.1119`

Depth summary:

- Early-layer mean `s_max`: `0.0490`
- Mid-layer mean `s_max`: `0.0480`
- Late-layer mean `s_max`: `0.0453`
- Early-layer mean `r90`: `6.11`
- Mid-layer mean `r90`: `6.00`
- Late-layer mean `r90`: `5.67`

## Interpretation

- As in the earlier `GSM8K` pilot, the adaptation signal is concentrated much
  more strongly in `MLP gate_proj` and `up_proj` than in attention `k_proj`
  or `v_proj`.
- `CommonsenseQA` shows slightly lower average `r90` than the `GSM8K` pilot,
  suggesting an even more compressible layerwise update structure in this short
  run.
- The strongest layers are again MLP-heavy, with no compelling evidence that
  attention-key/value updates dominate the single-task adaptation geometry.
- Depth effects exist but are weaker than module-type effects; the main signal
  is still "which projection family changed" rather than "which block depth
  changed."

## Limits

- This is a 50-step pilot rather than a registered Phase 1 run.
- Single-task SVD diagnostics cannot support shared-subspace claims.
- The raw Kaggle extraction and analysis reports had hardcoded `GSM8K` task
  metadata; the imported local copies were corrected for archive purposes.

## Next Step

Proceed to the next registered task collection run and keep checking:

- vector dimensionality matches `9,232,384`
- registered targets remain fully covered
- `MLP gate/up` dominance persists or breaks across tasks
