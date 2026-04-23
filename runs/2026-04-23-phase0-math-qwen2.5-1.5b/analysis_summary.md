# Phase 0 Pilot Analysis Summary

This note summarizes the completed `pilot_only` Kaggle run for `MATH` on
`Qwen/Qwen2.5-1.5B-Instruct`.

## Scope

- This is a `pilot_only` single-task diagnostic.
- It validates the Phase 0 pipeline and adds one more task vector.
- It is not registered evidence for H1a, H1b, or H2.

## Pipeline Outcome

- Full `train -> extract -> SVD` pilot completed successfully on `Tesla T4`
- Training reached `50` steps in `2296.7s`
- Per-step time was `45.93s`
- Adapter save and downstream reuse succeeded
- Registered update vector extraction succeeded
- Pilot SVD diagnostic ran across `196` LoRA layers

## Vector Summary

- Registered object: `concat(Delta W_A, Delta W_B)`
- Vector shape: `9,232,384`
- Vector norm: `22.8628`
- SHA-256: `dbbbae000afc6e2f037dd71c8a23c8e40bf8e9846092fd71ee2f1e8facf47fb5`
- Registered targets covered: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`

## SVD Pattern Summary

- Layers analyzed: `196`
- `r90` range: `5` to `7`
- Mean `r90`: `6.70`
- Mean `s_max`: `0.0177`

Per-module mean `s_max`:

- `up_proj`: `0.0353`
- `gate_proj`: `0.0342`
- `o_proj`: `0.0149`
- `q_proj`: `0.0144`
- `down_proj`: `0.0132`
- `v_proj`: `0.0063`
- `k_proj`: `0.0059`

Per-module mean `r90`:

- `down_proj`: `6.86`
- `gate_proj`: `6.82`
- `up_proj`: `6.82`
- `q_proj`: `6.79`
- `o_proj`: `6.75`
- `k_proj`: `6.64`
- `v_proj`: `6.25`

Top `s_max` layers:

- `layers.20.mlp.gate_proj`: `0.0457`
- `layers.3.mlp.gate_proj`: `0.0430`
- `layers.19.mlp.up_proj`: `0.0427`
- `layers.1.mlp.up_proj`: `0.0423`
- `layers.4.mlp.up_proj`: `0.0413`

Depth summary:

- Early-layer mean `s_max`: `0.0181`
- Mid-layer mean `s_max`: `0.0173`
- Late-layer mean `s_max`: `0.0177`
- Early-layer mean `r90`: `6.59`
- Mid-layer mean `r90`: `6.81`
- Late-layer mean `r90`: `6.73`

## Interpretation

- The `MATH` pilot again shows the same broad structural pattern as the earlier
  `GSM8K` and `CommonsenseQA` pilots: adaptation energy is concentrated much
  more strongly in `MLP gate_proj` and `up_proj` than in attention `k_proj`
  or `v_proj`.
- Compared with the other two pilots, the absolute `s_max` scale is smaller,
  but the module ordering is preserved.
- `MATH` shows the highest mean `r90` of the three pilot tasks so far, which
  suggests a somewhat less aggressively compressible layerwise update structure
  than `CommonsenseQA`, despite the same LoRA target set and pilot length.
- Depth effects remain weaker than module-type effects; the dominant signal is
  still "MLP-heavy change" rather than a clean early-vs-late block effect.

## Limits

- This is a 50-step pilot rather than a registered Phase 1 run.
- Single-task SVD diagnostics cannot support shared-subspace claims.
- `analysis_report.json` and `run_manifest.json` were recovered from the Kaggle
  execution log because the pre-fix launcher did not also mirror them into the
  top-level artifact bundle.

## Next Step

Proceed to the next registered task collection run and keep checking:

- vector dimensionality matches `9,232,384`
- registered targets remain fully covered
- `MLP gate/up` dominance persists or breaks across task families
