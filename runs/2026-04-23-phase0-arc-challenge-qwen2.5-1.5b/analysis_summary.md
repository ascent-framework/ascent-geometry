# ARC-Challenge Pilot Analysis

This note summarizes the completed `pilot_only` Kaggle run for `ARC-Challenge` on `Qwen/Qwen2.5-1.5B-Instruct`.

## Result

- Training completed on `Tesla T4` in `325.2s` over `50` steps (`6.50s/step`).
- The registered update vector is non-degenerate and has shape `9,232,384` with norm `22.8902`.
- All registered target modules were covered and the SHA-256 checksum was `cf94440107893191e1fe5cf65cfa63987f1676e0902f55303319df4f4f4a048f`.

## SVD Pattern

- Overall mean `r90`: `5.730`
- Overall mean `s_max`: `0.0470`
- Self-attention mean `s_max`: `0.0265`
- MLP mean `s_max`: `0.0744`
- Top layer by `s_max`: `base_model.model.model.layers.27.mlp.gate_proj` at `0.1228`

The same pattern seen in the earlier pilots holds here: the strongest adaptation signal is concentrated in `mlp.gate_proj` and `mlp.up_proj`, while `k_proj` and `v_proj` remain weak. This is a pilot diagnostic only and should not be interpreted as registered H1a/H1b evidence.

