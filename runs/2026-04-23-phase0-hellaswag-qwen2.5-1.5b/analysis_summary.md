# HellaSwag Pilot Analysis

This note summarizes the completed `pilot_only` Kaggle run for `HellaSwag` on `Qwen/Qwen2.5-1.5B-Instruct`.

## Result

- Training completed on `Tesla T4` in `674.4s` over `50` steps (`13.49s/step`).
- The registered update vector is non-degenerate and has shape `9,232,384` with norm `22.8755`.
- All registered target modules were covered and the SHA-256 checksum was `411b5696d868c764f131b34613a65f20b9f1e2331074c933147918ddf3ec5a70`.

## SVD Pattern

- Overall mean `r90`: `5.745`
- Overall mean `s_max`: `0.0380`
- Self-attention mean `s_max`: `0.0227`
- MLP mean `s_max`: `0.0583`
- Top layer by `s_max`: `base_model.model.model.layers.27.mlp.up_proj` at `0.0999`

The same pattern seen in the earlier pilots holds here: the strongest adaptation signal is concentrated in `mlp.gate_proj` and `mlp.up_proj`, while `k_proj` and `v_proj` remain weak. This is a pilot diagnostic only and should not be interpreted as registered H1a/H1b evidence.

