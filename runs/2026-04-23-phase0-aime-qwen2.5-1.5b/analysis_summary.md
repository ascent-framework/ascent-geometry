# AIME Pilot Analysis

This note summarizes the completed `pilot_only` Kaggle run for `AIME` on `Qwen/Qwen2.5-1.5B-Instruct`.

## Result

- Training completed on `Tesla T4` in `3605.5s` over `50` steps (`72.11s/step`).
- The registered update vector is non-degenerate and has shape `9,232,384` with norm `22.8726`.
- All registered target modules were covered and the SHA-256 checksum was `fed5ffd2789a049157a0c8a60bb4be21edcfc7fba993f05cb3aa6c51e3598cb8`.

## SVD Pattern

- Overall mean `r90`: `6.699`
- Overall mean `s_max`: `0.0298`
- Self-attention mean `s_max`: `0.0172`
- MLP mean `s_max`: `0.0465`
- Top layer by `s_max`: `base_model.model.model.layers.1.mlp.gate_proj` at `0.0730`

The pilot again shows an MLP-heavy pattern (`mlp.gate_proj` / `mlp.up_proj` stronger than attention `k_proj` / `v_proj`). This remains a pilot diagnostic only and should not be interpreted as registered H1a/H1b evidence.
