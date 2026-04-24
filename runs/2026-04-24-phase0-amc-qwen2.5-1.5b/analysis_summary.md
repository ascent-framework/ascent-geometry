# AMC Pilot Analysis

This note summarizes the completed `pilot_only` Kaggle run for `AMC` on `Qwen/Qwen2.5-1.5B-Instruct`.

## Result

- Training completed on `Tesla T4` in `3285.5s` over `50` steps (`65.71s/step`).
- The registered update vector is non-degenerate and has shape `9,232,384` with norm `22.8817`.
- All registered target modules were covered and the SHA-256 checksum was `ba5b982fe3f54cb5b939657e557c75062608935570b21957036411286a3c1844`.

## SVD Pattern

- Overall mean `r90`: `6.577`
- Overall mean `s_max`: `0.0389`
- Self-attention mean `s_max`: `0.0228`
- MLP mean `s_max`: `0.0603`
- Top layer by `s_max`: `base_model.model.model.layers.1.mlp.up_proj` at `0.1055`

AMC also shows the pilot MLP-heavy adaptation pattern (`mlp.gate_proj` / `mlp.up_proj` stronger than attention `k_proj` / `v_proj`). This remains a pilot diagnostic only and should not be interpreted as registered H1a/H1b evidence.
