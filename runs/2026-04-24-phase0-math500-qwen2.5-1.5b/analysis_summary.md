# MATH500 Pilot Analysis

This note summarizes the completed `pilot_only` Kaggle run for `MATH500` on `Qwen/Qwen2.5-1.5B-Instruct`.

## Result

- Training completed on `Tesla T4` in `3280.1s` over `50` steps (`65.6s/step`).
- The registered update vector is non-degenerate and has shape `9,232,384` with norm `22.8853`.
- All registered target modules were covered and the SHA-256 checksum was `85f7ad82ab0089e93d6445a708612e16d93d87199b947c7b4d0bb30a473b7d89`.

## SVD Pattern

- Overall mean `r90`: `6.541`
- Overall mean `s_max`: `0.0444`
- Self-attention mean `s_max`: `0.0254`
- MLP mean `s_max`: `0.0696`
- Top layer by `s_max`: `base_model.model.model.layers.1.mlp.gate_proj` at `0.1124`

MATH500 also shows the same pilot MLP-heavy adaptation pattern (`mlp.gate_proj` / `mlp.up_proj` stronger than attention `k_proj` / `v_proj`). This remains a pilot diagnostic only and should not be interpreted as registered H1a/H1b evidence.
