# HumanEval Pilot Analysis

This note summarizes the completed `pilot_only` Kaggle run for `HumanEval` on `Qwen/Qwen2.5-1.5B-Instruct`.

## Result

- Training completed on `Tesla T4` in `3191.6s` over `50` steps (`63.83s/step`).
- The registered update vector is non-degenerate and has shape `9,232,384` with norm `22.8905`.
- All registered target modules were covered and the SHA-256 checksum was `be63373d4f174d390ea130c37bb537c77754a02ffda01ef034502b77be9fe46b`.

## SVD Pattern

- Overall mean `r90`: `6.490`
- Overall mean `s_max`: `0.0451`
- Self-attention mean `s_max`: `0.0265`
- MLP mean `s_max`: `0.0698`
- Top layer by `s_max`: `base_model.model.model.layers.5.mlp.up_proj` at `0.1264`

HumanEval shows the same pilot MLP-heavy adaptation pattern (`mlp.gate_proj` / `mlp.up_proj` stronger than attention `k_proj` / `v_proj`). This remains a pilot diagnostic only and should not be interpreted as registered H1a/H1b evidence.
