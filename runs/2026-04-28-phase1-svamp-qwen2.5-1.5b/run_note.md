# SVAMP — 1000-step Exploratory Run (2026-04-28)

## Status: COMPLETED (RUNTIME CAP)

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training
- Requested steps: 1000
- Actual completed steps: 290
- Stop reason: runtime 224.1m exceeded MAX_RUNTIME_MINUTES=220 at step 290
- Best reward: 0.9500 at step 220
- Final reward: 0.7875 at step 290
- Mean reward across logged checkpoints: 0.8103
- Total time: 13446.1s (224.1m)
- Effective per-step time: 46.37s

## Update Vector
- Shape: [9232384]
- Norm: 23.1096
- SHA-256: fe51fbaf12778a2c37e00e32b077660dfcfaad117bc3e9914fc8bb937c942119
- Registered targets covered: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj

## Validation
- Training completed and adapter saved
- Registered update vector extracted successfully
- Update vector is non-degenerate (non-zero ratio: 1.0000)
- `checkpoint-250` exists alongside final adapter artifacts

## Notes
- SVAMP replaces AMC+MATH500 in exploratory task set v2 (수학 군집 유지 목적).
- Same reward function as GSM8K: `final_number_exact_match`.
- Best reward 0.9500 — highest of all math tasks, surpassing GSM8K (0.9125).
- Mean reward 0.8103 — highest across all completed tasks.
- Started at reward 0.6875 (step 10), already 0.8375 by step 60 — fast convergence.
- Stopped by runtime cap (not early-stop); reward was still active at step 290.
- Norm (23.11) consistent with other completed tasks (23.01~23.21). Vector is usable.
