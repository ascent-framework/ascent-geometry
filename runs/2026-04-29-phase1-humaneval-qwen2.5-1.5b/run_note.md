# HumanEval — 1000-step Exploratory Run (2026-04-29)

## Status: COMPLETED (RUNTIME CAP)

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training
- Requested steps: 1000
- Actual completed steps: 230
- Stop reason: runtime 229.6m exceeded MAX_RUNTIME_MINUTES=220 at step 230
- Best reward: 0.7125 at step 200
- Final reward: 0.5375 at step 230
- Mean reward across logged checkpoints: 0.5190
- Total time: 13774.1s (229.6m)
- Effective per-step time: 59.89s

## Update Vector
- Shape: [9232384]
- Norm: 23.1705
- SHA-256: 3e91e6fee7680831cecf65d463ce569640a40f49616da5112cf1470f542d70ae
- Registered targets covered: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj

## Validation
- Training completed and adapter saved
- Registered update vector extracted successfully
- Update vector is non-degenerate (non-zero ratio: 1.0000)
- `checkpoint-200` exists alongside final adapter artifacts

## Notes
- HumanEval: code generation task, pass@1-based reward.
- max_completion_length=256, MAX_RUNTIME_MINUTES=220.
- Reward noisy throughout (0.3125~0.7125) — expected for code generation with 1.5B model.
- No clear convergence trend; reward oscillated rather than monotonically increasing.
- Stopped by runtime cap (not early-stop); still active at step 230.
- Norm (23.17) consistent with all completed tasks (23.01~23.21). Vector is usable.
