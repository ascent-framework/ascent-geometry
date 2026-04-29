# MBPP — 1000-step Exploratory Run (2026-04-29)

## Status: COMPLETED (RUNTIME CAP)

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training
- Requested steps: 1000
- Actual completed steps: 250
- Stop reason: runtime 224.2m exceeded MAX_RUNTIME_MINUTES=220 at step 250
- Best reward: 0.6750 at step 160
- Final reward: 0.4375 at step 250
- Mean reward across logged checkpoints: 0.4675
- Total time: 13450.3s (224.2m)
- Effective per-step time: 53.80s

## Update Vector
- Shape: [9232384]
- Norm: 23.1905
- SHA-256: a64b4eb7b0a8f76f7282e3928ac46e4208c38442e73306a2bab152eee5bc21cb
- Registered targets covered: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj

## Validation
- Training completed and adapter saved
- Registered update vector extracted successfully
- Update vector is non-degenerate (non-zero ratio: 1.0000)
- `checkpoint-250` exists alongside final adapter artifacts

## Notes
- MBPP: code generation task, pass@1-based reward.
- max_completion_length=256, MAX_RUNTIME_MINUTES=220.
- Reward noisy throughout (0.3125~0.6750) — consistent with HumanEval pattern.
- Best reward 0.6750 < HumanEval 0.7125; mean 0.4675 < HumanEval 0.5190.
- Norm (23.19) — highest across all completed tasks, vector is usable.
- Stopped by runtime cap; this is the 10th and final task of the exploratory collection.
