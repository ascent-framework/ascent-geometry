# ARC-Easy — 1000-step Exploratory Run (2026-04-27)

## Status: COMPLETED WITH EARLY STOP

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training
- Requested steps: 1000
- Actual completed steps: 350
- Early stop reason: no reward improvement for 180 steps
- Best reward: 1.0000 at step 170
- Final reward: 0.9000 at step 350
- Mean reward across logged checkpoints: 0.7961
- Total time: 2594.4s (43.2m)
- Effective per-step time: 7.41s

## Update Vector
- Shape: [9232384]
- Norm: 23.1199
- SHA-256: 61d9faac590a74c8859fb0f6f24d6f387b35f16ac2c718c3d57274d9516cde82
- Registered targets covered: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj

## Validation
- Training completed and adapter saved
- Registered update vector extracted successfully
- Update vector is non-degenerate (non-zero ratio: 1.0000)
- `checkpoint-350` exists alongside final adapter artifacts

## Notes
- ARC-Easy replaces competition_math in exploratory task set v2.
- max_completion_length=64 (MCQ short answer). Same format as ARC-Challenge.
- Best reward 1.0000 — fastest convergence of all tasks so far.
- Completed in 43.2m, shortest runtime across all phase1 tasks.
- Norm (23.12) consistent with other completed tasks (23.11~23.21). Vector is usable.
