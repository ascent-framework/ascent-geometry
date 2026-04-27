# WinoGrande — 1000-step Exploratory Run (2026-04-27)

## Status: COMPLETED WITH EARLY STOP

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training
- Requested steps: 1000
- Actual completed steps: 270
- Early stop reason: no reward improvement for 180 steps
- Best reward: 0.6500 at step 90
- Final reward: 0.4000 at step 270
- Mean reward across logged checkpoints: 0.5005
- Total time: 2067.7s (34.5m)
- Effective per-step time: 7.66s

## Update Vector
- Shape: [9232384]
- Norm: 23.0079
- SHA-256: ea897f81214f4f44f94cefde63ad53c39a2a7e79bd32b3032ededd5d4c4ecd80
- Registered targets covered: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj

## Validation
- Training completed and adapter saved
- Registered update vector extracted successfully
- Update vector is non-degenerate (non-zero ratio: 1.0000)
- `checkpoint-250` exists alongside final adapter artifacts

## Notes
- WinoGrande uses binary choice format (option 1 or 2); random baseline = 0.5.
- max_completion_length=64, trust_remote_code=True (winogrande_xl config).
- Reward peaked at 0.6500 (step 90) but oscillated and never improved further.
- Fastest completion of all tasks: 34.5m. Early stop at step 270.
- Norm (23.01) slightly lower than other tasks (23.11~23.21) but still in usable range.
- Weak convergence expected: WinoGrande is a commonsense coreference task; GRPO signal sparse.
