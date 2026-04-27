# OpenbookQA — 1000-step Exploratory Run (2026-04-27)

## Status: COMPLETED WITH EARLY STOP

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training
- Requested steps: 1000
- Actual completed steps: 460
- Early stop reason: no reward improvement for 180 steps
- Best reward: 0.9250 at step 280
- Final reward: 0.5125 at step 460
- Mean reward across logged checkpoints: 0.6916
- Total time: 2918.5s (48.6m)
- Effective per-step time: 6.34s

## Update Vector
- Shape: [9232384]
- Norm: 23.1375
- SHA-256: 62c92d5212214381e28b83f3fd12695accc811fe53a82865f3a8f999b3cea27d
- Registered targets covered: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj

## Validation
- Training completed and adapter saved
- Registered update vector extracted successfully
- Update vector is non-degenerate (non-zero ratio: 1.0000)
- `checkpoint-450` exists alongside final adapter artifacts

## Notes
- OpenbookQA replaces AMC in exploratory task set v2 (4-choice science MCQ).
- max_completion_length=64 (MCQ short answer). No deviations from v2 plan.
- Reward started at 0.6500 (well above random baseline 0.25) and peaked at 0.9250.
- Same early-stop pattern as GSM8K and CommonsenseQA: convergence plateau after ~280 steps.
- Norm (23.14) consistent with other completed tasks (23.11~23.21). Vector is usable.
- Per-step time 6.34s is fastest of all tasks so far (max_completion_length=64 effect).
