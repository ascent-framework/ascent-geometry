# GSM8K — 1000-step Exploratory Run (2026-04-25)

## Status: COMPLETED WITH EARLY STOP

## Hardware
- GPU: Tesla T4
- Precision: bf16
- Torch: 2.10.0+cu128

## Training
- Requested steps: 1000
- Actual completed steps: 460
- Early stop reason: no reward improvement for 180 steps
- Best reward: 0.9125 at step 280
- Final reward: 0.6750 at step 460
- Mean reward across logged checkpoints: 0.7231
- Total time: 26032.3s (7.23h)
- Effective per-step time: 56.59s

## Update Vector
- Shape: [9232384]
- Norm: 23.2130
- SHA-256: 0054d0cc2476d96cec68e1b938877922fd22896b8f1e2840adae24507e73dd3f
- Registered targets covered: down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj

## Validation
- Training completed and adapter saved
- Registered update vector extracted successfully
- Update vector is non-degenerate
- `checkpoint-450` exists alongside final adapter artifacts

## Notes
- This run produced a usable full-run task vector even though it did not reach step 1000.
- The stop condition indicates plateau after convergence, not immediate reward collapse.
- The raw `run_report.json` from this run reports `per_step_time_sec=26.03` because it divides by requested `max_steps`; the corrected effective value is `56.59s` based on `460` actual steps.
- This result is exploratory evidence for vector collection and interim H1a/H1b checking, not registered primary evidence by itself.
