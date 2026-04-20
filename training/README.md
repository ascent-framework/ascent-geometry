# training

GRPO training scripts for registered tasks.

## Scope

Train task-specific LoRA adapters on registered models using TRL/GRPO.
Phase 0 target: `Qwen2.5-1.5B-Instruct` on `GSM8K`.

Phase 0 may use short pilot runs to validate the pipeline, but any deviation
from registered Phase 1 settings must be labeled explicitly as pilot-only in
the run log.

## Registered training settings (`v1.3`)

- Method: `GRPO`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- LoRA rank: `8`
- LoRA alpha: `16`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- Registered full run length: `1000` gradient steps per task

## Task exclusion criteria

Exclude a task from the registered primary analysis if any of the following
occur:

- Training loss diverges within the first 100 steps and does not recover
- Reward remains zero for more than 80% of rollouts at step 200
- No improvement over the base model after 1000 steps

## Registered adapter targets

`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Outputs

- Adapter checkpoint (saved to Kaggle Dataset or Google Drive — not committed here)
- Training log with run config, reward/loss traces, runtime notes
- Runtime metadata required by the Phase 0 execution plan:
  GPU model, precision mode, and per-step training time

## Current entry point

- `phase0_gsm8k_grpo.py`: Phase 0 pilot runner for GSM8K on
  `Qwen2.5-1.5B-Instruct`
- `train_grpo_task.py`: task-aware GRPO entrypoint backed by
  `config/task_registry.json`
