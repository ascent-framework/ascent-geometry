# training

GRPO training scripts for registered tasks.

## Scope

Train task-specific LoRA adapters on registered models using TRL/GRPO.
Phase 0 target: `Qwen2.5-1.5B-Instruct` on `GSM8K`.

## Registered adapter targets

`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Outputs

- Adapter checkpoint (saved to Kaggle Dataset or Google Drive — not committed here)
- Training log with run config, reward/loss traces, runtime notes
