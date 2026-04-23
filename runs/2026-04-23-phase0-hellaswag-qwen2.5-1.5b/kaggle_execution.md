# HellaSwag Kaggle Execution

This note prepares the next `pilot_only` Kaggle run for `HellaSwag` on
`Qwen/Qwen2.5-1.5B-Instruct`.

## Goal

Run one end-to-end `train -> extract -> SVD` pilot for `HellaSwag` using the
task-aware CLI path, then import the outputs into this dated run directory.

## Preconditions

- Kaggle notebook session
- GPU set to `Tesla T4`
- Internet enabled
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `HellaSwag`
- Scope label: `pilot_only`

## Verified Preflight

These checks already passed locally:

- `python training/train_grpo_task.py --task HellaSwag --print-task-config`
- `python training/train_grpo_task.py --task HellaSwag --smoke-test-prompt`
- `python runs/run_phase0_pipeline.py --date 2026-04-23 --task hellaswag --model-id Qwen/Qwen2.5-1.5B-Instruct --artifacts-root /tmp/ascent-artifacts --dry-run`

## Dataset Choice

- Dataset: `allenai/hellaswag`
- Split: `train`
- Prompt format: `ctx_a` + `ctx_b`
- Choice labels: `A`, `B`, `C`, `D`
- Answer field: `label`

The task-aware formatter normalizes numeric labels into `A-D` so the current
`mcq_label_exact_match` reward path can be reused.

## Kaggle Run Command

Use the task-aware pipeline entrypoint:

```bash
python runs/run_phase0_pipeline.py \
  --date 2026-04-23 \
  --task hellaswag \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --artifacts-root /kaggle/working
```

This resolves to:

- training output dir:
  `/kaggle/working/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/`
- run dir:
  `runs/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/`

## Expected Stage Commands

Training:

```bash
python training/train_grpo_task.py \
  --task HellaSwag \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /kaggle/working/2026-04-23-phase0-hellaswag-qwen2.5-1.5b \
  --max-steps 50 \
  --scope pilot_only
```

Extraction:

```bash
python extraction/extract_registered_update_vector.py \
  --task HellaSwag \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path /kaggle/working/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/adapter \
  --output-path /kaggle/working/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/update_vector.npy
```

Analysis:

```bash
python analysis/pilot_svd_diagnostic.py \
  --task HellaSwag \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path /kaggle/working/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/adapter \
  --output-path runs/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/analysis_report.json
```

## Expected Outputs

Under `/kaggle/working/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/`:

- `adapter/`
- `checkpoint-50/`
- `training_run_report.json`
- `update_vector.npy`
- `update_vector_report.json`

Under `runs/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/`:

- `analysis_report.json`
- `run_manifest.json`
- `run_note.md`

## After the Run

1. Download the Kaggle outputs.
2. Replace the placeholder sections in `run_note.md`.
3. Import stage reports into this run directory if they were generated outside
   the repository workspace.
4. Add an `analysis_summary.md` only after reading the actual `svd_results`.
5. Update `EXPERIMENT_CHECKLIST.md` only after the vector is confirmed valid.

## Acceptance Target

- Model loads successfully
- Training reaches stopping point
- Adapter saves and reloads
- Registered update vector extracts successfully
- Vector is non-degenerate
- At least one pilot SVD diagnostic completes
