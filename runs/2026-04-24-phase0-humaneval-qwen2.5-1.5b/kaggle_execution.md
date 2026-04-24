# HumanEval Kaggle Execution

This note prepares the next `pilot_only` Kaggle run for `HumanEval` on
`Qwen/Qwen2.5-1.5B-Instruct`.

## Goal

Run one end-to-end `train -> extract -> SVD` pilot for `HumanEval` using the
task-aware CLI path, then import the outputs into this dated run directory.

## Preconditions

- Kaggle notebook session
- GPU set to `Tesla T4`
- Internet enabled
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `HumanEval`
- Scope label: `pilot_only`

## Verified Preflight

These checks should pass locally before the Kaggle run:

- `python training/train_grpo_task.py --task HumanEval --print-task-config`
- `python training/train_grpo_task.py --task HumanEval --smoke-test-prompt`
- `python runs/run_phase0_pipeline.py --date 2026-04-24 --task humaneval --model-id Qwen/Qwen2.5-1.5B-Instruct --artifacts-root /tmp/ascent-artifacts --dry-run`

## Dataset Choice

- Dataset: `openai/openai_humaneval`
- Prompt field: `prompt`
- Reference solution field: `canonical_solution`
- Test fields: `test`, `entry_point`

The task uses the code prompt path and test-execution reward helper
`humaneval_test_pass`.

## Kaggle Run Command

Use the task-aware pipeline entrypoint:

```bash
python runs/run_phase0_pipeline.py \
  --date 2026-04-24 \
  --task humaneval \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --artifacts-root /kaggle/working
```

This resolves to:

- training output dir:
  `/kaggle/working/2026-04-24-phase0-humaneval-qwen2.5-1.5b/`
- run dir:
  `runs/2026-04-24-phase0-humaneval-qwen2.5-1.5b/`

## Expected Stage Commands

Training:

```bash
python training/train_grpo_task.py \
  --task HumanEval \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /kaggle/working/2026-04-24-phase0-humaneval-qwen2.5-1.5b \
  --max-steps 50 \
  --scope pilot_only
```

Extraction:

```bash
python extraction/extract_registered_update_vector.py \
  --task HumanEval \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path /kaggle/working/2026-04-24-phase0-humaneval-qwen2.5-1.5b/adapter \
  --output-path /kaggle/working/2026-04-24-phase0-humaneval-qwen2.5-1.5b/update_vector.npy
```

Analysis:

```bash
python analysis/pilot_svd_diagnostic.py \
  --task HumanEval \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path /kaggle/working/2026-04-24-phase0-humaneval-qwen2.5-1.5b/adapter \
  --output-path runs/2026-04-24-phase0-humaneval-qwen2.5-1.5b/analysis_report.json
```

## Expected Outputs

Under `/kaggle/working/2026-04-24-phase0-humaneval-qwen2.5-1.5b/`:

- `adapter/`
- `checkpoint-50/`
- `training_run_report.json`
- `update_vector.npy`
- `update_vector_report.json`

Under `runs/2026-04-24-phase0-humaneval-qwen2.5-1.5b/`:

- `analysis_report.json`
- `run_manifest.json`
- `run_note.md`

## After the Run

1. Download the Kaggle outputs.
2. Replace the placeholder sections in `run_note.md`.
3. Include a short `Failure History` section with any prior blockers.
4. Import stage reports into this run directory if they were generated outside
   the repository workspace.
5. Add an `analysis_summary.md` only after reading the actual `svd_results`.
6. Update `EXPERIMENT_CHECKLIST.md` only after the vector is confirmed valid.

## Acceptance Target

- Model loads successfully
- Training reaches stopping point
- Adapter saves and reloads
- Registered update vector extracts successfully
- Vector is non-degenerate
- At least one pilot SVD diagnostic completes
