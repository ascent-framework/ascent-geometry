# Experiment Checklist

Operational checklist for ASCENT-G experiment execution in this repository.

This checklist is aligned to:

- `../ascent-framework/docs/phase0/execution-plan.md`
- `../ascent-framework/docs/preregistration/v1.3.md`

## Phase 0: One complete pilot run

Goal: validate one end-to-end `train → extract → SVD` pipeline on
`Qwen2.5-1.5B-Instruct` and `GSM8K`.

### Preflight

- [ ] Confirm current branch / commit before running
- [ ] Confirm `config/registry.json` matches the intended Phase 0 target
- [ ] Confirm `config/task_registry.json` marks `GSM8K` as implemented
- [ ] Decide the execution environment: local GPU or Kaggle
- [ ] Decide artifact root location before the run starts
- [ ] Create or identify the target dated run directory name

### Environment

- [ ] Create and activate a dedicated Python environment
- [ ] Install dependencies from `requirements.txt`
- [ ] Run `python scripts/verify_env.py`
- [ ] Confirm GPU availability in the target environment
- [ ] Log Python version, torch version, GPU model, and precision mode
- [ ] Record whether the environment exactly matches `requirements.txt`
- [ ] Confirm that `runs/phase0_run_note_template.md` is available

### Orchestration

- [ ] Decide whether to run:
  - `runs/run_phase0_pipeline.py`
  - or stage scripts individually
- [ ] If using the pipeline runner, verify `--date`, `--task`, `--model-id`, and `--artifacts-root`
- [ ] Run `python runs/run_phase0_pipeline.py --date <YYYY-MM-DD> --dry-run`
- [ ] Confirm the dry-run command paths and output locations look correct

### Training

- [ ] Run `training/phase0_gsm8k_grpo.py`, `training/train_grpo_task.py`, or `runs/run_phase0_pipeline.py`
- [ ] Confirm the model loads successfully
- [ ] Confirm GRPO training reaches the planned stopping point
- [ ] Record actual training command used
- [ ] Confirm `training_run_report.json` is written
- [ ] Save the adapter checkpoint
- [ ] Reload the saved adapter successfully
- [ ] Confirm adapter artifact path is recorded in the report
- [ ] Confirm runtime metadata includes GPU model, precision, and per-step timing

### Extraction

- [ ] Run `extraction/extract_registered_update_vector.py`
- [ ] Confirm extraction uses the adapter from the training stage
- [ ] Confirm the registered object is `concat(ΔW_A, ΔW_B)`
- [ ] Confirm all registered target modules are covered
- [ ] Confirm the update vector is non-degenerate
- [ ] Save the vector file and provenance JSON
- [ ] Write the extraction stage report
- [ ] Confirm SHA-256 checksum exists in provenance / report
- [ ] Confirm vector shape is recorded

### Analysis

- [ ] Run `analysis/pilot_svd_diagnostic.py`
- [ ] Confirm analysis reads the same adapter used for extraction
- [ ] Confirm at least one SVD diagnostic completes successfully
- [ ] Save the analysis stage report
- [ ] Confirm the result is labeled `pilot_only`
- [ ] Confirm the report does not claim registered H1a/H1b evidence

### Run packaging

- [ ] Create a dated run directory under `runs/`
- [ ] Save `training_report.json`
- [ ] Save `extraction_report.json`
- [ ] Save `analysis_report.json`
- [ ] Save `run_manifest.json`
- [ ] Save `run_note.md`
- [ ] Copy or fill `runs/phase0_run_note_template.md` into the dated run directory
- [ ] Add a short run note: what worked, what failed, what to fix next
- [ ] Verify report paths inside the manifest
- [ ] Verify large artifacts are outside the repository tree

### Phase 0 completion check

- [ ] `model_loaded = true`
- [ ] `training_completed = true`
- [ ] `adapter_saved_and_reloaded = true`
- [ ] `update_vector_extracted = true`
- [ ] `update_vector_non_degenerate = true`
- [ ] `svd_diagnostic_ran = true`
- [ ] Mark Phase 0 as complete only if all items above are true

## Phase 1 preparation: multi-task collection

Goal: collect registered update vectors for the minimum registered task set.

### Task coverage

- [ ] GSM8K
- [ ] MATH
- [ ] AIME
- [ ] AMC
- [ ] MATH500
- [ ] HumanEval
- [ ] MBPP
- [ ] CommonsenseQA
- [ ] HellaSwag
- [ ] ARC-Challenge

### Collection workflow

- [x] Generalize the training runner beyond GSM8K
- [x] Generalize task-specific reward and formatting logic
- [x] Implement a second registered task path (`CommonsenseQA`)
- [x] Add preflight validation for task config and prompt rendering
- [ ] Run training for each registered task
- [ ] Extract one registered update vector per task
- [ ] Keep vector dimensions identical across tasks
- [ ] Record vector norms for magnitude analysis
- [ ] Record all task exclusions with reasons and diagnostics

### Minimum readiness

- [ ] At least 10 valid task vectors exist
- [ ] All vectors share the same dimensionality
- [ ] All vectors are non-zero
- [ ] Every task has a matching stage report or exclusion note

## Registered analysis: H1a and H1b

Goal: evaluate shared subspace structure and directional separability.

### H1a

- [ ] Run `analysis/h1a_h1b_task_matrix.py`
- [ ] Input at least 10 task vectors
- [ ] Normalize vectors for direction analysis
- [ ] Compute SVD on the task matrix
- [ ] Compute `r_90`
- [ ] Compute `rho`
- [ ] Compute bootstrap 95% CI for `rho`
- [ ] Record the H1a decision tier

### H1b

- [ ] Use the top-`r_90` basis from H1a
- [ ] Project all normalized task vectors into the shared subspace
- [ ] Compute pairwise cosine similarities
- [ ] Record mean `|cos|`
- [ ] Record max `|cos|`
- [ ] Record the H1b decision

### Outputs

- [ ] Save the registered H1a/H1b report under `runs/`
- [ ] Save a separate phase-level summary note
- [ ] Mark clearly whether the registered minimum task count was met

## Phase 2: H2 transfer experiments

Goal: evaluate transferability across the pre-registered model pairs.

### Primary setting

- [ ] Qwen2.5-1.5B-Instruct → Qwen2.5-1.5B

### Secondary settings

- [ ] Qwen2.5-1.5B → Qwen2.5-3B
- [ ] Qwen2.5-1.5B → google/gemma-2-2b-it

### Workflow

- [ ] Define the transfer evaluation procedure
- [ ] Compute retention for the primary setting
- [ ] Record the H2 decision for the primary setting
- [ ] Label secondary settings as exploratory

## Exploratory analyses

These do not block the registered primary path.

- [ ] H1c semantic alignment preparation
- [ ] H1c random-baseline comparison
- [ ] H3 self-similarity workflow
- [ ] Magnitude trend analysis from recorded vector norms

## Execution priority

Recommended order for the next concrete steps:

1. [ ] Finish one real Phase 0 run
2. [x] Generalize multi-task training and extraction
3. [ ] Collect 10 registered task vectors
4. [ ] Run H1a and H1b
5. [ ] Implement and run H2 transfer
6. [ ] Run exploratory analyses
