# Experiment Checklist

Operational checklist for ASCENT-G experiment execution in this repository.

This checklist is aligned to:

- `../ascent-framework/docs/phase0/execution-plan.md`
- `../ascent-framework/docs/preregistration/v1.3.md`

## Phase 0: One complete pilot run

Goal: validate one end-to-end `train → extract → SVD` pipeline on
`Qwen2.5-1.5B-Instruct` and `GSM8K`.

### Environment

- [ ] Create and activate a dedicated Python environment
- [ ] Install dependencies from `requirements.txt`
- [ ] Run `python scripts/verify_env.py`
- [ ] Confirm GPU availability in the target environment
- [ ] Log Python version, torch version, GPU model, and precision mode

### Training

- [ ] Run `training/phase0_gsm8k_grpo.py` or `runs/run_phase0_pipeline.py`
- [ ] Confirm the model loads successfully
- [ ] Confirm GRPO training reaches the planned stopping point
- [ ] Save the adapter checkpoint
- [ ] Reload the saved adapter successfully
- [ ] Write `training_run_report.json`

### Extraction

- [ ] Run `extraction/extract_registered_update_vector.py`
- [ ] Confirm the registered object is `concat(ΔW_A, ΔW_B)`
- [ ] Confirm all registered target modules are covered
- [ ] Confirm the update vector is non-degenerate
- [ ] Save the vector file and provenance JSON
- [ ] Write the extraction stage report

### Analysis

- [ ] Run `analysis/pilot_svd_diagnostic.py`
- [ ] Confirm at least one SVD diagnostic completes successfully
- [ ] Save the analysis stage report
- [ ] Confirm the result is labeled `pilot_only`

### Run packaging

- [ ] Create a dated run directory under `runs/`
- [ ] Save `training_report.json`
- [ ] Save `extraction_report.json`
- [ ] Save `analysis_report.json`
- [ ] Save `run_manifest.json`
- [ ] Add a short run note: what worked, what failed, what to fix next

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

- [ ] Generalize the training runner beyond GSM8K
- [ ] Generalize task-specific reward and formatting logic
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
2. [ ] Generalize multi-task training and extraction
3. [ ] Collect 10 registered task vectors
4. [ ] Run H1a and H1b
5. [ ] Implement and run H2 transfer
6. [ ] Run exploratory analyses
