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

- [x] `model_loaded = true`
- [x] `training_completed = true`
- [x] `adapter_saved_and_reloaded = true`
- [x] `update_vector_extracted = true`
- [x] `update_vector_non_degenerate = true`
- [x] `svd_diagnostic_ran = true`
- [x] Mark Phase 0 as complete only if all items above are true

## Phase 1 preparation: multi-task collection

Goal: collect registered update vectors for the minimum registered task set.

### Task coverage

- [x] GSM8K
- [x] MATH
- [x] AIME
- [x] AMC
- [x] MATH500
- [x] HumanEval
- [x] MBPP
- [x] CommonsenseQA
- [x] HellaSwag
- [x] ARC-Challenge

### Collection workflow

- [x] Generalize the training runner beyond GSM8K
- [x] Generalize task-specific reward and formatting logic
- [x] Implement a second registered task path (`CommonsenseQA`)
- [x] Add preflight validation for task config and prompt rendering
- [ ] Run training for each registered task
- [ ] Extract one registered update vector per task
- [ ] Keep vector dimensions identical across tasks
- [ ] Record vector norms for magnitude analysis
- [x] Record all task exclusions with reasons and diagnostics

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

1. [x] Finish one real Phase 0 run
2. [x] Generalize multi-task training and extraction
3. [ ] Collect 10 registered task vectors
4. [ ] Run H1a and H1b
5. [ ] Implement and run H2 transfer
6. [ ] Run exploratory analyses

## Status Note

- `2026-04-22`: One Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `GSM8K`, and the imported run record is stored
  under `runs/2026-04-22-phase0-gsm8k-qwen2.5-1.5b/`.
- `2026-04-22`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `CommonsenseQA`, and the imported run record is
  stored under `runs/2026-04-22-phase0-commonsenseqa-qwen2.5-1.5b/`.
- `2026-04-22`: Two valid pilot task vectors now exist: `GSM8K` and
  `CommonsenseQA`. Both runs produced non-degenerate registered update vectors
  with matching dimensionality (`9,232,384`) and recorded vector norms.
- `2026-04-22`: The reusable multi-task path has now been exercised end-to-end
  on both a reasoning task (`GSM8K`) and a multiple-choice task
  (`CommonsenseQA`).
- `2026-04-23`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `MATH`, and the imported run record is stored
  under `runs/2026-04-23-phase0-math-qwen2.5-1.5b/`.
- `2026-04-23`: Three valid pilot task vectors now exist: `GSM8K`,
  `CommonsenseQA`, and `MATH`. All three are non-degenerate and share the same
  dimensionality (`9,232,384`).
- `2026-04-23`: A launcher archival gap was identified because `analysis` and
  `manifest` reports were not mirrored into the top-level artifact bundle; the
  pipeline has now been updated so future runs copy standardized reports into
  `artifacts_root`.
- `2026-04-23`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `HellaSwag`, and the imported run record is stored
  under `runs/2026-04-23-phase0-hellaswag-qwen2.5-1.5b/`.
- `2026-04-23`: Four valid pilot task vectors now exist: `GSM8K`,
  `CommonsenseQA`, `MATH`, and `HellaSwag`. All four are non-degenerate and
  share the same dimensionality (`9,232,384`).
- `2026-04-23`: The reusable task path now also covers `HellaSwag`; the task
  registry, training formatter, and label normalization are wired and validated
  by a Kaggle pilot run.
- `2026-04-23`: The reusable task path now also covers `ARC-Challenge`; the
  task registry, training formatter, and label normalization are wired and
  validated by a Kaggle pilot run.
- `2026-04-23`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `ARC-Challenge`, and the imported run record is
  stored under `runs/2026-04-23-phase0-arc-challenge-qwen2.5-1.5b/`.
- `2026-04-23`: Five valid pilot task vectors now exist: `GSM8K`,
  `CommonsenseQA`, `MATH`, `HellaSwag`, and `ARC-Challenge`. All five are
  non-degenerate and share the same dimensionality (`9,232,384`).
- `2026-04-23`: The reusable task path now also covers `MBPP`; the task
  registry, code formatter, and normalized code exact-match reward are wired,
  and the Kaggle pilot run completed successfully on `Tesla T4`.
- `2026-04-23`: Six valid pilot task vectors now exist: `GSM8K`,
  `CommonsenseQA`, `MATH`, `HellaSwag`, `ARC-Challenge`, and `MBPP`. All six
  are non-degenerate and share the same dimensionality (`9,232,384`).
- `2026-04-23`: The MBPP pilot run completed after correcting the dataset
  field mapping to `text`/`code`; the notebook and registry now agree on the
  MBPP schema.
- `2026-04-23`: The MBPP pilot run is retained as a logged capture, but it is
  excluded from geometry analysis because the reward signal remained flat and
  the SVD diagnostic was degenerate.
- `2026-04-23`: MBPP is documented as an exclusion note with the prior
  blocker history and diagnostics preserved in `runs/`.
- `2026-04-24`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `MATH500`, and the imported run record is stored
  under `runs/2026-04-24-phase0-math500-qwen2.5-1.5b/`.
- `2026-04-24`: Seven pilot task vectors are now captured in run records
  (`GSM8K`, `CommonsenseQA`, `MATH`, `HellaSwag`, `ARC-Challenge`, `MBPP`,
  `MATH500`), all with matching dimensionality (`9,232,384`); `MBPP` remains
  excluded from geometry analysis due to prior degenerate SVD diagnostics.
- `2026-04-23`: The AIME task path has been wired using `math-ai/aime25`
  (`problem`/`answer`) and the reward remains
  `final_number_exact_match`.
- `2026-04-24`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `AIME`, and the imported run record is stored
  under `runs/2026-04-23-phase0-aime-qwen2.5-1.5b/`.
- `2026-04-24`: Six valid pilot task vectors now exist:
  `GSM8K`, `CommonsenseQA`, `MATH`, `HellaSwag`, `ARC-Challenge`, and `AIME`.
  These analyzed vectors are non-degenerate and share dimensionality
  (`9,232,384`).
- `2026-04-24`: The AMC task path has been wired using
  `kaggle-aimo/amc_filtered` (`task`/`answer`) and
  `final_number_exact_match`, with notebook and kernel metadata scaffolded.
- `2026-04-24`: The MATH500 task path has been wired using
  `HuggingFaceH4/MATH-500` (`problem`/`answer`) and
  `final_number_exact_match`, with notebook and kernel metadata scaffolded.
- `2026-04-24`: The HumanEval task path has been wired using
  `openai/openai_humaneval` (`prompt`/`canonical_solution`) and
  `humaneval_test_pass` over `test` plus `entry_point`, with notebook and
  kernel metadata scaffolded.
- `2026-04-24`: Code-task reward execution now preserves Python indentation
  when normalizing generated code, preventing false-zero reward outcomes from
  invalidly de-indented function bodies in `MBPP`/`HumanEval`.
- `2026-04-24`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `HumanEval`, and the imported run record is stored
  under `runs/2026-04-24-phase0-humaneval-qwen2.5-1.5b/`.
- `2026-04-24`: Nine analyzed pilot task vectors now exist:
  `GSM8K`, `CommonsenseQA`, `MATH`, `HellaSwag`, `ARC-Challenge`, `AIME`,
  `AMC`, `MATH500`, and `HumanEval`. These analyzed vectors are non-degenerate
  and share dimensionality (`9,232,384`); `MBPP` remains excluded from
  geometry analysis due to prior degenerate SVD diagnostics.
- `2026-04-24`: MBPP rerun patch set landed for the next pilot:
  (1) code normalization preserves indentation, (2) MBPP prompt includes the
  first test assertion, and (3) extraction report now logs
  `effective_delta_non_zero` via aggregate `b_norm_total` to prevent false-pass
  non-degeneracy checks from LoRA initialization alone.
- `2026-04-24`: One additional Kaggle Phase 0 pilot rerun completed for
  `Qwen2.5-1.5B-Instruct` on `MBPP`, and the imported run record is stored
  under `runs/2026-04-24-phase0-mbpp-qwen2.5-1.5b/`.
- `2026-04-24`: The MBPP rerun restored a live reward signal
  (`pass_rate ~= 0.395`, `timeouts = 0`), passed the new
  `effective_delta_non_zero` gate (`b_norm_total = 1.1453`), and produced
  non-zero `s_max` in all 196 captured layers, so MBPP is no longer excluded
  from pilot geometry analysis.
- `2026-04-24`: Ten analyzed pilot task vectors now exist:
  `GSM8K`, `CommonsenseQA`, `MATH`, `HellaSwag`, `ARC-Challenge`, `AIME`,
  `AMC`, `MATH500`, `HumanEval`, and `MBPP`. These analyzed vectors are
  non-degenerate and share dimensionality (`9,232,384`).
- `2026-04-24`: One additional Kaggle Phase 0 pilot run completed for
  `Qwen2.5-1.5B-Instruct` on `AMC`, and the imported run record is stored
  under `runs/2026-04-24-phase0-amc-qwen2.5-1.5b/`.
- `2026-04-24`: Seven valid pilot task vectors now exist:
  `GSM8K`, `CommonsenseQA`, `MATH`, `HellaSwag`, `ARC-Challenge`, `AIME`, and
  `AMC`. These analyzed vectors are non-degenerate and share dimensionality
  (`9,232,384`).
- Remaining unchecked Phase 0 items above should be treated as "not recorded"
  rather than "known failed" where the imported Kaggle report did not preserve
  full operator-side preflight or environment details.
- Remaining unchecked Phase 1 collection items should be read as "not complete
  for the full registered task set" rather than "no evidence exists yet" where
  partial progress is now documented in the run records above.
