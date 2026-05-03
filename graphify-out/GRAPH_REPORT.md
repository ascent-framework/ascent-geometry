# Graph Report - .  (2026-05-01)

## Corpus Check
- Corpus is ~35,593 words - fits in a single context window. You may not need a graph.

## Summary
- 369 nodes · 437 edges · 38 communities detected
- Extraction: 81% EXTRACTED · 18% INFERRED · 1% AMBIGUOUS · INFERRED: 80 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Registered Update Vector Dimension 9,232,384` - 10 edges
2. `main()` - 9 edges
3. `Plan A Attention/MLP Functional Split` - 8 edges
4. `2026-04-24 MBPP Rerun` - 8 edges
5. `H2 Retention Experiment` - 7 edges
6. `run_h1a()` - 6 edges
7. `H1a Shared Subspace Analysis` - 6 edges
8. `H1b Directional Separability Analysis` - 6 edges
9. `WinoGrande 1000-step Exploratory Run` - 6 edges
10. `H1a/H1b 3-Task Preview` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Pilot-only Notebook Rationale` --semantically_similar_to--> `Pilot-only Status`  [INFERRED] [semantically similar]
  notebooks/README.md → runs/2026-04-24-phase0-mbpp-qwen2.5-1.5b/run_note.md
- `Phase 1 Multi-Task Vector Collection` --references--> `Registered Update Object concat(Delta W_A, Delta W_B)`  [EXTRACTED]
  EXPERIMENT_CHECKLIST.md → README.md
- `OpenbookQA 1000-Step Exploratory Run` --references--> `Registered Update Object concat(Delta W_A, Delta W_B)`  [EXTRACTED]
  runs/2026-04-27-phase1-openbookqa-qwen2.5-1.5b/run_note.md → README.md
- `MATH Pilot SVD Pattern` --semantically_similar_to--> `Pilot MLP-Heavy Adaptation Signal`  [EXTRACTED] [semantically similar]
  runs/2026-04-23-phase0-math-qwen2.5-1.5b/analysis_summary.md → README.md
- `HumanEval Pilot Analysis Summary` --semantically_similar_to--> `Pilot MLP-Heavy Adaptation Signal`  [EXTRACTED] [semantically similar]
  runs/2026-04-24-phase0-humaneval-qwen2.5-1.5b/analysis_summary.md → README.md

## Hyperedges (group relationships)
- **Phase 0 Train Extract SVD Pipeline** — training_grpo_training, extraction_update_vector_extraction, analysis_svd_geometry_analysis, readme_registered_update_object [EXTRACTED 1.00]
- **Registered v1.3 Task Set** — task_gsm8k, task_math, task_aime, task_amc, task_math500, task_humaneval, task_mbpp, task_commonsenseqa, task_hellaswag, task_arc_challenge [EXTRACTED 1.00]
- **Revised Exploratory 10-Task Set v2** — task_commonsenseqa, task_arc_challenge, task_hellaswag, task_gsm8k, task_humaneval, task_mbpp, task_svamp, task_openbookqa, task_arc_easy, task_winogrande [EXTRACTED 1.00]
- **Phase 0 Train Extract SVD Pilot Pattern** — kaggle_execution_commonsenseqa_pipeline_plan, kaggle_execution_arc_challenge_pipeline_plan, kaggle_execution_amc_pipeline_plan [EXTRACTED 1.00]
- **Cross-Task MLP-Heavy Pilot Pattern** — analysis_summary_gsm8k_mlp_heavy_pattern, analysis_summary_commonsenseqa_mlp_heavy_pattern, analysis_summary_arc_challenge_mlp_heavy_pattern [EXTRACTED 1.00]
- **H2 Retention Protocol Components** — run_plan_h2_instruct_tokenizer_protocol, run_plan_h2_retention_metric, run_plan_h2_decision_criteria [EXTRACTED 1.00]
- **Phase 1 Revised H1a/H1b Task Matrix** — h1a_h1b_revised_analysis, h1a_common_low_dim_subspace, h1b_pairwise_similarity, shared_update_vector_9232384 [EXTRACTED 1.00]
- **Functional Split Subspace Follow-up** — followup_plan_a_attention_mlp_split, followup_attention_modules, followup_mlp_modules, conversation_mlp_heavy_pattern [EXTRACTED 1.00]
- **Phase 0 Pilot Pipeline Runs** — phase0_task_aware_pipeline, math500_phase0_run, hellaswag_phase0_run, mbpp_kaggle_execution, pilot_svd_diagnostic [EXTRACTED 1.00]
- **H1 Preview Three-task Input Set** — h1_preview_run_note_commonsenseqa, h1_preview_run_note_arc_challenge, h1_preview_run_note_hellaswag [EXTRACTED 1.00]
- **Phase 0 Notebooks Operational Purpose** — notebooks_readme_phase_0_pilot_runs, notebooks_readme_pipeline_validation, notebooks_readme_operational_debugging [EXTRACTED 1.00]
- **MBPP Reward Signal Repair** — mbpp_run_note_indentation_preservation, mbpp_run_note_first_test_assertion_prompt, mbpp_run_note_timed_subprocess_execution [EXTRACTED 1.00]

## Communities

### Community 0 - "Pipeline Artifacts"
Cohesion: 0.06
Nodes (36): h1a_h1b_task_matrix.py, pilot_svd_diagnostic.py, prepare_h1a_h1b_inputs.py, Exploratory Task Set v2, Local Artifact Readiness for H1a/H1b, Phase 0 Train Extract SVD Pilot Pipeline, Phase 1 Multi-Task Vector Collection, Registered H1a and H1b Analysis (+28 more)

### Community 1 - "ARC H1 Preview"
Cohesion: 0.07
Nodes (36): ARC-Challenge 1000-step Registered Run, ARC-Challenge Update Vector, Learning Signal Rationale, ARC r90 Range Note, ARC-Challenge, CommonsenseQA, H1a/H1b 3-Task Preview, H1a Inconclusive (+28 more)

### Community 2 - "Training Utilities"
Cohesion: 0.12
Nodes (29): build_formatted_example(), code_exact_match(), detect_hardware(), _ensure_list_of_strings(), extract_code_block(), extract_final_number(), extract_final_option(), final_number_exact_match() (+21 more)

### Community 3 - "Adversarial Review"
Cohesion: 0.08
Nodes (26): Exploratory Scope Label Correction, registered_ready Task-count-only Logic Bug, Adversarial Review Needs Attention Verdict, Negative Results Geometry Paper Alternative, Non-Falsifiable H2 Due to Low Independent-Variable Variance, Task Arithmetic Safety Overclaim, Task Identity Validation Rationale, Same-format Tasks May Share Subspace Rationale (+18 more)

### Community 4 - "Task Cluster Runs"
Cohesion: 0.14
Nodes (20): ARC-Easy 1000-step Exploratory Run, Code Generation Task Cluster, Language Inference Task Cluster, Math Word-problem Task Cluster, MCQ Task Cluster, CommonsenseQA 1000-step Registered Run, GSM8K 1000-step Exploratory Run, allenai/hellaswag Dataset (+12 more)

### Community 5 - "MLP Pilot Patterns"
Cohesion: 0.11
Nodes (19): AIME MLP-Heavy Adaptation Pattern, AMC MLP-Heavy Adaptation Pattern, ARC-Challenge MLP-Heavy Adaptation Pattern, CommonsenseQA MLP-Heavy Adaptation Pattern, GSM8K MLP-Heavy Adaptation Pattern, GSM8K Phase 0 Pilot Analysis, GSM8K Pilot-Only Evidence Limit, GSM8K Registered Update Vector (+11 more)

### Community 6 - "Functional Split Followup"
Cohesion: 0.14
Nodes (17): HumanEval Subprocess Timeout Fix, Layer/block SVD Diagnostics, MLP-heavy Adaptation Pattern, Raw exec Without Timeout Failure Mode, Attention Modules q/k/v/o_proj, Attention Shared Pattern Rationale, MLP Modules gate/up/down_proj, MLP Task-specific Knowledge Rationale (+9 more)

### Community 7 - "H1 Analysis"
Cohesion: 0.24
Nodes (13): H1a Shared Subspace Analysis, H1b Directional Separability Analysis, Normalized Multi-Task Update Matrix, SVD-Based Geometry Analysis, 50-Step Pilot Inconclusive Rationale, 50-Step Pilot H1a/H1b Results, H1a/H1b Pilot Analysis Run Note, Populate Statistics Only After Multi-Task Vectors Exist (+5 more)

### Community 8 - "H2 Retention Design"
Cohesion: 0.15
Nodes (13): Preregistered Cross-Model Retention Alternative, H1a Failure, Option A Preregistered H2 Transfer Design, H1a Final Failure Across Functional and Cluster Analyses, Plan A Attention MLP Functional Split, Plan B Task Cluster H1a Analysis, Cross-Model Adapter Retention Hypothesis, H2 Retention Decision Criteria (+5 more)

### Community 9 - "Notebook Operations"
Cohesion: 0.17
Nodes (13): Kernel Metadata, Notebook Matching Rule, Task-specific kernel-metadata.json Files, Informative Update Object Rationale, Pilot-only Status, Registered H1a/H1b/H2 Evidence, Kaggle-ready Notebooks, notebooks/kernel-metadata Directory (+5 more)

### Community 10 - "Phase 0 Runs"
Cohesion: 0.18
Nodes (12): GSM8K Phase 0 Pilot Run, GSM8K Phase 0 Run Note, HumanEval Phase 0 Pilot Run, HumanEval Phase 0 Run Note, MATH Phase 0 Pilot Run, MATH Phase 0 Run Note, Qwen/Qwen2.5-1.5B-Instruct, GSM8K (+4 more)

### Community 11 - "Functional Split Code"
Cohesion: 0.31
Nodes (10): build_index_map(), compute_r90(), compute_rho(), h1a_decision(), main(), parse_args(), parse_vector_arg(), Return (attn_indices, mlp_indices) into the concatenated vector. (+2 more)

### Community 12 - "Task Matrix Code"
Cohesion: 0.38
Nodes (9): compute_r90(), compute_rho(), h1a_decision(), h1b_decision(), load_vectors(), main(), pairwise_abs_cosines(), parse_args() (+1 more)

### Community 13 - "Replacement Tasks"
Cohesion: 0.22
Nodes (10): Replacement Task Rationale, OpenbookQA 1000-Step Exploratory Run, OpenbookQA Replaces AMC Rationale, OpenbookQA 1000-Step Exploratory Run Note, Task Replacement History, ARC-Challenge, ARC-Easy, OpenbookQA (+2 more)

### Community 14 - "Negative Result Direction"
Cohesion: 0.22
Nodes (9): H1a Failure Is Not Paper Failure, Avoid HARKing Rationale, Preregistration v1.3 Sections 7-9, Understanding TinyLoRA Without Shared Adaptive Subspaces Recovery Path, Cross-model Transfer from Instruct to Base, Adapter Geometry Entangled with Instruct Fine-tuning Rationale, H2 Overall Verdict FAIL, H2 Retention Eval Results (+1 more)

### Community 15 - "Cluster Analysis Code"
Cohesion: 0.46
Nodes (7): compute_r90(), compute_rho(), h1a_decision(), main(), parse_args(), parse_vector_arg(), run_h1a()

### Community 16 - "Environment Setup"
Cohesion: 0.25
Nodes (8): ASCENT-G Baseline Package Set, Torch and BitsAndBytes Platform-Sensitive Dependencies, Dependency Pin Rationale, Environment-Specific Torch Override Rule, Bootstrap Virtualenv Helper, Python 3.11 Reproducible Workflow, Environment Verification Script, Dedicated Virtual Environment

### Community 17 - "MBPP Fixes"
Cohesion: 0.25
Nodes (8): MBPP Flat SVD Diagnostic, Effective Delta Extraction Gate, Preserve Indentation in normalize_code_text, Include First MBPP Test in Prompt, MBPP Zero-Reward Root Cause Analysis, Subprocess Timeout for MBPP Reward Execution, MBPP Phase 0 Pipeline Plan, MBPP Degenerate Pilot Run

### Community 18 - "GSM8K Training Code"
Cohesion: 0.48
Nodes (6): correctness_reward(), detect_hardware(), extract_final_number(), HardwareMetadata, main(), parse_args()

### Community 19 - "Input Prep Code"
Cohesion: 0.6
Nodes (5): candidate_paths(), load_json(), main(), parse_args(), sha256_file()

### Community 20 - "Reporting Code"
Cohesion: 0.47
Nodes (4): _jsonify(), make_stage_report(), Common report builders for ASCENT-G experiment scripts., utc_now_iso()

### Community 21 - "Phase 0 Runner"
Cohesion: 0.6
Nodes (5): copy_if_exists(), main(), parse_args(), resolve_registered_task_name(), run_command()

### Community 22 - "Repository Overview"
Cohesion: 0.4
Nodes (5): ascent-framework Theory and Registration Repository, ascent-geometry Repository, Phase 0 Execution Plan, Preregistration v1.3, Repository Structure

### Community 23 - "Run Metadata Policy"
Cohesion: 0.4
Nodes (5): MATH Launcher Archival Gap, Run Environment Recording Requirements, No Checkpoints or Raw Vectors in Runs Directory Policy, Canonical Stage Report Schema, Run Logs and Metadata

### Community 24 - "Reward Path Rationale"
Cohesion: 0.4
Nodes (5): Reward-path Bug Rationale, First Test Assertion in Prompt, Python Indentation Preservation, MBPP Reward Path, Timed Subprocess Execution

### Community 25 - "Task Registry"
Cohesion: 0.67
Nodes (3): get_task_config(), load_task_registry(), Helpers for loading the ASCENT-G task registry.

### Community 26 - "Vector Extraction Code"
Cohesion: 0.83
Nodes (3): extract_registered_update_vector(), main(), parse_args()

### Community 27 - "SVD Metrics"
Cohesion: 0.67
Nodes (4): Bootstrap 95 Percent Confidence Interval, r90 Energy Dimension Count, rho Normalized r90, H1a Pass Possibility Analysis

### Community 28 - "Environment Verification"
Cohesion: 1.0
Nodes (2): main(), try_import()

### Community 29 - "Orthogonality Concerns"
Cohesion: 0.67
Nodes (3): High-Dimensional Artifact Concern, H1b Vector Similarity Pass, Task Vector Orthogonality

### Community 30 - "Shared Utilities"
Cohesion: 1.0
Nodes (1): Shared utilities for ASCENT-G experiment scripts.

### Community 31 - "H2 Model Scope"
Cohesion: 1.0
Nodes (2): Phase 2 H2 Transfer Experiments, Registered Models v1.3

### Community 32 - "Task Exclusions"
Cohesion: 1.0
Nodes (2): Registered Tasks v1.3, Task Exclusion Criteria

### Community 33 - "AIME Task"
Cohesion: 1.0
Nodes (1): AIME

### Community 34 - "AMC Task"
Cohesion: 1.0
Nodes (1): AMC

### Community 35 - "MATH500 Task"
Cohesion: 1.0
Nodes (1): MATH500

### Community 36 - "CommonsenseQA Task"
Cohesion: 1.0
Nodes (1): CommonsenseQA

### Community 37 - "HellaSwag Task"
Cohesion: 1.0
Nodes (1): HellaSwag

## Ambiguous Edges - Review These
- `HellaSwag 1000-step Registered Run` → `MCQ Task Cluster`  [AMBIGUOUS]
  runs/2026-04-29-phase1-h1a-h1b-revised/concepts_qa.md · relation: conceptually_related_to
- `HellaSwag 1000-step Registered Run` → `Language Inference Task Cluster`  [AMBIGUOUS]
  runs/2026-04-29-phase1-h1a-h1b-revised/followup_experiment_plan.md · relation: conceptually_related_to
- `r90` → `ARC r90 Range Note`  [AMBIGUOUS]
  runs/2026-04-25-phase1-arc-challenge-qwen2.5-1.5b/run_note.md · relation: references

## Knowledge Gaps
- **122 isolated node(s):** `Return (attn_indices, mlp_indices) into the concatenated vector.`, `Run H1a on a (dim x n_tasks) normalized matrix.`, `Helpers for loading the ASCENT-G task registry.`, `Shared utilities for ASCENT-G experiment scripts.`, `Common report builders for ASCENT-G experiment scripts.` (+117 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Shared Utilities`** (2 nodes): `__init__.py`, `Shared utilities for ASCENT-G experiment scripts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `H2 Model Scope`** (2 nodes): `Phase 2 H2 Transfer Experiments`, `Registered Models v1.3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Task Exclusions`** (2 nodes): `Registered Tasks v1.3`, `Task Exclusion Criteria`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `AIME Task`** (1 nodes): `AIME`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `AMC Task`** (1 nodes): `AMC`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `MATH500 Task`** (1 nodes): `MATH500`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CommonsenseQA Task`** (1 nodes): `CommonsenseQA`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `HellaSwag Task`** (1 nodes): `HellaSwag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `HellaSwag 1000-step Registered Run` and `MCQ Task Cluster`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `HellaSwag 1000-step Registered Run` and `Language Inference Task Cluster`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `r90` and `ARC r90 Range Note`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `Revised Exploratory H1a/H1b Analysis` connect `Adversarial Review` to `Task Cluster Runs`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `H2 Design Options` connect `Adversarial Review` to `H2 Retention Design`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `main()` (e.g. with `parse_args()` and `parse_task_vector_arg()`) actually correct?**
  _`main()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `H2 Retention Experiment` (e.g. with `Preregistered Cross-Model Retention Alternative` and `Option A Preregistered H2 Transfer Design`) actually correct?**
  _`H2 Retention Experiment` has 2 INFERRED edges - model-reasoned connections that need verification._