# H2 Retention Experiment — Run Plan
**Date:** 2026-04-30  
**Experiment ID:** H2  
**Status:** Ready to run on Kaggle

---

## Hypothesis

**H2:** LoRA adapters trained on `Qwen2.5-1.5B-Instruct` via GRPO transfer meaningfully to the raw base model `Qwen2.5-1.5B`.

This matters because:
- If retention is high, the adapter is encoding task-specific knowledge in a model-agnostic way
- If retention is low, the adapter is exploiting Instruct-specific features (chat format alignment, RLHF initialization)
- High retention would support combining adapters across base/instruct variants; low retention limits transfer to same SFT branch

---

## Experiment Design

### Protocol

For each of 10 tasks:

1. **Instruct eval:** Load Phase 1 LoRA adapter on `Qwen2.5-1.5B-Instruct` → greedy decode 200 validation examples → `instruct_acc`
2. **Base eval:** Load same adapter on `Qwen2.5-1.5B` (base) using **Instruct tokenizer** (no chat template for base model) → same 200 examples → `base_acc`
3. **Retention:** `retention = base_acc / instruct_acc`

### Key implementation notes

- **Tokenizer:** Always load from `Qwen/Qwen2.5-1.5B-Instruct` for both model variants — the base model has no chat template
- **Adapter loading:** `PeftModel.from_pretrained(..., autocast_adapter_dtype=False)` — required for T4/fp16
- **Decoding:** Greedy (`do_sample=False`), decode only new tokens (`output[0][input_len:]`)
- **Memory:** Unload each model + `torch.cuda.empty_cache()` between instruct and base eval
- **Seed:** `shuffle(seed=42)` for reproducible 200-example subsample

### Decision criteria

| Retention | Verdict |
|-----------|---------|
| >= 0.70 | **PASS** — strong transfer |
| 0.30–0.69 | **MARGINAL** — partial transfer |
| < 0.30 | **FAIL** — adapter is Instruct-specific |

---

## Task List

### Batch 1: MCQ (notebook: h2-retention-mcq-eval.ipynb)

| Task | Dataset | Split | Answer field | max_new_tokens | Phase 1 reward |
|------|---------|-------|--------------|---------------|----------------|
| ARC-Easy | allenai/ai2_arc, ARC-Easy | validation | answerKey | 64 | 1.0000 |
| ARC-Challenge | allenai/ai2_arc, ARC-Challenge | validation | answerKey | 64 | unknown (fresh eval) |
| OpenbookQA | allenai/openbookqa, main | validation | answerKey | 64 | 0.9250 |
| CommonsenseQA | tau/commonsense_qa | validation | answerKey | 64 | unknown (fresh eval) |

### Batch 2: Math & Code (notebook: h2-retention-math-code-eval.ipynb)

| Task | Dataset | Split | Answer field | max_new_tokens | Phase 1 reward |
|------|---------|-------|--------------|---------------|----------------|
| GSM8K | gsm8k, main | test | answer (#### N) | 256 | 0.9125 |
| SVAMP | ChilleD/SVAMP | test | Answer | 256 | 0.9500 |
| HumanEval | openai/openai_humaneval | test | canonical_solution | 256 | 0.7125 |
| MBPP | google-research-datasets/mbpp | test | code | 256 | 0.6750 |

Note: WinoGrande (phase1 reward: 0.6500) and HellaSwag (unknown) are not included in the H2 notebooks per scope — add if needed.

---

## Adapter Paths

### Local (for upload to Kaggle dataset)

| Task | Local adapter path |
|------|--------------------|
| CommonsenseQA | `/private/tmp/phase1-vectors/commonsenseqa/commonsenseqa-qwen2.5-1.5b-phase0/adapter/` |
| ARC-Challenge | `/private/tmp/phase1-vectors/arc-challenge/arc-challenge-qwen2.5-1.5b-phase0/adapter/` |
| GSM8K | `/private/tmp/phase1-vectors/gsm8k/gsm8k-qwen2.5-1.5b-phase0/adapter/` |
| HellaSwag | `/private/tmp/phase1-vectors/hellaswag/hellaswag-qwen2.5-1.5b-phase0/adapter/` |
| OpenbookQA | `/private/tmp/openbookqa-output/openbookqa-qwen2.5-1.5b-phase1/adapter/` |
| ARC-Easy | `/private/tmp/arc-easy-output/arc-easy-qwen2.5-1.5b-phase1/adapter/` |
| WinoGrande | `/private/tmp/winogrande-output2/winogrande-qwen2.5-1.5b-phase1/adapter/` |
| SVAMP | `/private/tmp/svamp-output/svamp-qwen2.5-1.5b-phase1/adapter/` |
| HumanEval | `/private/tmp/humaneval-output/humaneval-qwen2.5-1.5b-phase1/adapter/` |
| MBPP | `/private/tmp/mbpp-output/mbpp-qwen2.5-1.5b-phase1/adapter/` |

### On Kaggle (after dataset upload)

```
/kaggle/input/ascent-g-phase1-adapters/{task_name}/adapter/
```

Where `{task_name}` is the slug used in the dataset (e.g. `arc-easy`, `gsm8k`, `humaneval`).

---

## Kaggle Dataset Upload Instructions

1. Create a new Kaggle dataset named `ascent-g-phase1-adapters` (private)
2. For each task, create a subdirectory with the task slug and upload the adapter files:

```
ascent-g-phase1-adapters/
  arc-easy/adapter/          ← adapter_config.json, adapter_model.safetensors, tokenizer files
  arc-challenge/adapter/
  openbookqa/adapter/
  commonsenseqa/adapter/
  gsm8k/adapter/
  svamp/adapter/
  humaneval/adapter/
  mbpp/adapter/
```

3. In the notebook settings, add the dataset as an input data source so it appears at `/kaggle/input/ascent-g-phase1-adapters/`

Upload command pattern (using Kaggle API):
```bash
# Create dataset metadata first, then push
kaggle datasets create -p /path/to/ascent-g-phase1-adapters/ --dir-mode zip
# or update existing:
kaggle datasets version -p /path/to/ascent-g-phase1-adapters/ -m "Phase 1 adapters for H2 eval"
```

---

## Phase 1 Baseline Scores (instruct model, best_reward)

| Task | Phase 1 best_reward | Source |
|------|--------------------:|--------|
| ARC-Easy | 1.0000 | run_report.json |
| ARC-Challenge | unknown | use fresh eval |
| OpenbookQA | 0.9250 | run_report.json |
| CommonsenseQA | unknown | use fresh eval |
| HellaSwag | unknown | use fresh eval |
| WinoGrande | 0.6500 | run_report.json |
| GSM8K | 0.9125 | run_report.json |
| SVAMP | 0.9500 | run_report.json |
| HumanEval | 0.7125 | run_report.json |
| MBPP | 0.6750 | run_report.json |

---

## Output Files

- `/kaggle/working/h2_retention_results_mcq.json` (MCQ batch)
- `/kaggle/working/h2_retention_results_math_code.json` (Math & Code batch)

Download these after each notebook completes and commit to `runs/2026-04-30-h2-retention-eval/`.

---

## GPU Time Estimate (Kaggle T4, single GPU)

Per-task estimate:
- **Model load (instruct + base):** ~2 min × 2 = 4 min per task
- **MCQ eval (200 examples × 64 tokens):** ~8 min per model × 2 = 16 min per task
- **Math eval (200 examples × 256 tokens):** ~15 min per model × 2 = 30 min per task
- **Code eval (200 examples × 256 tokens + exec):** ~20 min per model × 2 = 40 min per task

| Batch | Tasks | Estimated time |
|-------|-------|---------------|
| MCQ (4 tasks) | ARC-Easy, ARC-Challenge, OpenbookQA, CommonsenseQA | ~80 min |
| Math (2 tasks) | GSM8K, SVAMP | ~70 min |
| Code (2 tasks) | HumanEval, MBPP | ~90 min |
| **Total** | **8 tasks** | **~240 min (4h)** |

Kaggle T4 session limit is 12 hours — both notebooks should complete within a single session.
Run MCQ notebook first (faster), then Math & Code.

---

## Interpretation Guide

After collecting results, compute aggregate retention:
- If median retention across all 8 tasks > 0.70 → **H2 PASS**: adapters are base-transferable
- If median < 0.30 → **H2 FAIL**: adapters are Instruct-locked
- Mixed results → stratify by task type (MCQ vs math vs code) and investigate whether chat-template sensitivity drives the pattern
