# ASCENT-G Conversation Summary

**Date:** 2026-04-26  
**Scope:** HumanEval rerun, H1a/H1b interpretation, layer/block diagnostics, and runtime bug fixes

## 1. Main Questions And Answers

- `H1a` is a task-level geometry hypothesis, so it must be evaluated on the normalized task matrix, not on a single layer.
- Layerwise or blockwise SVD is still useful, but it is a follow-up diagnostic that explains where the structure lives.
- The current training setup freezes the base model and trains only the LoRA adapter parameters.
- The HumanEval rerun initially stalled because the reward path used raw `exec` without timeout protection.
- The odd stdout values seen in the kernel log were not training metrics. They were leaked runtime prints from the old HumanEval evaluation path.

## 2. How To Read H1a

`H1a` asks whether task updates concentrate into a shared low-dimensional subspace.

That is why the registered analysis works on:

- `concat(ΔW_A, ΔW_B)` task vectors
- normalized task matrix
- SVD over the task dimension
- `r90`, `rho`, and bootstrap confidence intervals

This is the right level for H1a because the hypothesis is about **task-to-task geometry**.

Layerwise or blockwise SVD answers a different question:

- which layer family carries the strongest adaptation signal
- whether attention and MLP blocks behave differently
- whether the low-dimensional structure is localized to specific projections

So the correct reading is:

- task matrix = H1a proper
- layer/block diagnostics = explanatory follow-up

## 3. What The Current Results Say

### Task-matrix level

- `50-step` 10-task pilot: `H1a inconclusive`, `H1b pass`
- `1000-step` preview on 3 tasks: `H1b pass` with very small mean cosine similarity
- `1000-step` quick check on 4 tasks: still strongly directionally distinct

Interpretation:

- H1b is currently favorable.
- H1a is still not favorable enough to call a pass.
- The normalized task vectors do not yet show a clear shared low-dimensional manifold.

### Layer/block level

Across the pilot analyses, the repeated pattern is:

- MLP projections, especially `gate_proj` and `up_proj`, tend to dominate
- attention `k_proj` and `v_proj` are usually weaker
- this pattern shows up in GSM8K, CommonsenseQA, ARC-Challenge, HellaSwag, AIME, AMC, MATH500, HumanEval, and MBPP pilot notes

Interpretation:

- the adaptation signal appears MLP-heavy
- this is a useful architectural signal
- but it does not by itself prove H1a

## 4. HumanEval Runtime Bug And Fix

The HumanEval run initially looked broken because progress logs stopped after early steps.

Root cause:

- HumanEval evaluation used direct `exec` of candidate code plus tests
- there was no subprocess timeout
- a slow or hanging completion could block a whole batch
- that prevented later `logging_steps` output from being emitted

Fix:

- moved HumanEval evaluation into a subprocess
- added a timeout
- redirected stdout/stderr away from the main log stream
- added `[HUMANEVAL-REWARD]` batch progress logging
- removed noisy dataset sample prints from the notebook

Result:

- `train_progress.log` now records clean step logs
- the kernel stdout no longer mixes in the weird tuple/list prints from the old evaluation path
- the run advances past the previous stall point

## 5. LoRA Training Setup

The experiment is configured as:

- base model frozen
- LoRA adapters trained only
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

This means:

- the base `Qwen2.5-1.5B-Instruct` weights do not change
- the learnable parameters are the adapter matrices only
- the update-vector analysis is well-defined for comparing tasks

## 6. Current Working Conclusions

- H1b is the stronger signal so far.
- H1a is still inconclusive at the task-matrix level.
- The architecture-level pilot diagnostics suggest a consistent MLP-heavy update pattern.
- The HumanEval runtime bug was due to missing timeout isolation, not to an inherent failure of the training loop.
- The cleaned `train_progress.log` is the correct place to watch for progress.

## 7. Practical Next Step

If the goal is to test H1a seriously, the next analysis should be:

1. Build task matrices from the full set of registered task vectors.
2. Run the registered H1a/H1b analysis on the full task matrix.
3. Add an exploratory layer/block breakdown only after the task-matrix result is known.
4. Compare attention vs MLP separately if the full-matrix result is still inconclusive.

## 8. Files Referenced In This Summary

- [STATUS.md](/Users/son/prj/ascent-geometry/STATUS.md)
- [analysis/README.md](/Users/son/prj/ascent-geometry/analysis/README.md)
- [analysis/h1a_h1b_task_matrix.py](/Users/son/prj/ascent-geometry/analysis/h1a_h1b_task_matrix.py)
- [analysis/pilot_svd_diagnostic.py](/Users/son/prj/ascent-geometry/analysis/pilot_svd_diagnostic.py)
- [training/train_grpo_task.py](/Users/son/prj/ascent-geometry/training/train_grpo_task.py)
- [notebooks/phase0-humaneval-qwen2.5-1.5b-pilot.ipynb](/Users/son/prj/ascent-geometry/notebooks/phase0-humaneval-qwen2.5-1.5b-pilot.ipynb)
