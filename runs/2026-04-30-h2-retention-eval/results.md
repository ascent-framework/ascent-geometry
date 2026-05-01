# H2 Retention Eval Results

**Date**: 2026-04-30  
**Branch**: feature/h2-retention-eval  
**Hardware**: Kaggle Tesla T4 (15.6 GB VRAM, bf16, torch 2.10.0+cu128)

## Protocol

| Parameter | Value |
|-----------|-------|
| Instruct model | Qwen/Qwen2.5-1.5B-Instruct |
| Base model | Qwen/Qwen2.5-1.5B |
| Adapters | chson0316/ascent-g-phase1-adapters (Phase 1 GRPO) |
| n_eval | 200 samples per task |
| Decoding | Greedy |
| Retention formula | base_acc / instruct_acc |
| Pass threshold | > 70% |
| Fail threshold | < 30% |
| Marginal | 30–70% |

## Results

### MCQ Batch

| Task | Phase1 Reward | Instruct Acc | Base Acc | Retention | Verdict |
|------|--------------|-------------|---------|-----------|---------|
| ARC-Easy | 1.000 | 84.0% | 24.0% | 28.6% | **FAIL** |
| ARC-Challenge | — | 75.5% | 26.5% | 35.1% | MARGINAL |
| OpenbookQA | 0.925 | 74.0% | 28.5% | 38.5% | MARGINAL |
| CommonsenseQA | — | 75.5% | 18.5% | 24.5% | **FAIL** |

### Math + Code Batch

| Task | Phase1 Reward | Instruct Acc | Base Acc | Retention | Verdict |
|------|--------------|-------------|---------|-----------|---------|
| GSM8K | 0.9125 | 60.0% | 6.0% | 10.0% | **FAIL** |
| SVAMP | 0.950 | 81.5% | 5.5% | 6.7% | **FAIL** |
| HumanEval | 0.7125 | 64.0% | 35.4% | 55.2% | MARGINAL |
| MBPP | 0.675 | 4.5% | 1.0% | 22.2% | **FAIL** |

## Summary

| Verdict | Tasks | Count |
|---------|-------|-------|
| PASS (>70%) | — | 0 / 8 |
| MARGINAL (30–70%) | ARC-Challenge, OpenbookQA, HumanEval | 3 / 8 |
| FAIL (<30%) | ARC-Easy, CommonsenseQA, GSM8K, SVAMP, MBPP | 5 / 8 |

## H2 Overall Verdict: **FAIL**

Phase 1 LoRA adapters trained on Qwen2.5-1.5B-Instruct do **not** transfer to Qwen2.5-1.5B (base). Retention is near zero for math tasks (GSM8K 10%, SVAMP 6.7%), low for MCQ (ARC-Easy 28.6%, CommonsenseQA 24.5%), and only marginal for code/reading tasks.

## Interpretation

- **Math tasks** (GSM8K, SVAMP): Near-zero retention. The base model has almost no math reasoning capability (~5-6% vs instruct 60-81%), suggesting the adapter is entirely dependent on instruction-tuning capabilities in the base model.
- **Code tasks** (HumanEval MARGINAL 55.2%, MBPP FAIL 22.2%): HumanEval shows partial retention possibly because base Qwen2.5 has inherent code capability. MBPP near-zero may reflect task framing sensitivity.
- **MCQ tasks**: Base model retains ~25-28% of instruct performance — consistent with random baseline for 4-choice tasks (~25%), suggesting near-zero learned signal transfer.

## Implications for Paper Direction

H2 result = **FAIL** supports the Codex adversarial review recommendation to reframe as a **negative result / descriptive geometry paper**:
- Phase 1 LoRA adapters are NOT task-arithmetic-safe for cross-model transfer
- Geometric independence (near-orthogonal task vectors) does not imply functional independence across model variants
- The adapter's geometry is entangled with the instruct fine-tuning of the base model
