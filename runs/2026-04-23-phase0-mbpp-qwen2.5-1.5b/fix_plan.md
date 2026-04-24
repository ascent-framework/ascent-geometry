# MBPP Phase 0 — Fix Plan

Branch: `claude/debug-mbpp-rl-failure-gO1I5`
Target run: `runs/2026-04-23-phase0-mbpp-qwen2.5-1.5b/`
Observed symptom: `reward = 0.0`, `loss = 0.0`, `grad_norm = 0.0` across all 50 GRPO steps; every layer `s_max = 0.0`; extraction gate nevertheless passed (`vector_non_degenerate = true`).

## 1. Root cause (recap)

1. **`normalize_code_text` strips leading whitespace per line**, destroying Python indentation. Every model-generated function body becomes non-indented, so `exec(candidate_code, ns)` inside `_run_mbpp_tests` raises `IndentationError`, which the `except Exception` swallows as `reward = 0`. Deterministic, 100% of samples.
2. **Prompt contains only the `text` field of MBPP**, which describes the task but omits the expected function signature. The signature only appears in `test_list` (`assert foo(...) == ...`). Even with fix #1, most completions would still fail on `NameError` because the model has to guess the name.
3. **GRPO advantage is `r_i - mean(group)`.** When every generation in a group yields reward 0, mean = 0, advantage = 0 for every token, so loss = 0 and no gradient flows. This exactly matches the observed all-zero telemetry.
4. **Extraction gate only checks `||concat(A, B)||`.** Because PEFT initializes LoRA with `A ~ random, B = 0`, the concat norm is non-zero from initialization alone, so the gate reports "non-degenerate" even when `B` was never updated (effective delta `scaling · B @ A = 0`). This let a no-learning run pass pilot acceptance.

## 2. Fix tiers

The plan is split into three tiers so each can be reviewed and landed independently.

### Tier 1 — restore RL signal (mandatory)

Files touched:
- `training/train_grpo_task.py`
- `notebooks/phase0-mbpp-qwen2.5-1.5b-pilot.ipynb`

#### 1.1 Preserve indentation in `normalize_code_text`

Location: `training/train_grpo_task.py:115-123` (and the identical function in notebook cell index 106).

Before:
```python
def normalize_code_text(text: str) -> str:
    code = extract_code_block(text)
    lines = []
    for raw_line in code.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)
```

After:
```python
def normalize_code_text(text: str) -> str:
    code = extract_code_block(text)
    lines = []
    for raw_line in code.splitlines():
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("#"):
            continue
        lines.append(raw_line.rstrip())
    return "\n".join(lines)
```

Notes:
- Comment and blank-line skipping behaviour is preserved.
- Leading indentation (the actual problem) is kept.
- `rstrip()` only removes trailing whitespace, which is safe for `exec`.
- `code_exact_match` still works because it applies the same normalization to both sides.

#### 1.2 Include the first test in the MBPP user prompt

Location: `training/train_grpo_task.py:279-297` (`build_formatted_example`) and notebook cell 92 (`format_example`).

Problem: MBPP’s `text` field does not expose the function name. The model must read at least one test to know the target signature. This is how MBPP is conventionally benchmarked.

Change `build_formatted_example` so that when the task is MBPP (or more generally `prompt_style == "chat_code"` with a `test_list` field), the user prompt appends the first assertion:

```python
def build_formatted_example(task_config, example):
    if task_config["prompt_style"] == "chat_mcq":
        ...
    else:
        prompt_field = task_config["fields"]["prompt"]
        answer_field = task_config["fields"]["answer"]
        user_prompt = str(example[prompt_field])
        answer_value = str(example[answer_field])

        test_field = task_config.get("fields", {}).get("test_list")
        if test_field and example.get(test_field):
            tests = example[test_field]
            first_test = tests[0] if isinstance(tests, (list, tuple)) and tests else None
            if first_test:
                user_prompt = (
                    f"{user_prompt}\n\n"
                    f"Your solution must satisfy this test:\n{first_test}"
                )

        answer_value = str(example[answer_field])

    payload = {"user_prompt": user_prompt, "answer": answer_value}
    ...
```

Mirror the same change in the notebook’s `format_example` cell — the Kaggle run executes the notebook, not the CLI, so both paths must match.

#### 1.3 Local smoke test

Run the existing smoke path to confirm the prompt wiring does not regress for the other tasks:

```bash
python training/train_grpo_task.py --task MBPP          --smoke-test-prompt
python training/train_grpo_task.py --task GSM8K         --smoke-test-prompt
python training/train_grpo_task.py --task CommonsenseQA --smoke-test-prompt
python training/train_grpo_task.py --task HellaSwag     --smoke-test-prompt
```

Expected: all four print valid JSON; only the MBPP payload contains the `Your solution must satisfy this test:` line.

Additional unit-style check for the indentation fix (inline):

```python
from training.train_grpo_task import normalize_code_text, mbpp_test_pass

code = "```python\ndef add_one(x):\n    return x + 1\n```"
assert "    return x + 1" in normalize_code_text(code)

rewards = mbpp_test_pass(
    completions=[code],
    answer=["def add_one(x):\n    return x + 1"],
    test_list=[["assert add_one(3) == 4", "assert add_one(-1) == 0"]],
    test_setup_code=[""],
    challenge_test_list=[[]],
)
assert rewards == [1.0]
```

### Tier 2 — extraction gate hardening (strongly recommended)

File touched: `extraction/extract_registered_update_vector.py`.

#### 2.1 Add an effective-delta check

Location: `extraction/extract_registered_update_vector.py:145-148` (`validation` dict) and the existing layer-meta accumulator at `:60-71`.

The current gate:
```python
"vector_non_degenerate": bool(np.linalg.norm(update_vector) > 0),
```
is true-positive biased because LoRA init keeps `||concat(A, B)|| > 0` even when `B` never moves. Add an independent check that inspects `b_norm` (already computed per layer) and, optionally, the effective delta norm `scaling · B @ A`:

```python
b_norm_total = float(
    np.sqrt(sum(float(layer["b_norm"]) ** 2 for layer in layer_meta))
)
effective_delta_non_zero = b_norm_total > 0.0
```

Write it into both the provenance file and the stage report:

```python
provenance["b_norm_total"]            = b_norm_total
provenance["effective_delta_non_zero"] = effective_delta_non_zero

stage_report["validation"]["b_norm_total"]            = b_norm_total
stage_report["validation"]["effective_delta_non_zero"] = effective_delta_non_zero
```

Keep `vector_non_degenerate` as-is so existing consumers are not broken; the new flag is purely additive.

#### 2.2 Optional: acceptance criterion toggle

`runs/run_phase0_pipeline.py` already aggregates acceptance criteria into `run_report.json`. Add `effective_delta_non_zero` to the acceptance set once Tier 1 is in place and validated on a real rerun; doing it before Tier 1 lands would retroactively fail the current MBPP artifact, which is what we want but should be a separate, visible decision.

### Tier 3 — execution safety (recommended before the next rerun)

File touched: `training/train_grpo_task.py` (and the notebook mirror).

Once Tier 1 fixes land, `reward > 0` samples will start appearing, which means the model’s generated code will actually execute against `assert` lists. Running arbitrary generated code in-process is unsafe:

- infinite loops block the training step (no timeout today),
- `sys.exit()` / `os._exit()` can kill the trainer,
- file-system / network side effects are unrestricted.

Proposed shape:

```python
import multiprocessing as mp

def _run_tests_in_subprocess(candidate_code, setup_code, tests, timeout_sec=3.0):
    def _target(q):
        try:
            ns = {}
            if setup_code.strip():
                exec(setup_code, ns)
            exec(candidate_code, ns)
            for t in tests:
                exec(t, ns)
            q.put(True)
        except Exception:
            q.put(False)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_target, args=(q,))
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return False
    try:
        return bool(q.get_nowait())
    except Exception:
        return False
```

Use `_run_tests_in_subprocess` inside `mbpp_test_pass` in place of the direct `_run_mbpp_tests`. On Kaggle the `spawn` context is required because the training process already holds CUDA.

Caveats:
- Subprocess spawn overhead is ~100 ms. With `num_generations=4` and `per_device_batch_size=2`, that is ~800 ms per step of pure reward cost. Acceptable at 50 pilot steps; revisit for full 1000-step runs.
- Do not wrap this in `ThreadPoolExecutor` — Python threads will not preempt a CPU-bound `exec` for the timeout.

Defer Tier 3 if the next rerun has to happen immediately, but land it before any `max_steps > 200` run.

## 3. Commit and push plan

One commit per tier, on branch `claude/debug-mbpp-rl-failure-gO1I5`:

1. `Fix MBPP reward: preserve indentation and expose test signature in prompt` (Tier 1)
2. `Add effective-delta gate to update-vector extraction` (Tier 2)
3. `Sandbox MBPP reward exec in a spawn subprocess with timeout` (Tier 3)

After each commit, run the local smoke commands listed in §1.3. Push with `git push -u origin claude/debug-mbpp-rl-failure-gO1I5`. Do not open a PR automatically — the user will request one separately if needed.

## 4. Verification plan

Local (this repo):
- Tier 1 smoke tests above.
- `python -c "from extraction.extract_registered_update_vector import ..."` import-only check for Tier 2 (full run needs a trained adapter).

Kaggle (next rerun, tracked separately):
- Launch the updated notebook on T4.
- Acceptance: within the first 10 GRPO steps, `reward` should be non-zero on at least one group; `grad_norm` should be `> 0`; at least one layer should show `b_norm > 0` in the extraction provenance.
- SVD diagnostic: at least some layers should report `s_max > 0` and `r90 > 1`. If every layer is still zero after 50 steps, escalate to Codex’s shaping suggestions (compile-success partial credit, best-of-N, SFT warm start) rather than assuming another bug.

## 5. Documentation updates

After the Kaggle rerun succeeds:

- Update `runs/2026-04-23-phase0-mbpp-qwen2.5-1.5b/run_note.md` with a new "Fix landed" section referencing this plan.
- Create a new dated run directory (`runs/YYYY-MM-DD-phase0-mbpp-qwen2.5-1.5b/`) for the fixed rerun; keep the 2026-04-23 directory as the historical "failed" record.
- Reconsider the MBPP exclusion note in `README.md` and `analysis_summary.md` only after the new run demonstrates a non-degenerate SVD.

## 6. Out of scope (explicit)

- Reward shaping (compile-success credit, partial-test credit) — only relevant after Tier 1 confirms the gradient channel is alive. Tracked as a follow-up.
- Swapping the 1.5B base model for a larger one — not required to exit the identically-zero regime.
- Importing HumanEval — the task registry has `HumanEval` as `planned`; wiring it up depends on Tier 1’s prompt-with-test pattern generalizing, which is intentional.
