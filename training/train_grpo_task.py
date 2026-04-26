#!/usr/bin/env python3
"""Task-aware GRPO training entrypoint for ASCENT-G."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.reporting import make_stage_report, write_report
from common.task_registry import TASK_REGISTRY_PATH, get_task_config


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PROMPT_TEMPLATES = {
    "chat_reasoning": (
        "You are a reasoning assistant. "
        "Solve the problem carefully and end with a final answer."
    ),
    "chat_code": (
        "You are a coding assistant. "
        "Produce a correct solution that satisfies the task specification."
    ),
    "chat_mcq": (
        "You are a reasoning assistant. "
        "Choose the best answer option and make the final choice explicit as "
        "The answer is <option>."
    ),
}


@dataclass
class HardwareMetadata:
    gpu_model: str
    vram_gb: float
    bf16_supported: bool
    torch_version: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Registered task name, e.g. GSM8K")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model used for training.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for adapter outputs and the stage report.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Registered runs use 1000 steps. Phase 0 pilots may override downward.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override task-specific generation cap. Defaults to the task registry value.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--scope",
        choices=["registered", "pilot_only", "exploratory"],
        default="registered",
        help="Report scope label for the resulting stage report.",
    )
    parser.add_argument(
        "--print-task-config",
        action="store_true",
        help="Print the resolved task config and exit before loading ML dependencies.",
    )
    parser.add_argument(
        "--smoke-test-prompt",
        action="store_true",
        help="Render one synthetic formatted example for the task and exit.",
    )
    return parser.parse_args()


def extract_final_number(text: object) -> str | None:
    text_str = str(text)
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text_str.replace(",", ""))
    return matches[-1] if matches else None


def extract_final_option(text: str) -> str | None:
    matches = re.findall(r"\b([A-E])\b", text.upper())
    return matches[-1] if matches else None


def extract_code_block(text: str) -> str:
    fenced_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_blocks:
        return fenced_blocks[-1]
    return text


def normalize_code_text(text: str) -> str:
    code = extract_code_block(text)
    lines = []
    for raw_line in code.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(raw_line.rstrip())
    return "\n".join(lines)


def final_number_exact_match(completions: list[str], answer: list[str], **_: object) -> list[float]:
    n_prompts = len(answer)
    assert len(completions) % n_prompts == 0, (
        f"len(completions)={len(completions)} not divisible by n_prompts={n_prompts}"
    )
    n_gen = len(completions) // n_prompts
    rewards = []
    for i, completion in enumerate(completions):
        pred = extract_final_number(completion)
        gold = extract_final_number(answer[i // n_gen])
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards


def mcq_label_exact_match(completions: list[str], answer: list[str], **_: object) -> list[float]:
    n_prompts = len(answer)
    assert len(completions) % n_prompts == 0, (
        f"len(completions)={len(completions)} not divisible by n_prompts={n_prompts}"
    )
    n_gen = len(completions) // n_prompts
    rewards = []
    for i, completion in enumerate(completions):
        pred = extract_final_option(completion)
        gold = extract_final_option(answer[i // n_gen])
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards


def code_exact_match(completions: list[str], answer: list[str], **_: object) -> list[float]:
    n_prompts = len(answer)
    assert len(completions) % n_prompts == 0, (
        f"len(completions)={len(completions)} not divisible by n_prompts={n_prompts}"
    )
    n_gen = len(completions) // n_prompts
    rewards = []
    for i, completion in enumerate(completions):
        pred = normalize_code_text(completion)
        gold = normalize_code_text(answer[i // n_gen])
        rewards.append(1.0 if pred == gold and pred else 0.0)
    return rewards


def _ensure_list_of_strings(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _run_mbpp_tests(candidate_code: str, test_setup_code: object, tests: object) -> bool:
    namespace: dict[str, object] = {}
    setup = str(test_setup_code).strip()
    if setup:
        exec(setup, namespace)

    exec(candidate_code, namespace)
    for test in _ensure_list_of_strings(tests):
        exec(test, namespace)
    return True


_SUBPROCESS_TEST_RUNNER = r"""
import json
import sys

payload = json.loads(sys.argv[1])
namespace = {}
setup = str(payload["setup"]).strip()
if setup:
    exec(setup, namespace)
exec(payload["code"], namespace)
for test in payload["tests"]:
    exec(test, namespace)
"""


def _run_tests_in_subprocess(
    candidate_code: str,
    test_setup_code: object,
    tests: list[str],
    *,
    timeout_sec: float = 3.0,
) -> tuple[bool, bool]:
    payload = json.dumps(
        {
            "code": candidate_code,
            "setup": str(test_setup_code),
            "tests": tests,
        },
        ensure_ascii=False,
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_TEST_RUNNER, payload],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, True
    return completed.returncode == 0, False


MBPP_REWARD_LOG_EVERY = 32
MBPP_REWARD_LOG_SECS = 60.0
_MBPP_REWARD_STATE = {
    "seen": 0,
    "passed": 0,
    "timeouts": 0,
    "last_log_ts": 0.0,
}


def _log_mbpp_reward_progress(batch_total: int, batch_passed: int, batch_timeouts: int, batch_elapsed_sec: float) -> None:
    state = _MBPP_REWARD_STATE
    state["seen"] += batch_total
    state["passed"] += batch_passed
    state["timeouts"] += batch_timeouts
    now = time.time()
    should_log = (
        state["seen"] % MBPP_REWARD_LOG_EVERY == 0
        or now - state["last_log_ts"] >= MBPP_REWARD_LOG_SECS
    )
    if not should_log:
        return
    pass_rate = state["passed"] / state["seen"] if state["seen"] else 0.0
    print(
        "[MBPP-REWARD] "
        f"seen={state['seen']} passed={state['passed']} pass_rate={pass_rate:.3f} "
        f"timeouts={state['timeouts']} batch_n={batch_total} batch_sec={batch_elapsed_sec:.2f}",
        flush=True,
    )
    state["last_log_ts"] = now


def _resolve_humaneval_candidate(prompt: object, completion: object, entry_point: object) -> str:
    prompt_text = str(prompt)
    candidate_code = normalize_code_text(str(completion))
    entry_name = str(entry_point).strip()
    if not candidate_code:
        return ""
    if entry_name and re.search(rf"\bdef\s+{re.escape(entry_name)}\s*\(", candidate_code):
        return candidate_code
    return f"{prompt_text}{candidate_code}"


def _run_humaneval_check(candidate_program: str, test_code: object, entry_point: object) -> bool:
    passed, timed_out = _run_humaneval_check_in_subprocess(candidate_program, test_code, entry_point)
    if timed_out:
        raise TimeoutError("HumanEval test execution timed out.")
    return passed


_HUMANEVAL_SUBPROCESS_RUNNER = r"""
import json
import sys

payload = json.loads(sys.argv[1])
namespace = {}
exec(payload["candidate_program"], namespace)
exec(str(payload["test_code"]), namespace)

entry_name = str(payload["entry_point"]).strip()
if not entry_name or entry_name not in namespace:
    raise NameError(f"entry_point '{entry_name}' was not defined by candidate program.")
check_fn = namespace.get("check")
if not callable(check_fn):
    raise NameError("HumanEval test block did not define callable 'check'.")
check_fn(namespace[entry_name])
"""


def _run_humaneval_check_in_subprocess(
    candidate_program: str,
    test_code: object,
    entry_point: object,
    *,
    timeout_sec: float = 3.0,
) -> tuple[bool, bool]:
    payload = json.dumps(
        {
            "candidate_program": candidate_program,
            "test_code": str(test_code),
            "entry_point": str(entry_point),
        },
        ensure_ascii=False,
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _HUMANEVAL_SUBPROCESS_RUNNER, payload],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, True
    return completed.returncode == 0, False


HUMANEVAL_REWARD_LOG_EVERY = 32
HUMANEVAL_REWARD_LOG_SECS = 60.0
_HUMANEVAL_REWARD_STATE = {
    "seen": 0,
    "passed": 0,
    "timeouts": 0,
    "last_log_ts": 0.0,
}


def _log_humaneval_reward_progress(
    batch_total: int,
    batch_passed: int,
    batch_timeouts: int,
    batch_elapsed_sec: float,
) -> None:
    state = _HUMANEVAL_REWARD_STATE
    state["seen"] += batch_total
    state["passed"] += batch_passed
    state["timeouts"] += batch_timeouts
    now = time.time()
    should_log = (
        state["seen"] % HUMANEVAL_REWARD_LOG_EVERY == 0
        or now - state["last_log_ts"] >= HUMANEVAL_REWARD_LOG_SECS
    )
    if not should_log:
        return
    pass_rate = state["passed"] / state["seen"] if state["seen"] else 0.0
    print(
        "[HUMANEVAL-REWARD] "
        f"seen={state['seen']} passed={state['passed']} pass_rate={pass_rate:.3f} "
        f"timeouts={state['timeouts']} batch_n={batch_total} batch_sec={batch_elapsed_sec:.2f}",
        flush=True,
    )
    state["last_log_ts"] = now


def mbpp_test_pass(
    completions: list[str],
    answer: list[str],
    test_list: list[object] | None = None,
    test_setup_code: list[object] | None = None,
    challenge_test_list: list[object] | None = None,
    **_: object,
) -> list[float]:
    n_prompts = len(answer)
    assert len(completions) % n_prompts == 0, (
        f"len(completions)={len(completions)} not divisible by n_prompts={n_prompts}"
    )
    n_gen = len(completions) // n_prompts
    rewards = []
    batch_passed = 0
    batch_timeouts = 0
    batch_start = time.perf_counter()
    for i, completion in enumerate(completions):
        prompt_idx = i // n_gen
        candidate_code = normalize_code_text(completion)
        tests = []
        if test_list is not None:
            tests.extend(_ensure_list_of_strings(test_list[prompt_idx]))
        if challenge_test_list is not None:
            tests.extend(_ensure_list_of_strings(challenge_test_list[prompt_idx]))
        if not tests:
            rewards.append(0.0)
            continue
        passed, timed_out = _run_tests_in_subprocess(
            candidate_code,
            test_setup_code[prompt_idx] if test_setup_code is not None else "",
            tests,
            timeout_sec=3.0,
        )
        if passed:
            batch_passed += 1
            rewards.append(1.0)
        else:
            if timed_out:
                batch_timeouts += 1
            rewards.append(0.0)
    _log_mbpp_reward_progress(
        batch_total=len(completions),
        batch_passed=batch_passed,
        batch_timeouts=batch_timeouts,
        batch_elapsed_sec=time.perf_counter() - batch_start,
    )
    return rewards


def humaneval_test_pass(
    completions: list[str],
    answer: list[str],
    prompt: list[object] | None = None,
    test: list[object] | None = None,
    entry_point: list[object] | None = None,
    **_: object,
) -> list[float]:
    del answer
    if prompt is None or test is None or entry_point is None:
        raise ValueError("HumanEval reward requires prompt, test, and entry_point fields.")
    n_prompts = len(prompt)
    assert len(completions) % n_prompts == 0, (
        f"len(completions)={len(completions)} not divisible by n_prompts={n_prompts}"
    )
    n_gen = len(completions) // n_prompts
    rewards = []
    batch_passed = 0
    batch_timeouts = 0
    batch_start = time.perf_counter()
    for i, completion in enumerate(completions):
        prompt_idx = i // n_gen
        candidate_program = _resolve_humaneval_candidate(
            prompt[prompt_idx], completion, entry_point[prompt_idx]
        )
        if not candidate_program:
            rewards.append(0.0)
            continue
        passed, timed_out = _run_humaneval_check_in_subprocess(
            candidate_program,
            test[prompt_idx],
            entry_point[prompt_idx],
            timeout_sec=3.0,
        )
        if passed:
            batch_passed += 1
            rewards.append(1.0)
        else:
            if timed_out:
                batch_timeouts += 1
            rewards.append(0.0)
    _log_humaneval_reward_progress(
        batch_total=len(completions),
        batch_passed=batch_passed,
        batch_timeouts=batch_timeouts,
        batch_elapsed_sec=time.perf_counter() - batch_start,
    )
    return rewards


REWARD_BUILDERS = {
    "final_number_exact_match": final_number_exact_match,
    "mcq_label_exact_match": mcq_label_exact_match,
    "code_exact_match": code_exact_match,
    "mbpp_test_pass": mbpp_test_pass,
    "humaneval_test_pass": humaneval_test_pass,
}


def render_mcq_options(labels: list[object], texts: list[object]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


def format_commonsenseqa_prompt(example: dict[str, object]) -> str:
    question = str(example["question"])
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    rendered = render_mcq_options(labels, texts)
    return f"{question}\n\nOptions:\n{rendered}"


def format_hellaswag_prompt(example: dict[str, object]) -> str:
    activity_label = str(example.get("activity_label", "")).strip()
    ctx_a = str(example["ctx_a"]).strip()
    ctx_b = str(example.get("ctx_b", "")).strip()
    context = f"{ctx_a} {ctx_b}".strip()
    endings = list(example["endings"])
    rendered = render_mcq_options(["A", "B", "C", "D"], endings)
    prefix = f"Activity: {activity_label}\n\n" if activity_label else ""
    return f"{prefix}Context: {context}\n\nChoose the best continuation:\n{rendered}"


def format_winogrande_prompt(example: dict[str, object]) -> str:
    sentence = str(example["sentence"]).strip()
    option1 = str(example["option1"]).strip()
    option2 = str(example["option2"]).strip()
    rendered = render_mcq_options(["A", "B"], [option1, option2])
    return f"{sentence}\n\nChoose the better fill for the blank:\n{rendered}"


CHOICE_FORMATTERS = {
    "commonsenseqa": format_commonsenseqa_prompt,
    "hellaswag": format_hellaswag_prompt,
    "winogrande": format_winogrande_prompt,
}


def normalize_mcq_answer(
    raw_answer: object,
    *,
    choice_labels: list[str] | None = None,
    answer_index_base: int = 0,
) -> str:
    answer_text = str(raw_answer).strip()
    if choice_labels:
        if answer_text.isdigit():
            index = int(answer_text) - answer_index_base
            if 0 <= index < len(choice_labels):
                return choice_labels[index]
        if isinstance(raw_answer, int):
            index = raw_answer - answer_index_base
            if 0 <= index < len(choice_labels):
                return choice_labels[index]
    return answer_text.upper()


def format_mcq_answer(task_config: dict[str, object], example: dict[str, object]) -> str:
    answer_field = task_config["fields"]["label"]
    raw_answer = example[answer_field]
    return normalize_mcq_answer(
        raw_answer,
        choice_labels=task_config.get("choice_labels"),
        answer_index_base=int(task_config.get("answer_index_base", 0)),
    )


def build_formatted_example(task_config: dict[str, object], example: dict[str, object]) -> dict[str, str]:
    if task_config["prompt_style"] == "chat_mcq":
        choice_format = task_config.get("choice_format")
        if choice_format not in CHOICE_FORMATTERS:
            raise NotImplementedError(f"Choice formatter '{choice_format}' is not implemented.")
        user_prompt = CHOICE_FORMATTERS[choice_format](example)
        answer_value = format_mcq_answer(task_config, example)
    else:
        prompt_field = task_config["fields"]["prompt"]
        answer_field = task_config["fields"]["answer"]
        user_prompt = str(example[prompt_field])
        answer_value = str(example[answer_field])
        test_list_field = task_config.get("fields", {}).get("test_list")
        if test_list_field and test_list_field in example:
            tests = _ensure_list_of_strings(example.get(test_list_field))
            if tests:
                user_prompt = (
                    f"{user_prompt}\n\n"
                    "Your solution must satisfy this test:\n"
                    f"{tests[0]}"
                )
    payload = {"user_prompt": user_prompt, "answer": answer_value}
    for key, field_name in task_config.get("fields", {}).items():
        if key in {"prompt", "answer"}:
            continue
        if field_name in example:
            payload[key] = example[field_name]
    return payload


def synthetic_example(task_name: str) -> dict[str, object]:
    if task_name == "GSM8K":
        return {
            "question": "If Mina has 3 apples and buys 2 more, how many apples does she have?",
            "answer": "She has 5 apples. The answer is 5.",
        }
    if task_name == "MATH":
        return {
            "problem": "Solve for x: x + 7 = 10.",
            "extracted_solution": 3,
        }
    if task_name == "AIME":
        return {
            "problem": "Find the sum of the digits of 99.",
            "answer": "18",
        }
    if task_name == "AMC":
        return {
            "task": "Evaluate 2 + 3.",
            "answer": 5,
        }
    if task_name == "MATH500":
        return {
            "problem": "Solve for x: x + 7 = 10.",
            "answer": "3",
        }
    if task_name == "CommonsenseQA":
        return {
            "question": "Where would you usually keep milk cold?",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": ["desk drawer", "refrigerator", "bookshelf", "shoe box", "sink"],
            },
            "answerKey": "B",
        }
    if task_name == "ARC-Challenge":
        return {
            "question": "What do plants need for photosynthesis?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["oxygen and sugar", "sunlight and water", "soil and rocks", "wind and rain"],
            },
            "answerKey": "B",
        }
    if task_name == "ARC-Easy":
        return {
            "question": "What is the main source of energy for Earth?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["the Moon", "the Sun", "the ocean", "the soil"],
            },
            "answerKey": "B",
        }
    if task_name == "MBPP":
        return {
            "text": "Write a Python function `add_one` that returns the input plus one.",
            "code": "def add_one(x):\n    return x + 1",
            "test_list": [
                "assert add_one(3) == 4",
                "assert add_one(-1) == 0",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        }
    if task_name == "HumanEval":
        return {
            "task_id": "HumanEval/0",
            "prompt": (
                "def add_one(x):\n"
                "    \"\"\"Return x plus one.\"\"\"\n"
            ),
            "canonical_solution": "    return x + 1\n",
            "test": (
                "def check(candidate):\n"
                "    assert candidate(3) == 4\n"
                "    assert candidate(-1) == 0\n"
            ),
            "entry_point": "add_one",
        }
    if task_name == "HellaSwag":
        return {
            "activity_label": "Making tea",
            "ctx_a": "The person boils water in a kettle.",
            "ctx_b": "Then",
            "endings": [
                "they pour the water into a teacup and add a tea bag.",
                "they place the kettle inside the refrigerator.",
                "they hand the kettle to a soccer player.",
                "they turn off the stove and leave the room.",
            ],
            "label": 0,
        }
    if task_name == "WinoGrande":
        return {
            "sentence": "Jordan handed the book to Taylor because _ had finished reading it.",
            "option1": "Jordan",
            "option2": "Taylor",
            "answer": "1",
        }
    raise NotImplementedError(f"No synthetic example defined for task '{task_name}'.")


def detect_hardware() -> tuple[HardwareMetadata, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. GRPO training requires CUDA.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_supported = torch.cuda.is_bf16_supported()
    meta = HardwareMetadata(
        gpu_model=gpu_name,
        vram_gb=round(vram_gb, 1),
        bf16_supported=bf16_supported,
        torch_version=torch.__version__,
    )
    return meta, torch


def main() -> None:
    args = parse_args()
    task_config = get_task_config(args.task)
    if task_config["status"] != "implemented":
        raise NotImplementedError(
            f"Task '{args.task}' is registered but not implemented yet. "
            f"See {TASK_REGISTRY_PATH} for the current status."
        )
    if task_config["reward"] not in REWARD_BUILDERS:
        raise NotImplementedError(f"Reward '{task_config['reward']}' is not implemented.")

    resolved_max_new_tokens = int(task_config.get("default_max_new_tokens", 256))
    if args.max_new_tokens is not None:
        resolved_max_new_tokens = args.max_new_tokens

    if args.print_task_config:
        print(json.dumps(task_config, indent=2))
        return

    if args.smoke_test_prompt:
        payload = build_formatted_example(task_config, synthetic_example(args.task))
        smoke = {
            "task": args.task,
            "prompt_style": task_config["prompt_style"],
            "reward": task_config["reward"],
            "formatted_example": payload,
        }
        print(json.dumps(smoke, indent=2))
        return

    if not args.output_dir:
        raise ValueError("--output-dir is required for real training runs.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware, torch = detect_hardware()
    model_dtype = torch.bfloat16 if hardware.bf16_supported else torch.float16

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    loader = task_config["dataset_loader"]
    dataset = load_dataset(loader["path"], loader.get("name"))
    train_data = dataset[loader.get("split", "train")]

    system_prompt = PROMPT_TEMPLATES[task_config["prompt_style"]]
    def format_example(example: dict[str, str]) -> dict[str, str]:
        payload = build_formatted_example(task_config, example)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload["user_prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": payload["answer"]}

    train_data = train_data.map(format_example)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=resolved_max_new_tokens,
        learning_rate=args.learning_rate,
        bf16=hardware.bf16_supported,
        fp16=not hardware.bf16_supported,
        logging_steps=10,
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_data,
        reward_funcs=REWARD_BUILDERS[task_config["reward"]],
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    total_time = time.time() - t0
    step_time_sec = total_time / args.max_steps

    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    report = make_stage_report(
        stage="training",
        phase=0 if args.scope == "pilot_only" else 1,
        task=args.task,
        model=args.model_id,
        method="GRPO",
        scope=args.scope,
        summary={
            "description": f"Task-aware GRPO run for {args.task}.",
            "task_status": task_config["status"],
        },
        config={
            "task_registry_path": TASK_REGISTRY_PATH,
            "dataset_loader": loader,
            "prompt_style": task_config["prompt_style"],
            "reward": task_config["reward"],
            "choice_format": task_config.get("choice_format"),
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_generations": args.num_generations,
            "max_new_tokens": resolved_max_new_tokens,
            "learning_rate": args.learning_rate,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_target_modules": LORA_TARGET_MODULES,
        },
        metrics={
            "train_examples": len(train_data),
            "total_time_sec": round(total_time, 1),
            "per_step_time_sec": round(step_time_sec, 2),
        },
        artifacts={
            "adapter_path": adapter_path,
        },
        runtime={
            "hardware": asdict(hardware),
            "precision": "bf16" if hardware.bf16_supported else "fp16",
        },
        validation={
            "model_loaded": True,
            "training_completed": True,
            "adapter_saved": True,
        },
        notes=[
            f"Task config loaded from {TASK_REGISTRY_PATH}.",
        ],
    )
    write_report(output_dir / "training_run_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
