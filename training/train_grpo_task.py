#!/usr/bin/env python3
"""Task-aware GRPO training entrypoint for ASCENT-G."""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
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


def extract_final_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_final_option(text: str) -> str | None:
    matches = re.findall(r"\b([A-E])\b", text.upper())
    return matches[-1] if matches else None


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


REWARD_BUILDERS = {
    "final_number_exact_match": final_number_exact_match,
    "mcq_label_exact_match": mcq_label_exact_match,
}


def format_commonsenseqa_prompt(example: dict[str, object]) -> str:
    question = str(example["question"])
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"{question}\n\nOptions:\n{rendered}"


CHOICE_FORMATTERS = {
    "commonsenseqa": format_commonsenseqa_prompt,
}


def build_formatted_example(task_config: dict[str, object], example: dict[str, object]) -> dict[str, str]:
    if task_config["prompt_style"] == "chat_mcq":
        choice_format = task_config.get("choice_format")
        if choice_format not in CHOICE_FORMATTERS:
            raise NotImplementedError(f"Choice formatter '{choice_format}' is not implemented.")
        user_prompt = CHOICE_FORMATTERS[choice_format](example)
        answer_value = str(example[task_config["fields"]["label"]])
    else:
        prompt_field = task_config["fields"]["prompt"]
        answer_field = task_config["fields"]["answer"]
        user_prompt = str(example[prompt_field])
        answer_value = str(example[answer_field])
    return {"user_prompt": user_prompt, "answer": answer_value}


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
    if task_name == "CommonsenseQA":
        return {
            "question": "Where would you usually keep milk cold?",
            "choices": {
                "label": ["A", "B", "C", "D", "E"],
                "text": ["desk drawer", "refrigerator", "bookshelf", "shoe box", "sink"],
            },
            "answerKey": "B",
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
        max_completion_length=args.max_new_tokens,
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
            "max_new_tokens": args.max_new_tokens,
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
