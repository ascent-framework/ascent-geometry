#!/usr/bin/env python3
"""Phase 0 GRPO pilot runner for GSM8K on Qwen2.5-1.5B-Instruct.

This script is intentionally operational rather than fully general. It mirrors
the current Phase 0 notebook but exposes the run through a CLI so the training
step can be reused outside Kaggle notebooks.
"""

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


SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Think step by step, then end your answer with: The answer is <number>."
)

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class HardwareMetadata:
    gpu_model: str
    vram_gb: float
    bf16_supported: bool
    torch_version: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model used for the Phase 0 pilot.",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/gsm8k-qwen2.5-1.5b-phase0",
        help="Directory for adapter outputs and the run report.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Pilot-only override. Registered full runs should use 1000 steps.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Registered learning rate from preregistration v1.3.",
    )
    return parser.parse_args()


def extract_final_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def correctness_reward(completions: list[str], answer: list[str], **_: object) -> list[float]:
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


def detect_hardware() -> tuple[HardwareMetadata, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Phase 0 training requires CUDA.")

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

    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]

    def format_example(example: dict[str, str]) -> dict[str, str]:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": example["answer"]}

    train_data = train_data.map(format_example)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
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
        reward_funcs=correctness_reward,
        tokenizer=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    total_time = time.time() - t0
    step_time_sec = total_time / args.max_steps

    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    run_report = make_stage_report(
        stage="training",
        phase=0,
        task="GSM8K",
        model=args.model_id,
        method="GRPO",
        scope="pilot_only",
        summary={
            "description": "Phase 0 GRPO pilot training run for GSM8K.",
            "registered_primary_analysis_run": False,
        },
        config={
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_generations": args.num_generations,
            "max_new_tokens": args.max_new_tokens,
            "learning_rate": args.learning_rate,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_target_modules": LORA_TARGET_MODULES,
            "phase0_pilot_override": {
                "max_steps": args.max_steps,
            },
        },
        metrics={
            "total_time_sec": round(total_time, 1),
            "per_step_time_sec": round(step_time_sec, 2),
        },
        artifacts={
            "adapter_path": adapter_path,
        },
        runtime={
            "hardware": asdict(hardware),
            "precision": "bf16" if hardware.bf16_supported else "fp16",
            "dataset": "openai/gsm8k",
        },
        validation={
            "model_loaded": True,
            "training_completed": True,
            "adapter_saved": True,
        },
        notes=[
            "Phase 0 uses a reduced step count for pipeline validation only.",
        ],
    )

    write_report(output_dir / "training_run_report.json", run_report)

    print(json.dumps(run_report, indent=2))


if __name__ == "__main__":
    main()
