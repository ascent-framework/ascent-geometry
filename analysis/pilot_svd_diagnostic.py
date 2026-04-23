#!/usr/bin/env python3
"""Run the Phase 0 pilot SVD diagnostic on dense effective LoRA deltas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.reporting import make_stage_report, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="GSM8K", help="Registered task name for report metadata.")
    parser.add_argument("--model-id", required=True, help="Base model used with the adapter.")
    parser.add_argument("--adapter-path", required=True, help="Path to the saved PEFT adapter.")
    parser.add_argument(
        "--output-path",
        help="Optional JSON path for layerwise pilot SVD results.",
    )
    return parser.parse_args()


def compute_r90(singular_values: np.ndarray) -> int:
    total = (singular_values ** 2).sum()
    cumvar = np.cumsum(singular_values ** 2) / total
    return int(np.searchsorted(cumvar, 0.90)) + 1


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    results: list[dict[str, object]] = []
    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        a_weight = module.lora_A["default"].weight
        b_weight = module.lora_B["default"].weight
        scaling = module.scaling["default"]
        delta = (scaling * (b_weight @ a_weight)).detach().float().cpu().numpy()
        _, singular_values, _ = np.linalg.svd(delta, full_matrices=False)
        results.append(
            {
                "layer": name,
                "shape": list(delta.shape),
                "r90": compute_r90(singular_values),
                "s_max": float(singular_values[0]),
            }
        )

    payload = make_stage_report(
        stage="analysis",
        phase=0,
        task=args.task,
        model=args.model_id,
        method="SVD",
        scope="pilot_only",
        summary={
            "description": "Layerwise pilot SVD diagnostic on dense effective LoRA deltas.",
            "registered_primary_analysis_run": False,
        },
        config={
            "input_object": "dense_effective_delta",
        },
        metrics={
            "num_layers_analyzed": len(results),
            "results": results,
        },
        artifacts={},
        runtime={},
        validation={
            "svd_diagnostic_ran": bool(results),
        },
        notes=[
            "This is a pipeline diagnostic, not the registered H1a/H1b task-matrix analysis.",
        ],
    )

    if args.output_path:
        write_report(args.output_path, payload)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
