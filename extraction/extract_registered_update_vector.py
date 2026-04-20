#!/usr/bin/env python3
"""Extract the registered ASCENT-G update vector from a LoRA adapter.

Registered object in preregistration v1.3:
    concat(Delta W_A, Delta W_B) over all registered LoRA layers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.reporting import make_stage_report, write_report


REGISTERED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="Base model used with the adapter.")
    parser.add_argument("--adapter-path", required=True, help="Path to the saved PEFT adapter.")
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination .npy path for the registered update vector.",
    )
    return parser.parse_args()


def extract_registered_update_vector(peft_model: object) -> tuple[np.ndarray, list[dict[str, object]]]:
    deltas: list[np.ndarray] = []
    layer_meta: list[dict[str, object]] = []
    seen_targets: set[str] = set()

    for name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        a_weight = module.lora_A["default"].weight.detach().float().cpu().numpy()
        b_weight = module.lora_B["default"].weight.detach().float().cpu().numpy()

        deltas.append(a_weight.flatten())
        deltas.append(b_weight.flatten())
        layer_meta.append(
            {
                "name": name,
                "a_shape": list(a_weight.shape),
                "b_shape": list(b_weight.shape),
                "a_numel": int(a_weight.size),
                "b_numel": int(b_weight.size),
                "a_norm": float(np.linalg.norm(a_weight)),
                "b_norm": float(np.linalg.norm(b_weight)),
            }
        )

        for target in REGISTERED_TARGETS:
            if name.endswith(target):
                seen_targets.add(target)

    if not deltas:
        raise ValueError("No LoRA layers found. Check the adapter path and PEFT config.")

    missing = REGISTERED_TARGETS - seen_targets
    if missing:
        raise ValueError(f"Missing registered target modules: {sorted(missing)}")

    return np.concatenate(deltas), layer_meta


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    update_vector, layer_meta = extract_registered_update_vector(peft_model)
    np.save(output_path, update_vector)

    checksum = hashlib.sha256(update_vector.tobytes()).hexdigest()
    provenance = {
        "vector_path": str(output_path),
        "object_type": "registered_concat_lora_A_B",
        "sha256": checksum,
        "shape": list(update_vector.shape),
        "norm": float(np.linalg.norm(update_vector)),
        "registered_targets": sorted(REGISTERED_TARGETS),
        "layers": layer_meta,
    }
    provenance_path = output_path.with_name(f"{output_path.stem}_provenance.json")
    write_report(provenance_path, provenance)

    stage_report = make_stage_report(
        stage="extraction",
        phase=0,
        task="GSM8K",
        model=args.model_id,
        method="LoRA",
        scope="registered",
        summary={
            "description": "Registered update-vector extraction for ASCENT-G.",
            "object_type": "registered_concat_lora_A_B",
        },
        config={
            "registered_targets": sorted(REGISTERED_TARGETS),
        },
        metrics={
            "vector_norm": float(np.linalg.norm(update_vector)),
            "vector_numel": int(update_vector.size),
        },
        artifacts={
            "vector_path": output_path,
            "provenance_path": provenance_path,
        },
        runtime={
            "device_dtype": str(model_dtype).replace("torch.", ""),
        },
        validation={
            "registered_targets_all_covered": True,
            "vector_non_degenerate": bool(np.linalg.norm(update_vector) > 0),
            "checksum_sha256": checksum,
        },
        notes=[
            "This report tracks the preregistered concat(Delta W_A, Delta W_B) object.",
        ],
    )
    report_path = output_path.with_name(f"{output_path.stem}_report.json")
    write_report(report_path, stage_report)

    print(json.dumps(stage_report, indent=2))


if __name__ == "__main__":
    main()
