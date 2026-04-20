#!/usr/bin/env python3
"""Registered H1a/H1b analysis on a multi-task update-vector matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.reporting import make_stage_report, write_report


MIN_REGISTERED_TASKS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector",
        action="append",
        required=True,
        help="Task vector in TASK=/path/to/update_vector.npy form. Repeat for each task.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for H1a confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model identifier shared by the input task vectors.",
    )
    return parser.parse_args()


def parse_task_vector_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected TASK=PATH format, got: {raw}")
    task, path_str = raw.split("=", 1)
    task = task.strip()
    path = Path(path_str).expanduser()
    if not task:
        raise ValueError(f"Task name is empty in argument: {raw}")
    return task, path


def compute_r90(singular_values: np.ndarray) -> int:
    energy = singular_values**2
    total = energy.sum()
    if total <= 0:
        raise ValueError("Singular values have zero total energy.")
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, 0.90)) + 1


def compute_rho(r90: int, num_tasks: int, effective_dim: int) -> float:
    return float(r90 / min(num_tasks, math.sqrt(effective_dim)))


def h1a_decision(rho: float, ci_low: float, ci_high: float) -> str:
    if rho < 0.30 and ci_high < 0.35:
        return "strong_pass"
    if 0.30 <= rho < 0.50:
        return "marginal_pass"
    if rho >= 0.70 and ci_low >= 0.65:
        return "fail"
    if 0.50 <= rho < 0.70:
        return "inconclusive"
    return "inconclusive"


def pairwise_abs_cosines(matrix: np.ndarray) -> list[float]:
    cosines: list[float] = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            lhs = matrix[:, left]
            rhs = matrix[:, right]
            denom = np.linalg.norm(lhs) * np.linalg.norm(rhs)
            if denom == 0:
                cosines.append(0.0)
            else:
                cosines.append(float(abs(np.dot(lhs, rhs) / denom)))
    return cosines


def h1b_decision(mean_abs_cos: float, max_abs_cos: float) -> str:
    if mean_abs_cos < 0.30 and max_abs_cos < 0.60:
        return "pass"
    if mean_abs_cos > 0.50:
        return "fail"
    return "inconclusive"


def load_vectors(entries: list[tuple[str, Path]]) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    task_names: list[str] = []
    vectors: list[np.ndarray] = []
    magnitudes: list[dict[str, object]] = []

    expected_dim: int | None = None
    for task, path in entries:
        vector = np.load(path)
        if vector.ndim != 1:
            raise ValueError(f"Expected a 1-D vector for {task}, got shape {vector.shape}")
        if expected_dim is None:
            expected_dim = int(vector.shape[0])
        elif vector.shape[0] != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch for {task}: expected {expected_dim}, got {vector.shape[0]}"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError(f"Vector norm is zero for {task}")

        task_names.append(task)
        vectors.append(vector / norm)
        magnitudes.append(
            {
                "task": task,
                "path": str(path),
                "norm": norm,
                "numel": int(vector.shape[0]),
            }
        )

    matrix = np.column_stack(vectors)
    return task_names, matrix, magnitudes


def main() -> None:
    args = parse_args()
    entries = [parse_task_vector_arg(raw) for raw in args.vector]
    task_names, normalized_matrix, magnitudes = load_vectors(entries)

    effective_dim = int(normalized_matrix.shape[0])
    num_tasks = int(normalized_matrix.shape[1])
    u, singular_values, _ = np.linalg.svd(normalized_matrix, full_matrices=False)
    r90 = compute_r90(singular_values)
    rho = compute_rho(r90, num_tasks, effective_dim)

    rng = np.random.default_rng(args.seed)
    bootstrap_rhos: list[float] = []
    bootstrap_r90s: list[int] = []
    for _ in range(args.bootstrap_samples):
        sampled_indices = rng.integers(0, num_tasks, size=num_tasks)
        sampled = normalized_matrix[:, sampled_indices]
        _, sampled_singular_values, _ = np.linalg.svd(sampled, full_matrices=False)
        sampled_r90 = compute_r90(sampled_singular_values)
        bootstrap_r90s.append(sampled_r90)
        bootstrap_rhos.append(compute_rho(sampled_r90, num_tasks, effective_dim))

    ci_low, ci_high = np.quantile(bootstrap_rhos, [0.025, 0.975])
    h1a_tier = h1a_decision(rho, float(ci_low), float(ci_high))

    h1b_result: dict[str, object] = {
        "computed": False,
        "decision": None,
    }
    if h1a_tier != "fail":
        basis = u[:, :r90]
        projected = basis @ (basis.T @ normalized_matrix)
        abs_cosines = pairwise_abs_cosines(projected)
        mean_abs_cos = float(np.mean(abs_cosines)) if abs_cosines else 0.0
        max_abs_cos = float(np.max(abs_cosines)) if abs_cosines else 0.0
        h1b_result = {
            "computed": True,
            "decision": h1b_decision(mean_abs_cos, max_abs_cos),
            "mean_abs_cos": mean_abs_cos,
            "max_abs_cos": max_abs_cos,
            "pair_count": len(abs_cosines),
        }

    registered_ready = num_tasks >= MIN_REGISTERED_TASKS
    notes = [
        "Input vectors are normalized per preregistration v1.3 for direction analysis.",
        "Magnitude metadata is retained separately for B_adaptive follow-up analysis.",
    ]
    if not registered_ready:
        notes.append(
            f"Registered H1a evaluation requires at least {MIN_REGISTERED_TASKS} tasks; current N={num_tasks}."
        )

    report = make_stage_report(
        stage="analysis",
        phase=1,
        task="MULTI_TASK",
        model=args.model_id,
        method="SVD+cosine",
        scope="registered",
        summary={
            "description": "Registered H1a/H1b analysis on normalized task update vectors.",
            "tasks": task_names,
            "registered_ready": registered_ready,
        },
        config={
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "minimum_registered_tasks": MIN_REGISTERED_TASKS,
            "input_vectors": [{"task": task, "path": str(path)} for task, path in entries],
        },
        metrics={
            "num_tasks": num_tasks,
            "effective_dim": effective_dim,
            "singular_values": singular_values.tolist(),
            "r90": r90,
            "rho": rho,
            "rho_ci_95": [float(ci_low), float(ci_high)],
            "bootstrap_r90_mean": float(np.mean(bootstrap_r90s)),
            "bootstrap_rho_mean": float(np.mean(bootstrap_rhos)),
            "h1a_decision": h1a_tier,
            "h1b": h1b_result,
            "task_magnitudes": magnitudes,
        },
        artifacts={
            "report_path": args.output_path,
        },
        runtime={},
        validation={
            "all_vectors_same_dimension": True,
            "all_vectors_nonzero": True,
            "registered_minimum_tasks_met": registered_ready,
        },
        notes=notes,
    )
    write_report(args.output_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
