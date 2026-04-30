#!/usr/bin/env python3
"""Plan B: H1a analysis on predefined task clusters."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vector", action="append", required=True,
                   help="TASK=/path/to/update_vector.npy  (repeat per task)")
    p.add_argument("--output-path", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_vector_arg(raw: str) -> tuple[str, Path]:
    task, path_str = raw.split("=", 1)
    return task.strip(), Path(path_str).expanduser()


CLUSTERS: dict[str, list[str]] = {
    # 쌍 (pair) 군집
    "pair_ARC":    ["ARC-Challenge", "ARC-Easy"],
    "pair_Math":   ["GSM8K", "SVAMP"],
    "pair_Code":   ["HumanEval", "MBPP"],
    # 도메인 군집
    "science_MCQ": ["ARC-Challenge", "ARC-Easy", "OpenbookQA"],
    "MCQ_5":       ["CommonsenseQA", "ARC-Challenge", "ARC-Easy", "OpenbookQA", "HellaSwag"],
    "commonsense": ["CommonsenseQA", "HellaSwag", "WinoGrande"],
    # 전체
    "all_10":      ["CommonsenseQA", "ARC-Challenge", "HellaSwag", "GSM8K",
                    "OpenbookQA", "ARC-Easy", "WinoGrande", "SVAMP", "HumanEval", "MBPP"],
}


def compute_r90(sv: np.ndarray) -> int:
    energy = sv ** 2
    cumul = np.cumsum(energy) / energy.sum()
    return int(np.searchsorted(cumul, 0.90)) + 1


def compute_rho(r90: int, n: int, dim: int) -> float:
    return r90 / min(n, math.sqrt(dim))


def h1a_decision(rho: float, ci_lo: float, ci_hi: float) -> str:
    if rho < 0.30 and ci_hi < 0.35:
        return "strong_pass"
    if 0.30 <= rho < 0.50:
        return "marginal_pass"
    if rho >= 0.70 and ci_lo >= 0.65:
        return "fail"
    return "inconclusive"


def run_h1a(matrix: np.ndarray, rng: np.random.Generator,
            bootstrap_samples: int) -> dict:
    dim, n = matrix.shape
    _, sv, _ = np.linalg.svd(matrix, full_matrices=False)
    r90 = compute_r90(sv)
    rho = compute_rho(r90, n, dim)

    boot_rhos, boot_r90s = [], []
    for _ in range(bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        _, bsv, _ = np.linalg.svd(matrix[:, idx], full_matrices=False)
        br90 = compute_r90(bsv)
        boot_r90s.append(br90)
        boot_rhos.append(compute_rho(br90, n, dim))

    ci_lo = float(np.quantile(boot_rhos, 0.025))
    ci_hi = float(np.quantile(boot_rhos, 0.975))

    return {
        "n_tasks": n,
        "dim": dim,
        "r90": r90,
        "rho": round(rho, 4),
        "rho_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "h1a_decision": h1a_decision(rho, ci_lo, ci_hi),
        "singular_values": sv.tolist(),
    }


def main() -> None:
    args = parse_args()
    entries = dict(parse_vector_arg(r) for r in args.vector)

    # 벡터 로드 및 정규화
    vectors: dict[str, np.ndarray] = {}
    for task, path in entries.items():
        v = np.load(path).astype(np.float32)
        vectors[task] = v / np.linalg.norm(v)

    rng = np.random.default_rng(args.seed)
    cluster_results = {}

    for cluster_name, task_list in CLUSTERS.items():
        missing = [t for t in task_list if t not in vectors]
        if missing:
            cluster_results[cluster_name] = {"error": f"missing tasks: {missing}"}
            continue
        mat = np.column_stack([vectors[t] for t in task_list])
        result = run_h1a(mat, rng, args.bootstrap_samples)
        result["tasks"] = task_list
        cluster_results[cluster_name] = result
        print(f"[{cluster_name}] n={result['n_tasks']}  r90={result['r90']}  "
              f"rho={result['rho']:.4f}  CI=[{result['rho_ci_95'][0]:.3f}, "
              f"{result['rho_ci_95'][1]:.3f}]  → {result['h1a_decision']}")

    report = {
        "analysis": "h1a_cluster_analysis",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "clusters": cluster_results,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
