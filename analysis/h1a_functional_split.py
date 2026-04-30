#!/usr/bin/env python3
"""Plan A: H1a analysis split by functional component (Attention vs MLP)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES  = {"gate_proj", "up_proj", "down_proj"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vector", action="append", required=True,
                   help="TASK=/path/to/update_vector.npy  (repeat per task)")
    p.add_argument("--provenance", required=True,
                   help="Path to any update_vector_provenance.json (same arch for all tasks)")
    p.add_argument("--output-path", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_vector_arg(raw: str) -> tuple[str, Path]:
    task, path_str = raw.split("=", 1)
    return task.strip(), Path(path_str).expanduser()


def build_index_map(provenance_path: str) -> tuple[list[int], list[int]]:
    """Return (attn_indices, mlp_indices) into the concatenated vector."""
    with open(provenance_path) as f:
        prov = json.load(f)
    attn_idx, mlp_idx = [], []
    offset = 0
    for entry in prov["layers"]:
        numel = entry["a_numel"] + entry["b_numel"]
        mod = entry["name"].split(".")[-1]
        idx = list(range(offset, offset + numel))
        if mod in ATTN_MODULES:
            attn_idx.extend(idx)
        elif mod in MLP_MODULES:
            mlp_idx.extend(idx)
        offset += numel
    return attn_idx, mlp_idx


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


def run_h1a(matrix: np.ndarray, label: str, rng: np.random.Generator,
            bootstrap_samples: int) -> dict:
    """Run H1a on a (dim x n_tasks) normalized matrix."""
    dim, n = matrix.shape
    _, sv, _ = np.linalg.svd(matrix, full_matrices=False)
    r90 = compute_r90(sv)
    rho = compute_rho(r90, n, dim)

    boot_rhos = []
    for _ in range(bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        _, bsv, _ = np.linalg.svd(matrix[:, idx], full_matrices=False)
        boot_rhos.append(compute_rho(compute_r90(bsv), n, dim))

    ci_lo, ci_hi = float(np.quantile(boot_rhos, 0.025)), float(np.quantile(boot_rhos, 0.975))
    decision = h1a_decision(rho, ci_lo, ci_hi)

    return {
        "component": label,
        "dim": dim,
        "n_tasks": n,
        "r90": r90,
        "rho": round(rho, 4),
        "rho_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "h1a_decision": decision,
        "singular_values_top10": sv[:10].tolist(),
    }


def main() -> None:
    args = parse_args()
    entries = [parse_vector_arg(r) for r in args.vector]
    attn_idx, mlp_idx = build_index_map(args.provenance)

    task_names = []
    full_vecs, attn_vecs, mlp_vecs = [], [], []

    for task, path in entries:
        v = np.load(path).astype(np.float32)
        task_names.append(task)
        full_vecs.append(v / np.linalg.norm(v))
        va = v[attn_idx]; attn_vecs.append(va / np.linalg.norm(va))
        vm = v[mlp_idx];  mlp_vecs.append(vm / np.linalg.norm(vm))

    rng = np.random.default_rng(args.seed)
    results = []
    for label, vecs in [("full", full_vecs), ("attention", attn_vecs), ("mlp", mlp_vecs)]:
        mat = np.column_stack(vecs)
        results.append(run_h1a(mat, label, rng, args.bootstrap_samples))

    report = {
        "analysis": "h1a_functional_split",
        "tasks": task_names,
        "attn_dim": len(attn_idx),
        "mlp_dim": len(mlp_idx),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "results": results,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
