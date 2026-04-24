#!/usr/bin/env python3
"""Resolve and validate registered task vectors for H1a/H1b."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.reporting import make_stage_report, write_report


REGISTERED_TASK_RUNS = {
    "GSM8K": "runs/2026-04-22-phase0-gsm8k-qwen2.5-1.5b",
    "MATH": "runs/2026-04-23-phase0-math-qwen2.5-1.5b",
    "AIME": "runs/2026-04-23-phase0-aime-qwen2.5-1.5b",
    "AMC": "runs/2026-04-24-phase0-amc-qwen2.5-1.5b",
    "MATH500": "runs/2026-04-24-phase0-math500-qwen2.5-1.5b",
    "HumanEval": "runs/2026-04-24-phase0-humaneval-qwen2.5-1.5b",
    "MBPP": "runs/2026-04-24-phase0-mbpp-qwen2.5-1.5b",
    "CommonsenseQA": "runs/2026-04-22-phase0-commonsenseqa-qwen2.5-1.5b",
    "HellaSwag": "runs/2026-04-23-phase0-hellaswag-qwen2.5-1.5b",
    "ARC-Challenge": "runs/2026-04-23-phase0-arc-challenge-qwen2.5-1.5b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        action="append",
        default=[],
        help="Optional local root containing downloaded Kaggle artifacts. Repeat as needed.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model identifier shared by the registered task vectors.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_paths(run_dir: Path, vector_path_raw: str, artifacts_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    repo_local = run_dir / "update_vector.npy"
    candidates.append(repo_local)

    vector_path = Path(vector_path_raw).expanduser()
    candidates.append(vector_path)

    provenance_parent = vector_path.parent.name
    for root in artifacts_roots:
        candidates.append(root / provenance_parent / "update_vector.npy")
        candidates.append(root / run_dir.name / "update_vector.npy")
        candidates.append(root / "outputs" / provenance_parent / "update_vector.npy")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_roots = [Path(raw).expanduser().resolve() for raw in args.artifacts_root]

    resolved_vectors: list[dict[str, object]] = []
    missing_tasks: list[str] = []
    sha_mismatch_tasks: list[str] = []

    for task, run_rel in REGISTERED_TASK_RUNS.items():
        run_dir = (repo_root / run_rel).resolve()
        provenance_path = run_dir / "update_vector_provenance.json"
        extraction_report_path = run_dir / "extraction_report.json"
        run_report_path = run_dir / "run_report.json"
        extraction_report = load_json(extraction_report_path)
        run_report = load_json(run_report_path) if run_report_path.exists() else None

        vector_path_raw: str | None = None
        expected_sha: str | None = None
        metadata_source = "run_report"

        if provenance_path.exists():
            provenance = load_json(provenance_path)
            vector_path_raw = provenance["vector_path"]
            if run_report is not None:
                expected_sha = run_report["update_vector"]["sha256"]
            else:
                expected_sha = extraction_report["validation"].get("sha256")
            metadata_source = "update_vector_provenance.json"
        else:
            vector_path_raw = extraction_report["artifacts"]["vector_path"]
            expected_sha = extraction_report["validation"].get("sha256")
            metadata_source = "extraction_report.json"

        if not expected_sha and run_report is not None:
            expected_sha = run_report["update_vector"]["sha256"]

        selected_path: Path | None = None
        selected_sha: str | None = None
        sha_matches = False
        tried = []

        for path in candidate_paths(run_dir, vector_path_raw, artifacts_roots):
            tried.append(str(path))
            if not path.exists():
                continue
            selected_sha = sha256_file(path)
            selected_path = path
            sha_matches = selected_sha == expected_sha
            if sha_matches:
                break

        if selected_path is None:
            missing_tasks.append(task)
        elif not sha_matches:
            sha_mismatch_tasks.append(task)

        resolved_vectors.append(
            {
                "task": task,
                "run_dir": str(run_dir),
                "metadata_source": metadata_source,
                "expected_sha256": expected_sha,
                "recorded_vector_path": vector_path_raw,
                "resolved_path": str(selected_path) if selected_path else None,
                "resolved_exists": selected_path is not None,
                "resolved_sha256": selected_sha,
                "sha256_matches": sha_matches,
                "tried_paths": tried,
            }
        )

    ready = not missing_tasks and not sha_mismatch_tasks
    command = None
    if ready:
        vector_args = [f"--vector {item['task']}={item['resolved_path']}" for item in resolved_vectors]
        command = (
            "python analysis/h1a_h1b_task_matrix.py "
            + " ".join(vector_args)
            + " --output-path runs/phase1-h1a-h1b-report.json"
        )

    notes = [
        "This is a preflight resolver for the registered H1a/H1b analysis inputs.",
        "Vectors are validated against the SHA-256 recorded in each imported Phase 0 run report.",
    ]
    if missing_tasks:
        notes.append(
            "Some registered vectors are not present locally yet. Download the matching "
            "update_vector.npy artifacts before running H1a/H1b."
        )
    if sha_mismatch_tasks:
        notes.append(
            "Some local vector files exist but do not match the recorded SHA-256. "
            "Re-download those artifacts before running H1a/H1b."
        )

    report = make_stage_report(
        stage="analysis",
        phase=1,
        task="MULTI_TASK",
        model=args.model_id,
        method="input_resolution",
        scope="registered",
        summary={
            "description": "Resolve and validate registered task vectors for H1a/H1b.",
            "registered_task_count": len(REGISTERED_TASK_RUNS),
            "ready_for_h1a_h1b": ready,
        },
        config={
            "registered_tasks": list(REGISTERED_TASK_RUNS.keys()),
            "run_dirs": REGISTERED_TASK_RUNS,
            "artifacts_roots": [str(path) for path in artifacts_roots],
        },
        metrics={
            "resolved_vectors": resolved_vectors,
            "resolved_count": sum(
                1 for item in resolved_vectors if item["resolved_exists"] and item["sha256_matches"]
            ),
            "missing_tasks": missing_tasks,
            "sha_mismatch_tasks": sha_mismatch_tasks,
            "h1a_h1b_command": command,
        },
        artifacts={
            "report_path": args.output_path,
        },
        runtime={},
        validation={
            "all_registered_tasks_accounted_for": len(resolved_vectors) == len(REGISTERED_TASK_RUNS),
            "all_vectors_present_locally": not missing_tasks,
            "all_vectors_match_expected_sha256": not sha_mismatch_tasks,
            "ready_for_h1a_h1b": ready,
        },
        notes=notes,
    )
    write_report(args.output_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
