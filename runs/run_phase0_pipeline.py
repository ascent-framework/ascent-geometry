#!/usr/bin/env python3
"""Orchestrate the Phase 0 pipeline into a single run directory.

This runner creates a dated `runs/{date}-phase0-{task}-{model}/` directory,
executes the reusable training/extraction/analysis scripts in order, and writes
an aggregate manifest that points to all stage outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.reporting import make_stage_report, write_report

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TASK = "gsm8k"
REGISTRY_PATH = REPO_ROOT / "config" / "registry.json"
TASK_REGISTRY_PATH = REPO_ROOT / "config" / "task_registry.json"
RUN_NOTE_TEMPLATE_PATH = REPO_ROOT / "runs" / "phase0_run_note_template.md"


def resolve_registered_task_name(task_arg: str) -> str:
    with TASK_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        tasks = json.load(handle).get("tasks", {})

    lowered = task_arg.lower()
    for task_name, task_config in tasks.items():
        if task_name.lower() == lowered:
            return task_name
        if str(task_config.get("slug", "")).lower() == lowered:
            return task_name

    raise KeyError(f"Unknown task '{task_arg}'. Known tasks: {sorted(tasks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--phase", default="phase0", help="Pipeline phase label.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task slug for run naming.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Full Hugging Face model id for the pilot.",
    )
    parser.add_argument(
        "--model-slug",
        default="qwen2.5-1.5b",
        help="Short model slug used in the run directory name.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(REPO_ROOT / "runs"),
        help="Root directory where run directories are created.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="/kaggle/working",
        help="Root directory for large adapter/vector artifacts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Pilot training steps passed through to the training script.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Assume adapter artifacts already exist under the derived artifacts directory.",
    )
    parser.add_argument(
        "--adapter-path",
        help="Explicit adapter path to use when --skip-training is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and manifest paths without executing stage scripts.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(json.dumps({"dry_run_command": cmd}, indent=2))
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    registered_task_name = resolve_registered_task_name(args.task)
    run_name = f"{args.date}-{args.phase}-{args.task}-{args.model_slug}"
    runs_root = Path(args.runs_root)
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_root) / run_name
    training_report_path = run_dir / "training_report.json"
    extraction_report_path = run_dir / "extraction_report.json"
    analysis_report_path = run_dir / "analysis_report.json"
    manifest_path = run_dir / "run_manifest.json"
    run_note_path = run_dir / "run_note.md"

    derived_adapter_path = artifacts_dir / "adapter"
    adapter_path = Path(args.adapter_path) if args.adapter_path else derived_adapter_path
    vector_path = artifacts_dir / "update_vector.npy"

    training_cmd = [
        sys.executable,
        str(REPO_ROOT / "training" / "train_grpo_task.py"),
        "--task",
        registered_task_name,
        "--model-id",
        args.model_id,
        "--output-dir",
        str(artifacts_dir),
        "--max-steps",
        str(args.max_steps),
        "--scope",
        "pilot_only",
    ]
    extraction_cmd = [
        sys.executable,
        str(REPO_ROOT / "extraction" / "extract_registered_update_vector.py"),
        "--model-id",
        args.model_id,
        "--adapter-path",
        str(adapter_path),
        "--output-path",
        str(vector_path),
    ]
    analysis_cmd = [
        sys.executable,
        str(REPO_ROOT / "analysis" / "pilot_svd_diagnostic.py"),
        "--model-id",
        args.model_id,
        "--adapter-path",
        str(adapter_path),
        "--output-path",
        str(analysis_report_path),
    ]

    if not args.skip_training:
        run_command(training_cmd, dry_run=args.dry_run)
        source_training_report = artifacts_dir / "training_run_report.json"
        if source_training_report.exists() and not args.dry_run:
            training_report_path.write_text(source_training_report.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        note = {
            "skipped": True,
            "adapter_path": str(adapter_path),
        }
        if not args.dry_run:
            training_report_path.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")

    run_command(extraction_cmd, dry_run=args.dry_run)
    source_extraction_report = vector_path.with_name(f"{vector_path.stem}_report.json")
    if source_extraction_report.exists() and not args.dry_run:
        extraction_report_path.write_text(source_extraction_report.read_text(encoding="utf-8"), encoding="utf-8")

    run_command(analysis_cmd, dry_run=args.dry_run)

    if not args.dry_run and not run_note_path.exists():
        run_note_path.write_text(RUN_NOTE_TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = make_stage_report(
        stage="analysis",
        phase=0,
        task=registered_task_name,
        model=args.model_id,
        method="pipeline",
        scope="pilot_only",
        summary={
            "description": "Aggregate manifest for a full Phase 0 pilot pipeline run.",
            "run_name": run_name,
        },
        config={
            "phase_label": args.phase,
            "skip_training": args.skip_training,
            "max_steps": args.max_steps,
        },
        metrics={},
        artifacts={
            "run_dir": run_dir,
            "artifacts_dir": artifacts_dir,
            "registry_path": REGISTRY_PATH,
            "task_registry_path": TASK_REGISTRY_PATH,
            "run_note_template_path": RUN_NOTE_TEMPLATE_PATH,
            "run_note_path": run_note_path,
            "training_report": training_report_path,
            "extraction_report": extraction_report_path,
            "analysis_report": analysis_report_path,
            "adapter_path": adapter_path,
            "vector_path": vector_path,
        },
        runtime={
            "dry_run": args.dry_run,
        },
        validation={
            "training_stage_requested": not args.skip_training,
            "extraction_stage_requested": True,
            "analysis_stage_requested": True,
        },
        notes=[
            "This manifest organizes run-local metadata only.",
            "Large artifacts remain outside the repository under artifacts_root.",
        ],
    )
    if not args.dry_run:
        write_report(manifest_path, manifest)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
