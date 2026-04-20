"""Common report builders for ASCENT-G experiment scripts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


REPORT_SCHEMA_VERSION = "2026-04-21"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonify(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonify(inner) for inner in value]
    return value


def make_stage_report(
    *,
    stage: str,
    phase: int,
    task: str,
    model: str,
    method: str,
    scope: str,
    summary: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "stage": stage,
        "phase": phase,
        "task": task,
        "model": model,
        "method": method,
        "scope": scope,
        "summary": _jsonify(summary or {}),
        "config": _jsonify(config or {}),
        "metrics": _jsonify(metrics or {}),
        "artifacts": _jsonify(artifacts or {}),
        "runtime": _jsonify(runtime or {}),
        "validation": _jsonify(validation or {}),
        "notes": _jsonify(notes or []),
    }


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
