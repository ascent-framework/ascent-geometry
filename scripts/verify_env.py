#!/usr/bin/env python3
"""Verify the local Python environment for ascent-geometry."""

from __future__ import annotations

import importlib
import os
import platform
import sys


REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
]

OPTIONAL_PACKAGES = [
    "bitsandbytes",
]


def try_import(name: str) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(name)
    except Exception:
        return False, None
    return True, getattr(module, "__version__", "unknown")


def main() -> int:
    print("Environment verification")
    print(f"python_executable={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"virtual_env={os.environ.get('VIRTUAL_ENV')}")
    print()

    missing: list[str] = []

    print("Required packages")
    for name in REQUIRED_PACKAGES:
        ok, version = try_import(name)
        if ok:
            print(f"  ok   {name} {version}")
        else:
            print(f"  miss {name}")
            missing.append(name)

    print()
    print("Optional packages")
    for name in OPTIONAL_PACKAGES:
        ok, version = try_import(name)
        if ok:
            print(f"  ok   {name} {version}")
        else:
            print(f"  miss {name}")

    print()
    print("Checks")
    inside_venv = bool(os.environ.get("VIRTUAL_ENV"))
    print(f"  virtual_env_active={inside_venv}")

    if missing:
        print(f"  missing_required={','.join(missing)}")
        return 1

    print("  status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
