"""Unified desktop application: sample -> Phast -> train -> predict in one window."""

from __future__ import annotations

from pathlib import Path


def launch(project_root: Path | None = None) -> int:
    from .app import run

    return run(project_root)
