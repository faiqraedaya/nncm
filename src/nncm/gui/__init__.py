"""Unified desktop application: sample -> Phast -> train -> predict in one window."""

from __future__ import annotations

from pathlib import Path


def launch(project_root: Path | None = None) -> int:
    """Open the desktop application.

    Deferred so that importing :mod:`nncm.gui` — which the CLI does merely to
    find this function — does not drag in the whole widget stack.
    """
    from .startup import run

    return run(project_root)
