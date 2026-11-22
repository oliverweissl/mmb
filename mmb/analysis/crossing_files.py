"""Files of the first crossing of a run: first_cross.png/.json, written by the tester at the first sign flip."""
from __future__ import annotations

from pathlib import Path


def first_crossing(run_dir: Path) -> tuple[Path, Path] | None:
    """(png, json) of the first crossing input of the run, or None if the run did not cross."""
    run_dir = Path(run_dir)
    png, js = run_dir / "first_cross.png", run_dir / "first_cross.json"
    return (png, js) if png.exists() and js.exists() else None
