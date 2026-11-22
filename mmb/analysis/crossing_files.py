"""Files of the first ROBUST crossing of a run (|g| >= tau nats with reversed sign).

The tester stores the first plain sign flip (first_cross.png/.json). That input is also the first robust crossing
when its |g| >= tau. Otherwise scripts/minimize_crossings.py re-renders the first robust crossing evaluation from the
trace into first_cross_robust.png/.json. A run with neither has no robust crossing input available.
"""
from __future__ import annotations

import json
from pathlib import Path

ROBUST_TAU = 0.15


def robust_first_crossing(run_dir: Path, tau: float = ROBUST_TAU) -> tuple[Path, Path] | None:
    """(png, json) of the first robust crossing input of the run, or None if none is stored."""
    run_dir = Path(run_dir)
    plain_png, plain_json = run_dir / "first_cross.png", run_dir / "first_cross.json"
    if plain_png.exists() and plain_json.exists():
        g = json.loads(plain_json.read_text()).get("g")
        if g is not None and abs(float(g)) >= tau:
            return plain_png, plain_json
    rob_png, rob_json = run_dir / "first_cross_robust.png", run_dir / "first_cross_robust.json"
    if rob_png.exists() and rob_json.exists():
        return rob_png, rob_json
    return None
