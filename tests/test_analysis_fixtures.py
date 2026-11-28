"""Fixture builders for the analysis tests (no tests in here).

* :func:`write_run` writes a run directory (trace.parquet + stats.json) from a list of
  pair contrasts, with exactly the columns :func:`build_trace_rows` emits.
* :class:`ScriptedSUT` plugs into the real :class:`VLMBoundaryTester` and returns a
  prescribed contrast per call, so a full run can be checked against numbers worked out by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def logprobs_for(g: float, n_candidates: int = 2, third: float = -5.0) -> list[float]:
    """Length-N log-prob vector whose pair contrast lp[0] - lp[1] equals g."""
    lp = [-0.5 + g / 2.0, -0.5 - g / 2.0] + [third] * (n_candidates - 2)
    return lp


def write_run(
    root: Path,
    name: str,
    g_values,
    *,
    pop: int,
    stats_extra: dict | None = None,
    genotypes=None,
    cache_hits=None,
    n_candidates: int = 2,
    logprobs=None,
    img_mag=None,
    txt_mag=None,
    class_a: str = "junco",
    class_b: str = "sparrow",
    seed_idx: int = 0,
    generations: int | None = None,
    config: dict | None = None,
    min_cross: dict | None = None,
    boundary_pair: dict | None = None,
) -> Path:
    """Create ``<root>/<name>/{trace.parquet,stats.json}`` for a run with the given contrasts (plus min_cross.json /
    boundary_pair.json when given)."""
    g_values = list(g_values)
    n = len(g_values)
    gens = generations if generations is not None else n // pop
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(g_values):
        lp = logprobs[i] if logprobs is not None else logprobs_for(g, n_candidates)
        pa = float(1 / (1 + np.exp(-(lp[0] - lp[1]))))
        rows.append({
            "seed_id": seed_idx,
            "generation": i // pop,
            "individual": i % pop,
            "genotype": list(genotypes[i]) if genotypes is not None else [1, 0, 1],
            "logprobs": list(lp),
            "probs": [float(x) for x in torch.softmax(torch.tensor(lp), -1)],
            "decoded_text": "q",
            "cache_hit": bool(cache_hits[i]) if cache_hits is not None else False,
            "predicted_class": [class_a, class_b][int(np.argmax(lp[:2]))],
            "p_class_a": pa,
            "p_class_b": 1 - pa,
            "diag_img_rms": float(img_mag[i]) if img_mag is not None else 0.1 * (i + 1),
            "diag_txt_cosdist": float(txt_mag[i]) if txt_mag is not None else 0.01 * (i + 1),
            "fitness_TgtBal": abs(float(g)),
        })
    pd.DataFrame(rows).to_parquet(run / "trace.parquet")
    stats = {
        "schema_version": 1, "seed_idx": seed_idx, "class_a": class_a, "class_b": class_b,
        "runtime_seconds": 12.5, "generations": gens, "pop_size": pop,
        "categories": [class_a, class_b], "pair": [class_a, class_b], "target_classes": [0, 1],
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf", "seed_selection_mode": "roster",
        "seed_metadata": {
            "level_anchor": 0, "level_target": 1, "anchor_class_concrete": "junco",
            "target_class_concrete": "ostrich", "seed_idx_in_class": seed_idx,
        },
    }
    stats.update(stats_extra or {})
    (run / "stats.json").write_text(json.dumps(stats))
    if config is not None:
        (run / "config.json").write_text(json.dumps(config))
    if min_cross is not None:
        (run / "min_cross.json").write_text(json.dumps(min_cross))
    if boundary_pair is not None:
        (run / "boundary_pair.json").write_text(json.dumps(boundary_pair))
    return run


class ScriptedSUT:
    """Fake SUT for VLMBoundaryTester: call i returns a contrast of schedule[i]."""

    text_embedder = None
    device_str = "cpu"
    last_call_cached = False
    cache_stats = {"hits": 0, "misses": 0}

    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.calls = 0

    def process_input(self, image, text=None, categories=None):
        g = self.schedule[self.calls]
        self.calls += 1
        n = len(categories) if categories is not None else 2
        return torch.tensor(logprobs_for(g, n), dtype=torch.float32)
