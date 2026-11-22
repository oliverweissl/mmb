"""Collect one CSV row per evolutionary run.

Usage::

    python -m mmb.analysis.collect <runs_root> --out results/<exp>.csv \\
        [--default arm=a1_random_tiered] [--exp-id E1] [--epsilon 1e-2]

A run directory is any directory under ``runs_root`` that holds a ``trace.parquet``
(written by :class:`mmb.evolutionary.vlm_boundary_tester.VLMBoundaryTester`).

Definitions
-----------
* ``g = logprobs[class_a] - logprobs[class_b]`` per evaluation, with the pair positions
  taken from ``stats["target_classes"]`` (default ``(0, 1)``); class_a is the anchor answer.
* Anchor sign: sign of g on the unmodified seed. Preferred source: the explicit anchor reference
  evaluation the tester stores as ``stats["anchor_g"]`` (outside the budget; ``anchor_source =
  "reference_eval"``). Fallbacks: the first evaluation whose genotype is all zeros
  (``"zero_genotype"``), then an assumed class_a anchor (``"assumed_class_a"``, sign +1).
* A run *crosses* if some evaluation has ``sign(g) * anchor_sign < 0`` (strict: g == 0 is not a
  crossing). ``first_cross_eval`` is the 1-based position in evaluation order (generation-major,
  then individual) counting cache hits; a run that never crosses is censored at its budget
  (``generations * pop_size``, else the number of evaluations). ``first_cross_eval_nocache``
  counts only evaluations with ``cache_hit == False``.
* ``floor_nats = min |g|`` over all evaluations, ``reached_band = floor_nats <= epsilon``.
* ``label_flip``: the argmax over *all scored candidates* differs from the anchor's argmax in
  some evaluation.

Run metadata (``exp_id, arm, task, sut, cell_id, rng_seed``) comes from ``stats.json`` and a config
snapshot, never from the directory name; see :func:`resolve_meta`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mmb.analysis.cost import cost_of_crossing  # noqa: E402

logger = logging.getLogger(__name__)

EPSILON = 1e-2

#: Per-row magnitude diagnostics logged for EVERY arm (not optimised): RMS pixel difference to the origin at
#: codec resolution ([0, 1]) and the SUT text-embedding cosine distance to the original question. They are the
#: primary magnitude columns; the optimised fitness_* columns of the multi-objective arms are the fallback.
DIAG_COLUMNS = ("diag_img_rms", "diag_txt_cosdist")


def _mag_name(kind: str, columns, fit_cols, diag_only: bool = False):
    """Column holding the image ('img') or text ('txt') magnitude: diag_* first, else fitness_* (unless diag_only)."""
    diag = "diag_img_rms" if kind == "img" else "diag_txt_cosdist"
    if diag in columns:
        return diag
    if diag_only:
        return None
    prefix = "fitness_MatrixDistance" if kind == "img" else "fitness_TextDist"
    hit = [c for c in fit_cols if c.startswith(prefix)]
    return hit[0] if hit else None

#: Columns requested by the paper's E1 table, in this order.
CORE_COLUMNS = [
    "exp_id", "arm", "task", "sut", "cell_id", "rng_seed",
    "crossed", "crossed_at_gen0", "first_cross_eval", "label_flip",
    "floor_nats", "reached_band", "evals_total",
    "img_mag_at_cross", "txt_mag_at_cross", "wall_s", "run_dir",
]
#: Extra bookkeeping columns appended after the core ones.
EXTRA_COLUMNS = [
    "first_cross_gen", "first_cross_eval_nocache", "n_cross_evals",
    "budget", "complete", "anchor_source", "seed_idx",
    # predictors of E2b (see mmb.analysis.predictors)
    "anchor_g", "prior_g", "probe_gen0", "min_gen0", "answer_words_a", "answer_words_b",
]
#: Cost-of-crossing columns (see mmb.analysis.cost).
COST_COLUMNS = [
    "min_edits_cross", "edits_img_at_min", "edits_txt_at_min", "hv_edits_cross", "min_txt_chardist_cross",
    "best_img_mag_cross", "best_txt_mag_cross", "hv_mag_cross", "mag_source",
]
COLUMNS = CORE_COLUMNS + EXTRA_COLUMNS + COST_COLUMNS

CONFIG_FILENAMES = (
    "config.json", "config.yaml", "config.yml",
    "resolved_config.json", "resolved_config.yaml", "run_config.json", "run_config.yaml",
)
META_KEYS = ("exp_id", "arm", "task", "sut", "cell_id", "rng_seed")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def evaluate_trace(
    logprobs: np.ndarray,
    generation: np.ndarray,
    *,
    target_classes: tuple[int, int] = (0, 1),
    anchor_row: int | None = None,
    anchor_g: float | None = None,
    anchor_argmax: int | None = None,
    cache_hit: np.ndarray | None = None,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Crossing statistics of one run from rows already in evaluation order.

    :param logprobs: ``(n_evals, n_candidates)`` array, one row per evaluation, in evaluation
        order (generation-major, then individual).
    :param generation: ``(n_evals,)`` generation of every row.
    :param target_classes: positions of (class_a, class_b) inside the scored candidates.
    :param anchor_row: index of the unmodified-seed evaluation, if the trace has one.
    :param anchor_g: pair contrast of the explicit anchor reference evaluation (``stats["anchor_g"]``);
        preferred over ``anchor_row`` when both exist.
    :param anchor_argmax: position of the anchor's argmax among the scored candidates (reference eval).
    :param cache_hit: optional boolean array, used for ``first_cross_eval_nocache``.
    """
    lp = np.asarray(logprobs, dtype=float)
    gen = np.asarray(generation)
    ia, ib = target_classes
    g = lp[:, ia] - lp[:, ib]

    if anchor_g is not None:
        anchor_g_val = float(anchor_g)
        anchor_arg = anchor_argmax if anchor_argmax is not None else (ia if anchor_g_val >= 0 else ib)
        source = "reference_eval"
    elif anchor_row is not None:
        anchor_g_val = float(g[anchor_row])
        anchor_arg = int(np.argmax(lp[anchor_row]))
        source = "zero_genotype"
    else:
        anchor_g_val = 1.0
        anchor_arg = ia
        source = "assumed_class_a"
    anchor_sign = 1.0 if anchor_g_val >= 0 else -1.0

    with np.errstate(invalid="ignore"):
        crossing = (np.sign(g) * anchor_sign) < 0          # NaN -> False
    flips = np.argmax(lp, axis=1) != anchor_arg

    g0 = np.abs(g[gen == 0]) if len(gen) else np.zeros(0)
    g0 = g0[~np.isnan(g0)]
    out: dict[str, Any] = {
        "probe_gen0": float(np.median(g0)) if len(g0) else float("nan"),
        "min_gen0": float(g0.min()) if len(g0) else float("nan"),
        "crossed": bool(crossing.any()),
        "crossed_at_gen0": bool(crossing[gen == 0].any()),
        "label_flip": bool(flips.any()),
        "floor_nats": float(np.nanmin(np.abs(g))) if len(g) and not np.all(np.isnan(g)) else float("nan"),
        "n_cross_evals": int(crossing.sum()),
        "anchor_source": source,
        "anchor_g": anchor_g_val if source != "assumed_class_a" else float("nan"),
        "cross_index": None,
        "_crossing": crossing,
        "first_cross_eval": None,
        "first_cross_gen": None,
        "first_cross_eval_nocache": None,
    }
    out["reached_band"] = bool(out["floor_nats"] <= epsilon) if not np.isnan(out["floor_nats"]) else False
    if out["crossed"]:
        i = int(np.argmax(crossing))
        out["cross_index"] = i
        out["first_cross_eval"] = i + 1
        out["first_cross_gen"] = int(gen[i])
        if cache_hit is not None:
            fresh = ~np.asarray(cache_hit, dtype=bool)
            fresh_cross = crossing & fresh
            if fresh_cross.any():
                j = int(np.argmax(fresh_cross))
                out["first_cross_eval_nocache"] = int(fresh[: j + 1].sum())
            # else: every crossing was a cache hit, which needs an earlier identical evaluation that
            # crossed too; cannot happen, left as None.
    return out


def derive_arm(cfg: dict[str, Any]) -> str | None:
    """Arm id (see scripts/make_arm_configs.py) from a raw/nested config dict."""
    if not cfg:
        return None
    modality = _dig(cfg, "modality")
    algo = _dig(cfg, "optimizer", "algorithm")
    if modality == "text_only":
        return "a6_text_only"
    if modality == "image_only":
        return "a7_image_only"
    if algo is None:
        return None
    if algo == "random":
        mode = _dig(cfg, "optimizer", "sampling", "mode")
        return "a2_random_uniform" if mode == "uniform" else "a1_random_tiered"
    if algo == "ga":
        return "a3_ga_single"
    if algo == "nsga2":
        return "a4_nsga2"
    if algo == "deepjanus":
        return "a9_deepjanus"
    if algo == "agemoea2":
        if _dig(cfg, "optimizer", "boundary_objective") == "softmax_balance":
            return "a5_mimicry_objective"
        return "a0_directed"
    return str(algo)


def cfg_from_run_meta(rm: dict[str, Any]) -> dict[str, Any]:
    """Config-shaped dict from the arm-defining keys the tester stamps into ``stats["run_meta"]``."""
    if not rm or "algorithm" not in rm:
        return {}
    return {
        "modality": rm.get("modality"),
        "optimizer": {
            "algorithm": rm.get("algorithm"),
            "boundary_objective": rm.get("boundary_objective"),
            "proximity_only": rm.get("proximity_only"),
            "sampling": {"mode": rm.get("sampling_mode")},
        },
        "sut": {"model_id": rm.get("sut_model_id")},
    }


def _dig(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def derive_task(stats: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    if _dig(cfg, "modality") == "grounding":
        return "grounding"
    mode = _dig(cfg, "seeds", "mode") or stats.get("seed_selection_mode")
    if mode in ("roster", "gap_filter"):
        return "imagenet_pair"
    if mode == "refcocoplus":
        return "grounding"
    if mode == "slot_items":
        return "slot_items"
    return str(mode) if mode else None


def derive_sut(stats: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    model = _dig(cfg, "sut", "model_id") or stats.get("model_id")
    if not model:
        return None
    name = str(model).rstrip("/").split("/")[-1]
    if _dig(cfg, "sut", "load_in_8bit"):
        name += "-int8"
    elif _dig(cfg, "sut", "load_in_4bit"):
        name += "-int4"
    return name


def derive_cell_id(stats: dict[str, Any]) -> str:
    """Identifier of the (cell, anchor photograph) unit a run belongs to.

    Roster runs: ``<anchor>|<target>|La<l>-Lt<l>|p<seed_idx_in_class>`` from ``seed_metadata``
    (concrete classes, abstraction levels and the photograph index within the anchor class), so
    that arms run on the same cell *and* photograph match. Otherwise ``<class_a>|<class_b>|i<seed_idx>``.
    """
    meta = stats.get("seed_metadata") or {}
    if "anchor_class_concrete" in meta and "target_class_concrete" in meta:
        parts = [
            str(meta["anchor_class_concrete"]), str(meta["target_class_concrete"]),
            f"La{meta.get('level_anchor', '?')}-Lt{meta.get('level_target', '?')}",
        ]
        if "seed_idx_in_class" in meta:
            parts.append(f"p{meta['seed_idx_in_class']}")
        return "|".join(parts)
    return f"{stats.get('class_a', '?')}|{stats.get('class_b', '?')}|i{stats.get('seed_idx', '?')}"


def resolve_meta(
    stats: dict[str, Any],
    cfg: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run metadata. Priority: explicit stats keys > stats["run_meta"] > config > derived > defaults.

    ``defaults`` (CLI ``--default k=v``) fill only what is still missing.
    """
    defaults = defaults or {}
    explicit = dict(stats.get("run_meta") or {})
    rm_cfg = cfg_from_run_meta(explicit)
    if rm_cfg:  # run_meta is what the tester actually ran; the config snapshot only fills gaps
        cfg = {**cfg, **{k: v for k, v in rm_cfg.items() if v is not None}}
        cfg["optimizer"] = {**(_dig(cfg, "optimizer") or {}), **rm_cfg["optimizer"]}
    out: dict[str, Any] = {}
    for key in META_KEYS:
        if key in stats and stats[key] is not None:
            out[key] = stats[key]
        elif explicit.get(key) is not None:
            out[key] = explicit[key]
    derived = {
        "exp_id": _dig(cfg, "exp_id") or _dig(cfg, "name"),
        "arm": _dig(cfg, "arm") or derive_arm(cfg),
        "task": _dig(cfg, "task") or derive_task(stats, cfg),
        "sut": derive_sut(stats, cfg),
        "cell_id": derive_cell_id(stats),
        "rng_seed": stats.get("rng_seed", _dig(cfg, "seed")),
    }
    for key in META_KEYS:
        if out.get(key) is None and derived.get(key) is not None:
            out[key] = derived[key]
        if out.get(key) is None and key in defaults:
            out[key] = defaults[key]
        out.setdefault(key, None)
    return out


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def find_runs(runs_root: Path) -> list[Path]:
    """Directories under *runs_root* (inclusive) that contain a ``trace.parquet``."""
    runs_root = Path(runs_root)
    found = []
    for dirpath, _dirs, files in os.walk(runs_root):
        if "trace.parquet" in files:
            found.append(Path(dirpath))
    return sorted(found)


def _load_config(run_dir: Path, stats: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    for key in ("config", "config_snapshot"):
        if isinstance(stats.get(key), dict):
            return stats[key]
    d = run_dir
    while True:
        for fn in CONFIG_FILENAMES:
            p = d / fn
            if p.is_file():
                try:
                    if p.suffix == ".json":
                        return json.loads(p.read_text())
                    import yaml
                    return yaml.safe_load(p.read_text()) or {}
                except Exception as exc:  # pragma: no cover - corrupt file
                    logger.warning("unreadable config %s: %s", p, exc)
        if d == runs_root or d.parent == d:
            break
        d = d.parent
    return {}


def load_run(
    run_dir: Path,
    runs_root: Path | None = None,
    defaults: dict[str, Any] | None = None,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """One output row (a dict with :data:`COLUMNS`) for the run in *run_dir*."""
    run_dir = Path(run_dir)
    runs_root = Path(runs_root) if runs_root is not None else run_dir
    stats_path = run_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    cfg = _load_config(run_dir, stats, runs_root)

    import pyarrow.parquet as pq

    names = set(pq.read_schema(run_dir / "trace.parquet").names)
    want = [c for c in ("generation", "individual", "genotype", "logprobs", "cache_hit", "decoded_text") if c in names]
    fit_cols = sorted(c for c in names if c.startswith("fitness_"))
    diag_cols = sorted(c for c in names if c in DIAG_COLUMNS)
    df = pd.read_parquet(run_dir / "trace.parquet", columns=want + fit_cols + diag_cols)
    df = df.sort_values(["generation", "individual"], kind="stable")
    df = df.drop_duplicates(["generation", "individual"], keep="last").reset_index(drop=True)

    tc = stats.get("target_classes") or (0, 1)
    lp = np.vstack([np.asarray(v, dtype=float) for v in df["logprobs"]]) if len(df) else np.zeros((0, 2))
    gen = df["generation"].to_numpy() if len(df) else np.zeros(0, dtype=int)

    anchor_row = None
    if "genotype" in df and len(df):
        for i, gt in enumerate(df["genotype"]):
            if not np.any(np.asarray(gt)):
                anchor_row = i
                break
    cache_hit = df["cache_hit"].to_numpy() if "cache_hit" in df else None
    anchor_argmax = None
    if stats.get("anchor_argmax") is not None:
        cats = stats.get("categories") or []
        a = stats["anchor_argmax"]
        anchor_argmax = cats.index(a) if isinstance(a, str) and a in cats else (a if isinstance(a, int) else None)
    res = evaluate_trace(
        lp, gen, target_classes=(int(tc[0]), int(tc[1])), anchor_row=anchor_row,
        anchor_g=stats.get("anchor_g"), anchor_argmax=anchor_argmax,
        cache_hit=cache_hit, epsilon=epsilon,
    )

    n_evals = len(df)
    gens, pop = stats.get("generations"), stats.get("pop_size")
    rm_budget = (stats.get("run_meta") or {}).get("eval_budget")
    budget = int(rm_budget) if rm_budget else (int(gens) * int(pop) if gens and pop else n_evals)
    meta = resolve_meta(stats, cfg, defaults)

    def mag(prefix: str) -> float:
        c = _mag_name("img" if "Matrix" in prefix else "txt", df.columns, fit_cols)
        if res["cross_index"] is None or c is None:
            return float("nan")
        return float(df[c].iloc[res["cross_index"]])

    def mag_col(kind: str):
        c = _mag_name(kind, df.columns, fit_cols)
        return df[c].to_numpy(float) if c else None

    seed_text = (stats.get("seed_metadata") or {}).get("prompt_template") or stats.get("prompt_template")
    cost = cost_of_crossing(
        res["_crossing"], list(df["genotype"]) if "genotype" in df else [],
        int(stats.get("image_dim") or 0), int(stats.get("text_dim") or 0),
        img_mag=mag_col("img"), txt_mag=mag_col("txt"),
        texts=list(df["decoded_text"]) if "decoded_text" in df else None, seed_text=seed_text,
    ) if "genotype" in df and len(df) else {}

    row = {
        **meta,
        **cost,
        "mag_source": "diag" if _mag_name("img", df.columns, fit_cols, True) else (
            "fitness" if _mag_name("img", df.columns, fit_cols) else "none"),
        "crossed": res["crossed"],
        "crossed_at_gen0": res["crossed_at_gen0"],
        "first_cross_eval": res["first_cross_eval"] if res["crossed"] else budget,
        "label_flip": res["label_flip"],
        "floor_nats": res["floor_nats"],
        "reached_band": res["reached_band"],
        "evals_total": n_evals,
        "img_mag_at_cross": mag("fitness_MatrixDistance"),
        "txt_mag_at_cross": mag("fitness_TextDist"),
        "wall_s": stats.get("runtime_seconds", float("nan")),
        "run_dir": str(run_dir),
        "first_cross_gen": res["first_cross_gen"],
        "first_cross_eval_nocache": res["first_cross_eval_nocache"],
        "n_cross_evals": res["n_cross_evals"],
        "budget": budget,
        # complete = all configured generations are in the trace (a generation can hold fewer than
        # pop_size rows when the optimizer drops duplicates) or the run stopped early on purpose
        "complete": bool(
            (gens and len(gen) and int(gen.max()) + 1 >= int(gens)) or "early_stop" in stats
            or (not gens and n_evals > 0)
        ),
        "anchor_source": res["anchor_source"],
        "seed_idx": stats.get("seed_idx"),
        "anchor_g": res["anchor_g"],
        "prior_g": float(stats["prior_g"]) if stats.get("prior_g") is not None else float("nan"),
        "probe_gen0": res["probe_gen0"],
        "min_gen0": res["min_gen0"],
        "answer_words_a": len(str(stats.get("class_a", "")).split()) or None,
        "answer_words_b": len(str(stats.get("class_b", "")).split()) or None,
    }
    return {c: row.get(c) for c in COLUMNS}


def collect(
    runs_root: Path,
    defaults: dict[str, Any] | None = None,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    runs_root = Path(runs_root)
    rows = []
    for rd in find_runs(runs_root):
        try:
            rows.append(load_run(rd, runs_root, defaults, epsilon))
        except Exception as exc:
            logger.warning("skipping %s: %s", rd, exc)
    return pd.DataFrame(rows, columns=COLUMNS)


def _parse_kv(items: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--default", action="append", default=[], metavar="KEY=VALUE",
                    help="fill a metadata key (exp_id, arm, task, sut, cell_id, rng_seed) only where "
                         "stats/config do not provide it; repeatable")
    ap.add_argument("--exp-id", default=None, help="shorthand for --default exp_id=...")
    ap.add_argument("--epsilon", type=float, default=EPSILON)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    defaults = _parse_kv(args.default)
    if args.exp_id:
        defaults["exp_id"] = args.exp_id
    df = collect(args.runs_root, defaults, args.epsilon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_unknown = int(df["arm"].isna().sum()) if len(df) else 0
    logger.info("wrote %d runs to %s", len(df), args.out)
    if n_unknown:
        logger.warning("%d run(s) have no arm: pass --default arm=<id> or stamp stats.json/config", n_unknown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
