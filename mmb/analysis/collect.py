"""Collect one CSV row per run (continuous genotype, 2026-11-28).

Usage::

    python -m mmb.analysis.collect <runs_root> --out results/<exp>.csv [--exp-id E] [--default arm=a0_directed]

A run directory is any directory under ``runs_root`` that holds a ``trace.parquet`` (written by
:class:`mmb.evolutionary.vlm_boundary_tester.VLMBoundaryTester`).

Definitions
-----------
* ``g = logprobs[class_a] - logprobs[class_b]`` per evaluation (pair positions ``stats["target_classes"]``).
  Anchor sign: sign of ``stats["anchor_g"]`` (reference evaluation outside the budget); fallbacks: the first
  all-zero genotype, then class_a.
* Budget: ``run_meta.eval_budget`` (else generations x pop_size). Every trace-based metric uses only the first
  ``budget`` evaluations (evaluation order = generation-major, then individual).
* A run *crosses* if some in-budget evaluation has ``sign(g) * anchor_sign < 0``. ``first_cross_eval`` is the
  1-based position of the first such evaluation, censored at the budget.
* Smallest crossing (``min_cross.json``, written by the tester): the in-budget crossing of smallest weighted
  magnitude. Columns ``magnitude, mag_image, mag_text`` (weighted block magnitudes), ``min_eval_idx``,
  ``min_confirmed`` (re-scored on the canonical path). Budget-bound: they are NaN unless ``within_budget``.
  ``min_text_changed``: the rendered question differs from the original; ``min_text_d``: its MiniLM distance
  from the question pool (configs/text/question_pool.json; 0 = original). ``min_modality``: image (image
  edited, question unchanged), text (question changed, no image edit), joint, or none.
* Boundary pair (``boundary_pair.json``): ``bp_source`` (own_phase2 | posthoc_bisection | none), ``bp_gap``,
  ``bp_lam_on/off``, ``bp_kind`` (text_step | continuous), ``bp_text_differs``, per side g / magnitude /
  mag_image / mag_text / text, and ``nb_extra_calls`` (canonical calls outside the budget).
* ``floor_nats = min |g|`` over in-budget evaluations, ``reached_band = floor_nats <= epsilon``; ``label_flip``:
  the argmax over all scored candidates differs from the anchor's in some in-budget evaluation.

Run metadata (``exp_id, arm, task, sut, cell_id, rng_seed``) comes from ``stats.json`` (run_meta) and the config
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

logger = logging.getLogger(__name__)

EPSILON = 1e-2

#: Arms of the study (balanced design): MMBA, the two baselines, the ablations.
ARMS = ("a0_directed", "a5_mimicry_objective", "a9_deepjanus", "a6_text_only", "a7_image_only", "x1_mmba_unguided")

#: Per-row magnitude diagnostics of the trace (every arm): RMS pixel difference to the origin and the SUT text-embedding
#: cosine distance to the original question, read at the first crossing.
DIAG_COLUMNS = ("diag_img_rms", "diag_txt_cosdist")

ID_COLUMNS = ["exp_id", "arm", "task", "sut", "cell_id", "rng_seed", "seed_idx", "algorithm", "w_img", "w_txt"]
CROSS_COLUMNS = [
    "budget", "evals_total", "evals_in_budget", "complete", "anchor_source",
    "crossed", "crossed_at_gen0", "first_cross_eval", "first_cross_gen", "first_cross_eval_nocache", "n_cross_evals",
    "label_flip", "floor_nats", "reached_band", "img_mag_at_cross", "txt_mag_at_cross",
]
MIN_COLUMNS = [
    "min_found", "min_within_budget", "min_eval_idx", "min_confirmed", "magnitude", "mag_image", "mag_text",
    "min_text", "min_text_changed", "min_text_d", "min_modality",
]
PAIR_COLUMNS = [
    "bp_source", "bp_gap", "bp_lam_on", "bp_lam_off", "bp_kind", "bp_text_differs", "bp_eps", "nb_extra_calls",
    "bp_on_g", "bp_on_magnitude", "bp_on_mag_image", "bp_on_mag_text", "bp_on_text",
    "bp_off_g", "bp_off_magnitude", "bp_off_mag_image", "bp_off_mag_text", "bp_off_text",
]
PREDICTOR_COLUMNS = ["anchor_g", "prior_g", "probe_gen0", "min_gen0", "answer_words_a", "answer_words_b"]
COLUMNS = ID_COLUMNS + CROSS_COLUMNS + MIN_COLUMNS + PAIR_COLUMNS + PREDICTOR_COLUMNS + ["question", "wall_s", "run_dir"]

CONFIG_FILENAMES = (
    "config.json", "config.yaml", "config.yml",
    "resolved_config.json", "resolved_config.yaml", "run_config.json", "run_config.yaml",
)
META_KEYS = ("exp_id", "arm", "task", "sut", "cell_id", "rng_seed")
POOL_PATH = Path(__file__).resolve().parents[2] / "configs" / "text" / "question_pool.json"


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
    """Arm id (scripts/make_arm_configs.py) from a raw/nested config dict; unknown algorithms pass through."""
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
    if _dig(cfg, "optimizer", "boundary_objective") == "softmax_balance":
        return "a5_mimicry_objective"
    if algo == "mmba":
        return "a0_directed"
    if algo == "mmba_unguided":
        return "x1_mmba_unguided"
    if algo == "deepjanus":
        return "a9_deepjanus"
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
    if meta.get("cell_key"):                       # cells/*_cells.yaml: shared by every arm and SUT
        return str(meta["cell_key"])
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


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (ValueError, OSError) as exc:
        logger.warning("unreadable %s: %s", p, exc)
        return None


_POOLS: dict[str, dict[str, dict[str, float]]] = {}


def pool_distance(question: str | None, text: str | None, pool_path: Path | None = None) -> float:
    """MiniLM distance of an edited question from the question pool (0 for the original, NaN when unknown)."""
    if question is None or text is None:
        return float("nan")
    if text.strip() == question.strip():
        return 0.0
    key = str(Path(pool_path or POOL_PATH))
    if key not in _POOLS:
        P = _read_json(Path(key)) or {}
        _POOLS[key] = {q: {x["text"]: float(x["dist"]) for x in e.get("pool", [])}
                       for q, e in (P.get("questions") or {}).items()}
    return float((_POOLS[key].get(question.strip()) or {}).get(text.strip(), float("nan")))


def modality_of(mag_image: float, text_changed: bool | None) -> str | None:
    """image | text | joint | none for a crossing (image edited = mag_image > 0)."""
    if mag_image != mag_image or text_changed is None:
        return None
    img = mag_image > 0
    return "joint" if img and text_changed else "image" if img else "text" if text_changed else "none"


def block_magnitudes(genotype, stats: dict[str, Any]) -> tuple[float, float] | None:
    """(mag_image, mag_text) of a real genotype as in mmb.optimizer.genotype: w_img * RMS(c) and w_txt * t (the text
    block's magnitude gene is its first gene; u is a choice). None when the stats lack the block slices."""
    if genotype is None or not stats.get("image_slice") or not stats.get("text_slice"):
        return None
    g = np.asarray(genotype, dtype=float)
    (a, b), (c, d) = stats["image_slice"], stats["text_slice"]
    img, txt = g[int(a):int(b)], g[int(c):int(d)]
    mi = float(stats.get("w_img") or 1.0) * (float(np.linalg.norm(img) / np.sqrt(img.size)) if img.size else 0.0)
    mt = float(stats.get("w_txt") or 1.0) * (float(txt[0]) if txt.size else 0.0)
    return mi, mt


def min_cross_columns(mc: dict[str, Any] | None, question: str | None, pool_path: Path | None = None,
                      stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """MIN_COLUMNS from a min_cross.json dict (budget-bound: magnitudes NaN unless within_budget). Block magnitudes
    missing from the file (tester branch without a confirmed crossing) are recomputed from the genotype."""
    nan = float("nan")
    out: dict[str, Any] = {c: None for c in MIN_COLUMNS}
    out.update(min_found=mc is not None, magnitude=nan, mag_image=nan, mag_text=nan, min_text_d=nan)
    if mc is None:
        return out
    wb = bool(mc.get("within_budget", mc.get("eval_idx") is not None))
    text = mc.get("text")
    changed = None if text is None or question is None else text.strip() != question.strip()
    out.update(min_within_budget=wb, min_confirmed=mc.get("confirmed"), min_text=text, min_text_changed=changed)
    if wb:
        if mc.get("mag_image") is None and stats is not None:
            bm = block_magnitudes(mc.get("genotype"), stats)
            if bm is not None:
                mc = {**mc, "mag_image": bm[0], "mag_text": bm[1]}
        out.update(min_eval_idx=mc.get("eval_idx"),
                   magnitude=float(mc.get("magnitude", nan)), mag_image=float(mc.get("mag_image", nan)),
                   mag_text=float(mc.get("mag_text", nan)),
                   min_text_d=pool_distance(question, text, pool_path), min_modality=modality_of(
                       float(mc.get("mag_image", nan)), changed))
    return out


def pair_columns(bp: dict[str, Any] | None) -> dict[str, Any]:
    """PAIR_COLUMNS from a boundary_pair.json dict (missing file: bp_source None)."""
    out: dict[str, Any] = {c: None for c in PAIR_COLUMNS}
    if bp is None:
        return out
    out.update(bp_source=bp.get("pair_source"), bp_gap=bp.get("gap"), bp_lam_on=bp.get("lam_on"),
               bp_lam_off=bp.get("lam_off"), bp_kind=bp.get("kind"), bp_text_differs=bp.get("text_differs"),
               bp_eps=bp.get("eps"), nb_extra_calls=bp.get("nb_extra_calls"))
    for side in ("on", "off"):
        s = bp.get(side) or {}
        for k in ("g", "magnitude", "mag_image", "mag_text", "text"):
            out[f"bp_{side}_{k}"] = s.get(k)
    return out


def original_question(stats: dict[str, Any]) -> str | None:
    return (stats.get("seed_metadata") or {}).get("prompt_template") or stats.get("prompt_template")


def load_run(
    run_dir: Path,
    runs_root: Path | None = None,
    defaults: dict[str, Any] | None = None,
    epsilon: float = EPSILON,
    pool_path: Path | None = None,
) -> dict[str, Any]:
    """One output row (a dict with :data:`COLUMNS`) for the run in *run_dir*."""
    run_dir = Path(run_dir)
    runs_root = Path(runs_root) if runs_root is not None else run_dir
    stats = _read_json(run_dir / "stats.json") or {}
    cfg = _load_config(run_dir, stats, runs_root)

    import pyarrow.parquet as pq

    names = set(pq.read_schema(run_dir / "trace.parquet").names)
    want = [c for c in ("generation", "individual", "genotype", "logprobs", "cache_hit") if c in names]
    diag_cols = sorted(c for c in names if c in DIAG_COLUMNS)
    df = pd.read_parquet(run_dir / "trace.parquet", columns=want + diag_cols)
    df = df.sort_values(["generation", "individual"], kind="stable")
    df = df.drop_duplicates(["generation", "individual"], keep="last").reset_index(drop=True)

    n_evals = len(df)
    gens, pop = stats.get("generations"), stats.get("pop_size")
    rm = stats.get("run_meta") or {}
    budget = int(rm["eval_budget"]) if rm.get("eval_budget") else (int(gens) * int(pop) if gens and pop else n_evals)
    dfb = df.iloc[:budget]                         # budget-bound: in-budget evaluations only

    tc = stats.get("target_classes") or (0, 1)
    lp = np.vstack([np.asarray(v, dtype=float) for v in dfb["logprobs"]]) if len(dfb) else np.zeros((0, 2))
    gen = dfb["generation"].to_numpy() if len(dfb) else np.zeros(0, dtype=int)
    anchor_row = None
    if "genotype" in dfb and len(dfb):
        for i, gt in enumerate(dfb["genotype"]):
            if not np.any(np.asarray(gt)):
                anchor_row = i
                break
    anchor_argmax = None
    if stats.get("anchor_argmax") is not None:
        cats = stats.get("categories") or []
        a = stats["anchor_argmax"]
        anchor_argmax = cats.index(a) if isinstance(a, str) and a in cats else (a if isinstance(a, int) else None)
    res = evaluate_trace(
        lp, gen, target_classes=(int(tc[0]), int(tc[1])), anchor_row=anchor_row,
        anchor_g=stats.get("anchor_g"), anchor_argmax=anchor_argmax,
        cache_hit=dfb["cache_hit"].to_numpy() if "cache_hit" in dfb else None, epsilon=epsilon,
    )

    def at_cross(col: str) -> float:
        if res["cross_index"] is None or col not in dfb:
            return float("nan")
        return float(dfb[col].iloc[res["cross_index"]])

    meta = resolve_meta(stats, cfg, defaults)
    question = original_question(stats)
    row = {
        **meta,
        **min_cross_columns(_read_json(run_dir / "min_cross.json"), question, pool_path, stats),
        **pair_columns(_read_json(run_dir / "boundary_pair.json")),
        "seed_idx": stats.get("seed_idx"),
        "algorithm": rm.get("algorithm") or _dig(cfg, "optimizer", "algorithm"),
        "w_img": stats.get("w_img"), "w_txt": stats.get("w_txt"),
        "budget": budget, "evals_total": n_evals, "evals_in_budget": len(dfb),
        # complete = all configured generations are in the trace, or the run stopped early on purpose
        "complete": bool((gens and len(df) and int(df["generation"].max()) + 1 >= int(gens))
                         or "early_stop" in stats or (not gens and n_evals > 0)),
        "anchor_source": res["anchor_source"],
        "crossed": res["crossed"],
        "crossed_at_gen0": res["crossed_at_gen0"],
        "first_cross_eval": res["first_cross_eval"] if res["crossed"] else budget,
        "first_cross_gen": res["first_cross_gen"],
        "first_cross_eval_nocache": res["first_cross_eval_nocache"],
        "n_cross_evals": res["n_cross_evals"],
        "label_flip": res["label_flip"],
        "floor_nats": res["floor_nats"],
        "reached_band": res["reached_band"],
        "img_mag_at_cross": at_cross("diag_img_rms"),
        "txt_mag_at_cross": at_cross("diag_txt_cosdist"),
        "anchor_g": res["anchor_g"],
        "prior_g": float(stats["prior_g"]) if stats.get("prior_g") is not None else float("nan"),
        "probe_gen0": res["probe_gen0"],
        "min_gen0": res["min_gen0"],
        "answer_words_a": len(str(stats.get("class_a", "")).split()) or None,
        "answer_words_b": len(str(stats.get("class_b", "")).split()) or None,
        "question": question,
        "wall_s": stats.get("runtime_seconds", float("nan")),
        "run_dir": str(run_dir),
    }
    return {c: row.get(c) for c in COLUMNS}


def collect(
    runs_root: Path,
    defaults: dict[str, Any] | None = None,
    epsilon: float = EPSILON,
    require_stats: bool = False,
) -> pd.DataFrame:
    """require_stats skips runs without stats.json: a run that is still being written has a trace only."""
    runs_root = Path(runs_root)
    rows = []
    for rd in find_runs(runs_root):
        if require_stats and not (rd / "stats.json").is_file():
            logger.info("skipping %s: run in progress (no stats.json)", rd)
            continue
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
    ap.add_argument("--require-stats", action="store_true", help="skip runs that have no stats.json yet (still running)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    defaults = _parse_kv(args.default)
    if args.exp_id:
        defaults["exp_id"] = args.exp_id
    df = collect(args.runs_root, defaults, args.epsilon, args.require_stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_unknown = int(df["arm"].isna().sum()) if len(df) else 0
    logger.info("wrote %d runs to %s", len(df), args.out)
    if n_unknown:
        logger.warning("%d run(s) have no arm: pass --default arm=<id> or stamp stats.json/config", n_unknown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
