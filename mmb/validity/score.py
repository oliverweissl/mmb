"""Score validity proxies for every crossing run.

Usage::

    python -m mmb.validity.score <runs_root> --out results/validity_<exp>.csv

A crossing run is a directory with ``stats.json``, ``origin.png`` (the unmodified seed at codec
resolution), ``first_cross.png`` and ``first_cross.json`` (``text, full_prompt, g, eval_idx,
generation, genotype``). Runs without ``first_cross.*`` never crossed and get no row.
The original question is ``stats["prompt_template"]``, unless the run records a per-item prompt
(``seed_metadata["prompt_template"]``). The object word for the question check comes from
``first_cross.json["meta"]["object"]``, ``seed_metadata["object"]`` or ``stats["object"]`` (POPE).

One row per crossing run: identifiers (``item_id`` = directory name, run_dir, cell_id, arm, sut, task,
rng_seed), the crossing (first_cross_eval, generation, g) and the proxies of
:mod:`mmb.validity.proxies`. The proxies are descriptive, not validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from mmb.analysis.collect import _load_config, resolve_meta
from mmb.analysis.crossing_files import first_crossing
from mmb.validity.proxies import ProxyModels, image_proxies, text_proxies

logger = logging.getLogger(__name__)

ID_COLUMNS = ["item_id", "run_dir", "cell_id", "arm", "sut", "task", "rng_seed"]
CROSS_COLUMNS = ["first_cross_eval", "first_cross_gen", "g_at_cross"]
IMAGE_COLUMNS = ["img_clip_cos", "img_dino_cos", "img_lpips", "img_ssim"]
TEXT_COLUMNS = ["txt_minilm_cos", "txt_word_edit", "question_preserved", "question_keyword_sim",
                "question_content_kept", "txt_edited",
                "vis_char_dist", "vis_word_edit", "vis_minilm_cos", "vis_identical", "vis_within_one_per_word"]
COLUMNS = ID_COLUMNS + ["view", "n_edits"] + CROSS_COLUMNS + ["img_ref"] + IMAGE_COLUMNS + TEXT_COLUMNS


def find_crossing_runs(root: Path) -> list[Path]:
    out = []
    for dirpath, _d, files in os.walk(root):
        if {"stats.json", "origin.png", "first_cross.png", "first_cross.json"} <= set(files):
            out.append(Path(dirpath))
    return sorted(out)


def object_word(stats: dict, fc: dict) -> str | None:
    for src in ((fc.get("meta") or {}), (stats.get("seed_metadata") or {}), stats, fc):
        if src.get("object"):
            return str(src["object"])
    return None


def original_question(stats: dict) -> str:
    return str((stats.get("seed_metadata") or {}).get("prompt_template") or stats.get("prompt_template") or "")


_CODEC = None
_RECON_CACHE: dict[str, Image.Image] = {}


def _codec():
    global _CODEC
    if _CODEC is None:
        from mmb.manipulator.image.codec import VQGANCodec
        from mmb.manipulator.image.loading import load_vqgan
        _CODEC = VQGANCodec(load_vqgan("f8-16384"), device="cpu")
    return _CODEC


def anchor_reference(run_dir: Path) -> tuple[Image.Image, str]:
    """The unedited input as the SUT sees it: the codec reconstruction of the seed (resize + centre crop to the
    codec resolution, then encode/decode). Runs save it as anchor_recon.png; for older runs it is computed here
    (CPU, cached per origin image). origin.png itself is the raw photograph and is never the reference."""
    run_dir = Path(run_dir)
    saved = run_dir / "anchor_recon.png"
    if saved.exists():
        return Image.open(saved).convert("RGB"), "saved"
    raw = (run_dir / "origin.png").read_bytes()
    key = hashlib.sha1(raw).hexdigest()
    if key not in _RECON_CACHE:
        codec = _codec()
        _RECON_CACHE[key] = codec.decode(codec.encode(Image.open(run_dir / "origin.png").convert("RGB"))).convert("RGB")
    return _RECON_CACHE[key], "computed"


def min_edit_crossing(run_dir: Path) -> dict | None:
    """The search's own minimal-edit crossing among the saved archive (Pareto) members: the crossing evaluation
    with the fewest active genes whose genotype is a saved Pareto member (those are the ones with an image on
    disk); ties by the smaller text cosine distance, then image RMS, then the earlier evaluation. None when no
    archive member crosses. Returns {png, text, eval_idx, n_edits}."""
    import numpy as np
    import pyarrow.parquet as pq
    from mmb.analysis.collect import evaluate_trace

    run_dir = Path(run_dir)
    stats = json.loads((run_dir / "stats.json").read_text())
    names = set(pq.read_schema(run_dir / "trace.parquet").names)
    want = [c for c in ("generation", "individual", "genotype", "logprobs", "decoded_text", "diag_img_rms",
                        "diag_txt_cosdist") if c in names]
    df = pd.read_parquet(run_dir / "trace.parquet", columns=want).sort_values(["generation", "individual"], kind="stable")
    df = df.drop_duplicates(["generation", "individual"], keep="last").reset_index(drop=True)
    if not len(df):
        return None
    tc = stats.get("target_classes") or (0, 1)
    lp = np.vstack([np.asarray(v, dtype=float) for v in df["logprobs"]])
    anchor_row = next((i for i, gt in enumerate(df["genotype"]) if not np.any(np.asarray(gt))), None)
    cats = stats.get("categories") or []
    a = stats.get("anchor_argmax")
    anchor_argmax = cats.index(a) if isinstance(a, str) and a in cats else (a if isinstance(a, int) else None)
    res = evaluate_trace(lp, df["generation"].to_numpy(), target_classes=(int(tc[0]), int(tc[1])), anchor_row=anchor_row,
                         anchor_g=stats.get("anchor_g"), anchor_argmax=anchor_argmax)
    cross = np.asarray(res["_crossing"], dtype=bool)
    arch = {}
    for jf in sorted(run_dir.glob("pareto_*.json")):
        try:
            j = json.loads(jf.read_text())
        except ValueError:
            continue
        if jf.with_suffix(".png").exists():
            arch[tuple(int(x) for x in j["genotype"])] = (jf.with_suffix(".png"), j.get("text"))
    best = None
    for i in np.flatnonzero(cross):
        key = tuple(int(x) for x in df["genotype"].iloc[i])
        if key not in arch:
            continue
        n_edits = int(np.count_nonzero(key))
        tx = float(df["diag_txt_cosdist"].iloc[i]) if "diag_txt_cosdist" in df else 0.0
        im = float(df["diag_img_rms"].iloc[i]) if "diag_img_rms" in df else 0.0
        rank = (n_edits, tx if tx == tx else 9.0, im if im == im else 9.0, int(i))
        if best is None or rank < best[0]:
            best = (rank, i, key)
    if best is None:
        return None
    png, text = arch[best[2]]
    return dict(png=png, text=text or str(df["decoded_text"].iloc[best[1]]), eval_idx=int(best[1]), n_edits=best[0][0])


def minimal_crossing(run_dir: Path) -> tuple[Path, Path] | None:
    """(png, json) of the run's smallest crossing: min_cross.* written by the tester (fewest edits, ties by the
    smaller |g|), else the minimiser's minimal_cross.*; None if neither is stored (the run has no minimal view)."""
    run_dir = Path(run_dir)
    for stem in ("min_cross", "minimal_cross"):
        png, js = run_dir / f"{stem}.png", run_dir / f"{stem}.json"
        if png.exists() and js.exists():
            return png, js
    return None


def score_run(run_dir: Path, models: ProxyModels, root: Path | None = None, defaults: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    stats = json.loads((run_dir / "stats.json").read_text())
    found = first_crossing(run_dir)
    if found is None:
        return []                                  # the run did not cross
    cross_png, cross_json = found
    fc = json.loads(cross_json.read_text())
    cfg = _load_config(run_dir, stats, Path(root) if root else run_dir)
    meta = resolve_meta(stats, cfg, defaults)
    seed_img, ref_kind = anchor_reference(run_dir)
    cross_img = Image.open(cross_png).convert("RGB")
    row = {
        "item_id": run_dir.name, "run_dir": str(run_dir),
        **{k: meta.get(k) for k in ("cell_id", "arm", "sut", "task", "rng_seed")},
        "first_cross_eval": fc.get("eval_idx"), "first_cross_gen": fc.get("generation"),
        "g_at_cross": fc.get("g"), "img_ref": ref_kind,
    }
    row["view"], row["n_edits"] = "first", None
    row.update(image_proxies(models, seed_img, cross_img))
    row.update(text_proxies(models, original_question(stats), str(fc.get("text", "")), object_word(stats, fc)))
    out = [{c: row.get(c) for c in COLUMNS}]
    mc = minimal_crossing(run_dir)
    if mc is not None:                               # primary view: the run's smallest crossing (MMBA's output)
        mpng, mjson = mc
        mj = json.loads(mjson.read_text())
        r2 = dict(row, view="min", first_cross_eval=mj.get("eval_idx"), first_cross_gen=mj.get("generation"),
                  g_at_cross=mj.get("g"))
        r2.update(image_proxies(models, seed_img, Image.open(mpng).convert("RGB")))
        r2.update(text_proxies(models, original_question(stats), str(mj.get("text", "")), object_word(stats, fc)))
        r2["n_edits"] = mj.get("n_edits")
        out.append({c: r2.get(c) for c in COLUMNS})
    return out


def score_runs(root: Path, models: ProxyModels | None = None, defaults: dict | None = None) -> pd.DataFrame:
    models = models or ProxyModels()
    rows = []
    for rd in find_crossing_runs(Path(root)):
        try:
            rows.extend(score_run(rd, models, Path(root), defaults))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s: %s", rd, exc)
    return pd.DataFrame(rows, columns=COLUMNS)


def _parse_kv(items):
    return {k.strip(): v.strip() for k, _, v in (i.partition("=") for i in items or [])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--default", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--offline", action="store_true", help="load models from the local cache only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = score_runs(args.runs_root, ProxyModels(local_files_only=args.offline), _parse_kv(args.default))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %d crossing runs to %s", len(df), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
