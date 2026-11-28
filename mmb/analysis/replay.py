"""Replays of stored crossings on the canonical scoring path (transfer, greedy agreement, modality swap).

A run stores the renderings of its genotypes losslessly: anchor_recon.png (the all-zero genotype = the unedited input
as the SUT sees it), min_cross.png/.json (the smallest in-budget crossing; json: text, full_prompt = text + answer
suffix, g, genotype), first_cross.*, min_cross_neighbour.* (the OFF side of the boundary pair). Replaying these files
is replaying the genotypes through the manipulator, without rebuilding the image backend per run.

Scoring is the tester's canonical path (VLMBoundaryTester._canonical_g -> VLMSUT.process_input): ONE input per call,
full-sequence teacher forcing, scorer.score_categories_tensor(image, prompt, categories). g = logp[class_a] - logp[class_b]
with the run's target_classes; an input crosses when sign(g) differs from the anchor's.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

#: SUTs of the study (LLaVA is not in it): name -> (model id, NF4 weight-only)
SUT_IDS = {"qwen": ("Qwen/Qwen3.5-4B", False), "internvl": ("OpenGVLab/InternVL3_5-4B-HF", False),
           "ministral": ("mistralai/Ministral-3-3B-Instruct-2512", False)}
#: crossing views: file stem of the stored crossing
VIEWS = {"min": "min_cross", "first": "first_cross"}
META = ("exp_id", "task", "sut", "arm", "rng_seed")


def load_scorer(sut: str, device: str = "cuda"):
    """The SUT's scorer as the runner builds it (hub kernels off except for Ministral)."""
    from mmb.runners.run_boundary_test import _disable_hub_kernels, _warm_up_numba
    from mmb.sut.scorer import create_scorer
    _warm_up_numba()
    if sut != "ministral":
        _disable_hub_kernels()
    model_id, nf4 = SUT_IDS[sut]
    return create_scorer(model_id, device, enable_thinking=False, load_in_4bit=nf4)


def canonical_g(scorer, image: Image.Image, prompt: str, cands: list[str], target=(0, 1)) -> float:
    lp = scorer.score_categories_tensor(image, prompt, tuple(cands))
    lp = np.asarray(lp.float().cpu().numpy() if hasattr(lp, "cpu") else lp, dtype=float).ravel()
    return float(lp[target[0]] - lp[target[1]])


def crossing_item(run_dir: Path, view: str = "min") -> dict | None:
    """Everything a replay needs for one stored crossing (None when the run has no such crossing)."""
    d = Path(run_dir)
    js, png = d / f"{VIEWS[view]}.json", d / f"{VIEWS[view]}.png"
    if not (js.exists() and png.exists() and (d / "stats.json").exists() and (d / "anchor_recon.png").exists()):
        return None
    st = json.loads((d / "stats.json").read_text())
    fc = json.loads(js.read_text())
    text, full = str(fc.get("text") or ""), str(fc.get("full_prompt") or "")
    q0 = (st.get("seed_metadata") or {}).get("prompt_template") or st.get("prompt_template") or ""
    suffix = full[len(text):] if full.startswith(text) else st.get("answer_format", "").format(
        categories=", ".join(st.get("categories") or []))
    meta = st.get("run_meta") or {}
    return dict(run_dir=str(d), view=view, stats=st, **{k: meta.get(k) for k in META},
                cands=list(st.get("categories") or []), target=tuple(st.get("target_classes") or (0, 1)),
                q0=q0, text=text, anchor_prompt=q0 + suffix, cross_prompt=text + suffix,
                anchor_png=d / "anchor_recon.png", cross_png=png, g_search=fc.get("g"),
                g_anchor_search=st.get("anchor_g"))


def items(roots: Iterable[Path], view: str = "min", arms=None, sut: str | None = None) -> list[dict]:
    out = []
    for r in roots:
        for js in sorted(Path(r).rglob(f"{VIEWS[view]}.json")):
            it = crossing_item(js.parent, view)
            if it is None or (arms and it["arm"] not in arms) or (sut and it["sut"] not in (None, sut)):
                continue
            out.append(it)
    return out


def _img(p) -> Image.Image:
    return Image.open(p).convert("RGB")


def transfer_rows(its: list[dict], scorer, dst: str) -> list[dict]:
    """Every crossing of another SUT replayed on dst: anchor and crossing scored alone; transfers when the signs differ."""
    rows = []
    for it in its:
        if it["sut"] == dst:
            continue
        ga = canonical_g(scorer, _img(it["anchor_png"]), it["anchor_prompt"], it["cands"], it["target"])
        gx = canonical_g(scorer, _img(it["cross_png"]), it["cross_prompt"], it["cands"], it["target"])
        rows.append(dict(run_dir=it["run_dir"], src_sut=it["sut"], task=it["task"], arm=it["arm"], rng_seed=it["rng_seed"],
                         dst_sut=dst, g_anchor_dst=ga, g_dst=gx, flipped_on_dst=bool((ga > 0) != (gx > 0))))
    return rows


def match_answer(answer: str, cands: list[str]) -> str:
    """Map a free-text answer to one candidate ('<ambiguous>' / '<other>' otherwise)."""
    import re
    a = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
    hits = [c for c in cands if re.search(r"\b" + re.escape(c.lower()) + r"\b", a)]
    if len(hits) == 1:
        return hits[0]
    starts = [c for c in cands if a.startswith(c.lower())]
    return starts[0] if len(starts) == 1 else ("<ambiguous>" if hits else "<other>")


def agreement_row(it: dict, scorer) -> dict:
    """Greedy answer vs the teacher-forced argmax, for the anchor and the crossing; re-verification of the crossing."""
    out = {"run_dir": it["run_dir"], **{k: it[k] for k in META}}
    for tag, png, prompt in (("anchor", it["anchor_png"], it["anchor_prompt"]), ("cross", it["cross_png"], it["cross_prompt"])):
        img = _img(png)
        ans, _, _ = scorer.generate(img, prompt)
        lp = scorer.score_categories_tensor(img, prompt, tuple(it["cands"]))
        lp = np.asarray(lp.float().cpu().numpy() if hasattr(lp, "cpu") else lp, dtype=float).ravel()
        res = dict(zip(it["cands"], lp.tolist()))
        tf = max(res, key=res.get)
        gm = match_answer(ans, it["cands"])
        ia, ib = it["target"]
        out.update({f"{tag}_greedy": ans, f"{tag}_greedy_label": gm, f"{tag}_tf_argmax": tf, f"{tag}_agree": gm == tf,
                    f"{tag}_g_single": float(lp[ia] - lp[ib])})
    out["cross_verified"] = (out["anchor_g_single"] > 0) != (out["cross_g_single"] > 0)
    out["cross_g_search"] = it["g_search"]
    return out


def classify(a_cross: bool, b_cross: bool) -> str:
    """Swap class of a verified crossing: (a) original image + edited question, (b) edited image + original question."""
    return {(False, True): "image", (True, False): "text", (True, True): "either", (False, False): "joint"}[(a_cross, b_cross)]


def swap_row(it: dict, scorer, label: str = "") -> dict:
    """Modality attribution by swapping, all inputs scored alone on the canonical path: anchor, crossing, (a) original
    image + edited question, (b) edited image + original question. Because the image depends only on the image block and
    the question only on the text block, (a) and (b) are exactly the renderings of the crossing's genotype with the image
    block, resp. the text block, set to zero."""
    img0, img1 = _img(it["anchor_png"]), _img(it["cross_png"])
    p0, p1 = it["anchor_prompt"], it["cross_prompt"]
    g = lambda im, p: canonical_g(scorer, im, p, it["cands"], it["target"])
    g0, gx, ga, gb = g(img0, p0), g(img1, p1), g(img0, p1), g(img1, p0)
    side = np.sign(g0)
    crosses = lambda x: bool(np.sign(x) != side)
    verified = crosses(gx)
    a0 = np.asarray(img0, dtype=np.int16)
    return dict(run_dir=it["run_dir"], view=it["view"], label=label, sut=it["sut"],
                **{k: it[k] for k in ("exp_id", "task", "arm", "rng_seed")}, seed_idx=it["stats"].get("seed_idx"),
                g_anchor=g0, g_anchor_search=it["g_anchor_search"], g_cross=gx, g_cross_search=it["g_search"],
                g_text_only=ga, g_image_only=gb, text_edited=p1 != p0,
                image_edited=bool(np.abs(np.asarray(img1, dtype=np.int16) - a0).max() > 0) if img1.size == img0.size else True,
                verified=verified, a_crosses=crosses(ga), b_crosses=crosses(gb),
                cls=classify(crosses(ga), crosses(gb)) if verified else "unverified",
                rel_text_only=ga / g0 if g0 else np.nan, rel_image_only=gb / g0 if g0 else np.nan,
                rel_cross=gx / g0 if g0 else np.nan)


def write_csv(rows: list[dict], out: Path) -> None:
    import csv
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["run_dir"])
        w.writeheader()
        w.writerows(rows)
