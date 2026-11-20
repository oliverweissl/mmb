"""E-agree: does the teacher-forced argmax agree with the SUT's greedy answer?

For every run with a first crossing, replay the crossing input (first_cross.png +
full prompt) and the anchor (VQGAN reconstruction of origin.png + original prompt)
through greedy generation, and map the free-text answer to the candidate set.

  python scripts/agreement_replay.py runs/E1 --sut llava --out results/agree_E1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image

SUT_IDS = {
    "llava": ("llava-hf/llava-v1.6-mistral-7b-hf", True),
    "qwen": ("Qwen/Qwen3.5-4B", False),
    "internvl": ("OpenGVLab/InternVL3_5-4B-HF", False),
}


def match(answer: str, cands: list[str]) -> str:
    a = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
    hits = [c for c in cands if re.search(r"\b" + re.escape(c.lower()) + r"\b", a)]
    if len(hits) == 1:
        return hits[0]
    starts = [c for c in cands if a.startswith(c.lower())]
    return starts[0] if len(starts) == 1 else ("<ambiguous>" if hits else "<other>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--sut", required=True, choices=list(SUT_IDS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from mmb.manipulator.image.codec import VQGANCodec
    from mmb.manipulator.image.loading import load_vqgan
    from mmb.sut.scorer import create_scorer

    model_id, int8 = SUT_IDS[a.sut]
    sc = create_scorer(model_id, a.device, enable_thinking=False, load_in_8bit=int8)
    codec = VQGANCodec(load_vqgan("f8-16384"), device=a.device)
    rows = []
    dirs = sorted(p.parent for p in a.runs.rglob("first_cross.json"))
    for d in dirs:
        st = json.loads((d / "stats.json").read_text())
        if st.get("run_meta", {}).get("sut") not in (None, a.sut):
            continue
        fc = json.loads((d / "first_cross.json").read_text())
        cands = list(st["categories"])
        suffix = st["answer_format"].format(categories=", ".join(cands))
        prompt0 = (st.get("seed_metadata") or {}).get("prompt_template") or st["prompt_template"]
        anchor_img = codec.decode(codec.encode(Image.open(d / "origin.png").convert("RGB")))
        out = {"run_dir": str(d), **{k: st.get("run_meta", {}).get(k) for k in ("exp_id", "task", "sut", "arm", "rng_seed")}}
        for tag, img, prompt in (("anchor", anchor_img, prompt0 + suffix),
                                 ("cross", Image.open(d / "first_cross.png").convert("RGB"), fc["full_prompt"])):
            ans, _, _ = sc.generate(img, prompt)
            res = {lbl: lpn for (lbl, _lp, lpn, _n) in sc.score_categories(img, prompt, cands)}
            tf = max(res, key=res.get)
            gm = match(ans, cands)
            out.update({f"{tag}_greedy": ans, f"{tag}_greedy_label": gm, f"{tag}_tf_argmax": tf,
                        f"{tag}_agree": gm == tf})
        rows.append(out)
        if a.limit and len(rows) >= a.limit:
            break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["run_dir"])
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    if n:
        for tag in ("anchor", "cross"):
            k = sum(r[f"{tag}_agree"] for r in rows)
            print(f"{tag}: greedy == teacher-forced argmax in {k}/{n} ({k / n:.1%})")


if __name__ == "__main__":
    main()
