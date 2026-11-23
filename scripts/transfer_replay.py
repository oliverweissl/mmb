"""Transfer of crossings across SUTs (W2 / E10c).

For every run with a minimal crossing, replay it (min_cross.png + full
prompt) and the anchor (VQGAN reconstruction of origin.png + original prompt) on every
OTHER SUT, scored alone (full-sequence teacher forcing). A crossing transfers if, on the
destination SUT, the crossing input has the opposite sign of g to the destination's anchor.

  python scripts/transfer_replay.py runs/MAIN --arms a0_directed a1_random_tiered --out results/transfer_MAIN.csv
One SUT is loaded at a time (16 GB GPU).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from PIL import Image

SUT_IDS = {
    "qwen": ("Qwen/Qwen3.5-4B", False),
    "internvl": ("OpenGVLab/InternVL3_5-4B-HF", False),
    "ministral": ("mistralai/Ministral-3-3B-Instruct-2512", False),
    "llava": ("llava-hf/llava-v1.6-mistral-7b-hf", True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--dst", nargs="+", default=["qwen", "internvl", "ministral"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from mmb.manipulator.image.codec import VQGANCodec
    from mmb.manipulator.image.loading import load_vqgan
    from mmb.sut.scorer import create_scorer

    items = []
    for fc in sorted(a.runs.rglob("min_cross.json")):
        d = fc.parent
        st = json.loads((d / "stats.json").read_text())
        meta = st.get("run_meta", {})
        if a.arms and meta.get("arm") not in a.arms:
            continue
        f = json.loads(fc.read_text())
        cands = list(st["categories"])
        suffix = st["answer_format"].format(categories=", ".join(cands))
        p0 = (st.get("seed_metadata") or {}).get("prompt_template") or st["prompt_template"]
        items.append({"run_dir": str(d), "src_sut": meta.get("sut"), "task": meta.get("task"),
                      "arm": meta.get("arm"), "rng_seed": meta.get("rng_seed"),
                      "cands": cands, "anchor_prompt": p0 + suffix, "cross_prompt": f["full_prompt"]})
    codec = VQGANCodec(load_vqgan("f8-16384"), device=a.device)
    anchors = {it["run_dir"]: codec.decode(codec.encode(Image.open(Path(it["run_dir"]) / "origin.png").convert("RGB")))
               for it in items}
    rows = []
    for dst in a.dst:
        model_id, nf4 = SUT_IDS[dst]
        sc = create_scorer(model_id, a.device, enable_thinking=False, load_in_4bit=nf4)
        for it in items:
            if it["src_sut"] == dst:
                continue
            c = it["cands"]
            ga = sc.score_batch_tensor([anchors[it["run_dir"]]], [it["anchor_prompt"]], c)[0]
            gx = sc.score_batch_tensor([Image.open(Path(it["run_dir"]) / "min_cross.png").convert("RGB")],
                                       [it["cross_prompt"]], c)[0]
            g_a, g_x = float(ga[0] - ga[1]), float(gx[0] - gx[1])
            rows.append({k: it[k] for k in ("run_dir", "src_sut", "task", "arm", "rng_seed")}
                        | {"dst_sut": dst, "g_anchor_dst": g_a, "g_dst": g_x,
                           "flipped_on_dst": (g_a > 0) != (g_x > 0)})
        del sc
        gc.collect()
        torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["run_dir"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} replays -> {a.out}")


if __name__ == "__main__":
    main()
