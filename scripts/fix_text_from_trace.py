"""Repair min_cross / min_cross_neighbour text fields from the trace.

Text renderings were process-dependent before the stable-hash fix, so text re-rendered
outside the run's process may differ from what the search scored. The trace stores the
exact scored text (decoded_text). This script:
- sets min_cross.json text/full_prompt from the trace row it refers to;
- neighbour: if the removed edit is an IMAGE gene, its text equals the crossing's text;
  if it is a TEXT gene, its text is taken from the trace when that genotype was evaluated
  there, otherwise the neighbour is removed (cannot be reproduced) and must be re-scored.
  python scripts/fix_text_from_trace.py runs/MAIN
"""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

root = Path(sys.argv[1]); fixed = dropped = kept = 0
for mcp in sorted(root.rglob("min_cross.json")):
    d = mcp.parent
    mc = json.loads(mcp.read_text())
    rows = pq.read_table(d / "trace.parquet", columns=["genotype", "decoded_text"]).to_pylist()
    row = rows[mc["eval_idx"] - 1]
    if list(row["genotype"]) != list(mc["genotype"]):
        print("genotype mismatch", d); continue
    suffix = mc["full_prompt"][len(mc["text"]):]
    if row["decoded_text"] != mc["text"]:
        mc["text"] = row["decoded_text"]; mc["full_prompt"] = row["decoded_text"] + suffix
        mc["text_from_trace"] = True; mcp.write_text(json.dumps(mc, indent=2)); fixed += 1
    nbp = d / "min_cross_neighbour.json"
    if not nbp.exists():
        continue
    nb = json.loads(nbp.read_text())
    ctx = json.loads((d / "context.json").read_text())
    n_img = len(ctx.get("image_patch_positions") or [])
    G, H = np.array(mc["genotype"]), np.array(nb["genotype"])
    diff = np.nonzero(G != H)[0]
    if len(diff) == 1 and diff[0] < n_img:
        new_text = mc["text"]
    else:
        hit = [r for r in rows if list(r["genotype"]) == list(H)]
        new_text = hit[0]["decoded_text"] if hit else None
    if new_text is None:
        nbp.unlink(); (d / "min_cross_neighbour.png").unlink(missing_ok=True); dropped += 1
    else:
        if nb.get("text") != new_text:
            nb["text"] = new_text; nb["full_prompt"] = new_text + suffix; nb["text_from_trace"] = True
            nbp.write_text(json.dumps(nb, indent=2))
        kept += 1
print(f"min_cross texts fixed {fixed}; neighbours kept {kept}, dropped {dropped}")
