"""Check that batched teacher-forced scoring equals the sequential KV-cache path.

  python scripts/check_batched_scoring.py --sut llava-hf/llava-v1.6-mistral-7b-hf --int8 --n 24
Prints max |diff| of the per-candidate normalised log-likelihoods and of g, and the speed-up.
"""
import argparse
import random
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from mmb.manipulator.image.codec import VQGANCodec
from mmb.manipulator.image.loading import load_vqgan
from mmb.sut.scorer import create_scorer

ap = argparse.ArgumentParser()
ap.add_argument("--sut", required=True)
ap.add_argument("--int8", action="store_true")
ap.add_argument("--nf4", action="store_true")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--cells", default="cells/imagenet_cells.yaml")
a = ap.parse_args()

items = yaml.safe_load(open(a.cells))["seeds"]["slot_items"]["items"]
rng = random.Random(0)
codec = VQGANCodec(load_vqgan("f8-16384"), device="cuda")
sc = create_scorer(a.sut, "cuda", enable_thinking=False, load_in_8bit=a.int8, load_in_4bit=a.nf4)
sc.batch_chunk = a.chunk
fill = ["please", "the", "image", "carefully", "kindly", "now", "photo"]
imgs, prompts, cands = [], [], None
for k in range(a.n):
    it = items[k % len(items)]
    cands = cands or it["fillers"]
    im = codec.decode(codec.encode(Image.open(Path(it["image"]).expanduser()).convert("RGB")))
    extra = " ".join(rng.choice(fill) for _ in range(rng.randint(0, 6)))
    prompts.append((it["prompt"] + " " + extra).strip() + " Answer with exactly one of these options: " + ", ".join(it["fillers"]) + ".")
    imgs.append(im)
cands = items[0]["fillers"]
torch.cuda.synchronize(); t = time.time()
seq = torch.stack([sc.score_batch_tensor([i], [p], cands)[0] for i, p in zip(imgs, prompts)])
torch.cuda.synchronize(); t_seq = time.time() - t
t = time.time()
bat = sc.score_batch_tensor(imgs, prompts, cands)
torch.cuda.synchronize(); t_bat = time.time() - t
d = (seq - bat).abs()
g_seq = seq[:, 0] - seq[:, 1]
g_bat = bat[:, 0] - bat[:, 1]
print(f"{a.sut}: n={a.n} max|dlp|={d.max():.4g} mean={d.mean():.3g} max|dg|={(g_seq-g_bat).abs().max():.4g} "
      f"sign agree={(torch.sign(g_seq)==torch.sign(g_bat)).float().mean():.3f} "
      f"t_seq={t_seq:.1f}s t_batch={t_bat:.1f}s speedup={t_seq/t_bat:.2f}x")
