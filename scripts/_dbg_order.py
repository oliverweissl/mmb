
import torch, yaml
from pathlib import Path
from PIL import Image
from mmb.sut.scorer import create_scorer
import sys
sut, q = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
sc = create_scorer(sut, "cuda", enable_thinking=False, load_in_8bit=(q=="int8"), load_in_4bit=(q=="nf4"))
items = yaml.safe_load(open("cells/imagenet_cells.yaml"))["seeds"]["slot_items"]["items"][:6]
for it in items:
    im = Image.open(Path(it["image"]).expanduser()).convert("RGB").resize((256,256))
    a, b = it["fillers"]
    prompt = it["prompt"] + f" Answer with exactly one of these options: {a}, {b}."
    ab = sc.score_categories_tensor(im, prompt, [a, b]).tolist()
    ba = sc.score_categories_tensor(im, prompt, [b, a]).tolist()
    bat = sc.score_batch_tensor([im], [prompt], [a, b])[0].tolist()
    print(f"{a!r:22} {b!r:22} order AB: a={ab[0]:.3f} b={ab[1]:.3f} | order BA: a={ba[1]:.3f} b={ba[0]:.3f} | fullseq: a={bat[0]:.3f} b={bat[1]:.3f}")
