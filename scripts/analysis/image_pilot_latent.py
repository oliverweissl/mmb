#!/usr/bin/env python
"""Step-0 pilot (2026-11-28): continuous, low-rank latent edits with the current VQGAN f8-16384 (candidate A).

Edit: z = z0 + m * ||z0|| * (B c), z0 = quantised latent of the seed (its decode = codec reconstruction), B = d
orthonormal basis vectors of the 32x32x4 latent, c a unit direction in R^d, m the relative magnitude. Decoded without
re-quantisation. Bases: 'dct' (lowest 2-D DCT frequencies x 4 channels) and 'bump' (Gaussian bumps on a coarse grid x
4 channels), d = 16 and 32.
Measures per (image, basis, direction): LPIPS(reconstruction, edit) vs m (smoothness, monotonicity); off-manifold proxy
= LPIPS(edit, codec round trip of the edit) (how far the decode is from what the codec can represent); VRAM, speed.
CPU or GPU, no SUT.  ../.venv/bin/python scripts/analysis/image_pilot_latent.py --out docs/figs/image_pilot
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy.stats import spearmanr

MAGS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)


# dct_basis / bump_basis moved into the package (2026-11-28)
from mmb.manipulator.image.latent_bases import bump_basis, dct_basis  # noqa: E402


def _sk(o):
    """JSON-safe: tuple keys -> 'a/b' strings, recursively."""
    if isinstance(o, dict):
        return {("/".join(map(str, k)) if isinstance(k, tuple) else str(k)): _sk(v) for k, v in o.items()}
    return o


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-img", type=int, default=4); ap.add_argument("--n-dir", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    import torchvision.transforms.functional as TF
    from mmb.manipulator.image.loading import load_vqgan
    from mmb.manipulator.image.codec import VQGANCodec
    from mmb.validity.proxies import ProxyModels
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = load_vqgan("f8-16384").to(dev).eval()
    codec = VQGANCodec(model, device=dev)
    vram_codec = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
    models = ProxyModels(local_files_only=True)

    def encode_q(imgs):
        x = torch.stack([TF.to_tensor(codec.preprocess(i)) * 2 - 1 for i in imgs]).to(dev)
        with torch.no_grad():
            zq, _, _ = model.encode(x)
        return zq                                      # (n, 4, 32, 32)

    def decode(z):
        with torch.no_grad():
            x = model.decode(z.float()).clamp(-1, 1).add(1).mul(0.5)
        return [TF.to_pil_image(x[i].cpu()) for i in range(len(x))]

    imgs = []
    for task in ("imagenet", "pope"):
        items = yaml.safe_load(open(f"cells/{task}_calib.yaml"))["seeds"]["slot_items"]["items"]
        seen = []
        for it in items:
            if it["image"] not in seen:
                seen.append(it["image"])
        imgs += [(task, p) for p in seen[: a.n_img]]
    rng = np.random.default_rng(0)
    bases = {}
    fams = (("dct", dct_basis), ("bump", bump_basis), ("dctmid", lambda h, w, ch, d: dct_basis(h, w, ch, d, skip=15)))
    for fam, fn in fams:
        for d in (16, 32):
            q, _ = np.linalg.qr(fn(32, 32, 4, d)); bases[(fam, d)] = torch.tensor(q.T, dtype=torch.float32, device=dev)
    rows, montage = [], {}
    t_dec, n_dec = 0.0, 0
    for task, path in imgs:
        img = Image.open(Path(path).expanduser()).convert("RGB")
        z0 = encode_q([img])[0]                          # (4, 32, 32)
        # layout (h, w, ch) of the basis -> (ch, h, w)
        rec = decode(z0[None])[0]
        nz = float(z0.norm())
        for (fam, d), B in bases.items():
            for k in range(a.n_dir):
                c = rng.standard_normal(d); c /= np.linalg.norm(c)
                v = (torch.tensor(c, dtype=torch.float32, device=dev) @ B).reshape(32, 32, 4).permute(2, 0, 1)
                zs = torch.stack([z0 + m * nz * v for m in MAGS])
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter(); ims = decode(zs)
                if dev.type == "cuda": torch.cuda.synchronize()
                t_dec += time.perf_counter() - t0; n_dec += len(ims)
                rt = decode(encode_q(ims))                 # codec round trip of each edit
                for m, im, r in zip(MAGS, ims, rt):
                    rows.append(dict(task=task, image=Path(path).name, basis=fam, d=d, dir=k, mag=m,
                                     lpips=float(models.lpips(rec, im)), offman=float(models.lpips(im, r))))
                if k == 0 and d == 16:
                    montage[(task, Path(path).name, fam)] = ims
        print(Path(path).name, "done", flush=True)
    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True); df.to_csv(a.out / "pilot_a.csv", index=False)
    g = df.groupby(["basis", "d", "image", "dir"])
    per = g.apply(lambda s: pd.Series(dict(rho=spearmanr(s["mag"], s["lpips"])[0],
                                           viol=float(np.mean(np.diff(s.sort_values("mag")["lpips"].to_numpy()) < 0)))),
                  include_groups=False)
    summ = dict(
        vram_codec_mib=vram_codec,
        vram_peak_mib=torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        decode_ms_per_image=1000 * t_dec / max(n_dec, 1),
        rho_median=per.groupby(["basis", "d"])["rho"].median().round(3).to_dict(),
        violation_share=per.groupby(["basis", "d"])["viol"].mean().round(3).to_dict(),
        lpips_by_mag=df.groupby(["basis", "d", "mag"])["lpips"].median().round(4).unstack([0, 1]).to_dict(),
        lpips_iqr_by_mag=df.groupby(["basis", "d", "mag"])["lpips"].agg(lambda s: s.quantile(.75) - s.quantile(.25)).round(4).unstack([0, 1]).to_dict(),
        offman_by_mag=df.groupby(["basis", "d", "mag"])["offman"].median().round(4).unstack([0, 1]).to_dict(),
    )
    (a.out / "summary.json").write_text(json.dumps(_sk(summ), indent=1, default=str))
    # montage: rows = image x basis (d=16, first direction), cols = magnitudes
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7})
    names = {"dct": "low-freq DCT", "bump": "coarse bumps", "dctmid": "mid-freq DCT"}
    pick = []
    for task in ("imagenet", "pope"):
        ims_t = sorted({k[1] for k in montage if k[0] == task})[:2]
        pick += [(task, n) for n in ims_t]
    keys = [(t, n, f) for t, n in pick for f in ("dct", "bump", "dctmid")]
    fig, ax = plt.subplots(len(keys), len(MAGS), figsize=(len(MAGS) * 0.95, len(keys) * 0.95))
    for i, kk in enumerate(keys):
        for j, im in enumerate(montage[kk]):
            ax[i, j].imshow(im); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(f"m={MAGS[j]}")
        ax[i, 0].set_ylabel(f"{kk[1][:14]}\n{names[kk[2]]}", fontsize=5.5)
    fig.tight_layout(pad=0.2); fig.savefig(a.out / "montage.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    cols = {"dct": "#0072B2", "bump": "#E69F00", "dctmid": "#009E73"}
    for (fam, d), s in df.groupby(["basis", "d"]):
        for _, g in s.groupby(["image", "dir"]):
            ax[0].plot(g["mag"], g["lpips"], color=cols[fam], linewidth=0.3, alpha=0.25)
        m = s.groupby("mag")["lpips"].median()
        ax[0].plot(m.index, m.values, color=cols[fam], linewidth=1.6, linestyle="-" if d == 16 else "--", label=f"{names[fam]}, d={d}")
        o = s.groupby("mag")["offman"].median()
        ax[1].plot(o.index, o.values, color=cols[fam], linewidth=1.4, linestyle="-" if d == 16 else "--")
    ax[0].set_xlabel("relative latent magnitude m"); ax[0].set_ylabel("LPIPS vs codec reconstruction")
    ax[0].set_title("Image change vs strength (thin: single directions)"); ax[0].legend(frameon=False, fontsize=6)
    ax[1].set_xlabel("relative latent magnitude m"); ax[1].set_ylabel("LPIPS(edit, codec round trip)")
    ax[1].set_title("Off-manifold proxy (median)")
    fig.tight_layout(); fig.savefig(a.out / "curves.png", dpi=160); plt.close(fig)
    print(json.dumps(_sk(summ), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
