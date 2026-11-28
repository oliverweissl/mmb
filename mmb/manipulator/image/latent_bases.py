"""Low-rank bases of the VQGAN latent for the continuous image block (2026-11-28).

Moved from scripts/analysis/image_pilot_latent.py (the Step-0 pilot, docs/figs/image_pilot).
A basis is d orthonormal vectors of the (ch, h, w) latent, returned as a (d, ch, h, w) array.

* ``dct``: the lowest 2-D DCT frequencies (DC skipped) x every channel;
* ``dctmid``: the same, 15 frequencies further up (mid frequencies);
* ``bump``: Gaussian bumps on a coarse grid x every channel.
d must be a multiple of the channel count (4 for f8-16384).
"""
from __future__ import annotations

import math

import numpy as np


def dct_basis(h: int, w: int, ch: int, d: int, skip: int = 1) -> np.ndarray:
    """(h*w*ch, d) in (h, w, ch) ravel order, columns not yet orthonormalised (pilot layout)."""
    nf = d // ch
    freqs = sorted(((u, v) for u in range(h) for v in range(w)), key=lambda t: (t[0] + t[1], t))[skip:skip + nf]
    ys, xs = np.arange(h)[:, None], np.arange(w)[None, :]
    out = []
    for u, v in freqs:
        s = np.cos(math.pi * (ys + 0.5) * u / h) * np.cos(math.pi * (xs + 0.5) * v / w)
        for c in range(ch):
            b = np.zeros((h, w, ch)); b[..., c] = s; out.append(b.ravel())
    return np.stack(out, 1)


def _grid(n: int) -> tuple[int, int]:
    gy = int(math.floor(math.sqrt(n)))
    while n % gy:
        gy -= 1
    return gy, n // gy


def bump_basis(h: int, w: int, ch: int, d: int) -> np.ndarray:
    """(h*w*ch, d) Gaussian bumps on a gy x gx grid (gy*gx = d/ch; 2x2 for d=16, 2x4 for d=32)."""
    n = d // ch
    gy, gx = _grid(n)
    sig = h / (2.0 * max(gy, gx))
    ys, xs = np.arange(h)[:, None], np.arange(w)[None, :]
    out = []
    for i in range(gy):
        for j in range(gx):
            cy, cx = (i + 0.5) * h / gy, (j + 0.5) * w / gx
            s = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig ** 2))
            for c in range(ch):
                b = np.zeros((h, w, ch)); b[..., c] = s; out.append(b.ravel())
    return np.stack(out, 1)


FAMILIES = ("dct", "dctmid", "bump")


def build_basis(family: str, d: int, ch: int = 4, h: int = 32, w: int = 32) -> np.ndarray:
    """Orthonormal basis as a (d, ch, h, w) float32 array."""
    if d <= 0 or d % ch:
        raise ValueError(f"basis dimension d={d} must be a positive multiple of the channel count {ch}")
    if family == "dct":
        M = dct_basis(h, w, ch, d)
    elif family == "dctmid":
        M = dct_basis(h, w, ch, d, skip=15)
    elif family == "bump":
        M = bump_basis(h, w, ch, d)
    else:
        raise ValueError(f"unknown latent basis {family!r}; choose from {FAMILIES}")
    q, _ = np.linalg.qr(M)                                  # (h*w*ch, d), orthonormal columns
    return q.T.reshape(d, h, w, ch).transpose(0, 3, 1, 2).astype(np.float32).copy()
