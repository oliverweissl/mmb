"""Real-coded genotype space shared by every search method.

Contract (C2, 2026-11-28): the genotype is a float64 vector
``[image block c in [-1, 1]^d_img | text block (t, u)]`` where
``t in [0, 1]`` is the text edit magnitude and ``u in [0, 1)`` picks a
candidate inside the distance band that ``t`` selects. The all-zero
genotype is the unedited seed. The manipulator owns the decoding and the
per-modality weights (w_img, w_txt); the optimizer only sees the box
bounds, the two block slices and the manipulator's ``scale(g, lam)``
(image c -> lam*c, text t -> lam*t, u unchanged).

:class:`GenotypeSpace` is the optimizer-side view of that contract. Build it
with :meth:`GenotypeSpace.contract` (synthetic tests, stub manipulators) or
:meth:`GenotypeSpace.from_manipulator` (any object exposing ``xl``/``xu``
or ``bounds``, ``image_slice``, ``text_slice`` and optionally ``scale``).

Magnitudes (what MMBA minimises and the initial sampler stratifies):

* image block: ``w_img * RMS(c)``, RMS(c) = ``||c||_2 / sqrt(d_img)``;
* text block: ``w_txt * t`` (mean of the magnitude genes; ``u`` is a
  choice, not a magnitude);
* genotype: the L2 norm of the weighted block magnitudes (C2 decision
  2026-11-28). w_img / w_txt are read from the manipulator (default 1.0);
  the optimizer never applies them to the genes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

IMAGE, TEXT = "image", "text"


def _as_slice(s) -> slice:
    if s is None:
        return slice(0, 0)
    if isinstance(s, slice):
        return s
    a, b = s
    return slice(int(a), int(b))


@dataclass(frozen=True)
class GenotypeSpace:
    """Bounds, blocks and scale operation of the real-coded genotype.

    :param xl: inclusive lower bound per gene.
    :param xu: upper bound per gene (inclusive unless ``upper_open``).
    :param image_slice: slice of the image block (may be empty).
    :param text_slice: slice of the text block (may be empty).
    :param magnitude_mask: genes that ``scale`` multiplies by lam (image c
        and text t); other genes (text u) are left unchanged.
    :param upper_open: genes whose upper bound is exclusive (text u); the
        sanitiser keeps them strictly below ``xu``.
    :param scale_fn: the manipulator's ``scale(g, lam)``; ``None`` uses
        the contract definition through ``magnitude_mask``.
    :param w_img, w_txt: per-modality weights of the block magnitudes.
    """

    xl: NDArray[np.float64]
    xu: NDArray[np.float64]
    image_slice: slice
    text_slice: slice
    magnitude_mask: NDArray[np.bool_]
    upper_open: NDArray[np.bool_]
    scale_fn: Optional[Callable[[NDArray, float], NDArray]] = field(default=None, compare=False)
    w_img: float = 1.0
    w_txt: float = 1.0

    # ------------------------------------------------------------ builders
    @classmethod
    def contract(cls, d_img: int, text: bool = True, w_img: float = 1.0,
                 w_txt: float = 1.0) -> "GenotypeSpace":
        """The C2 contract: ``[c (d_img) | t, u]``; ``text=False`` drops the text block."""
        d_txt = 2 if text else 0
        n = d_img + d_txt
        xl = np.concatenate([-np.ones(d_img), np.zeros(d_txt)])
        xu = np.ones(n)
        mag = np.zeros(n, bool); mag[:d_img] = True
        opn = np.zeros(n, bool)
        if text:
            mag[d_img] = True      # t
            opn[d_img + 1] = True  # u in [0, 1)
        return cls(xl, xu, slice(0, d_img), slice(d_img, n), mag, opn, None,
                   float(w_img), float(w_txt))

    @classmethod
    def from_manipulator(cls, m) -> "GenotypeSpace":
        """Read the contract from a manipulator (or any duck-typed stub).

        Bounds come from ``m.xl``/``m.xu`` or ``m.bounds`` (``(xl, xu)`` or an
        ``(n, 2)`` array). Text genes: the first is t (scaled), the rest
        are choices (u, not scaled, upper bound exclusive), unless the
        manipulator exposes ``magnitude_mask`` / ``upper_open``.
        """
        if hasattr(m, "xl") and hasattr(m, "xu"):
            xl, xu = m.xl, m.xu
        else:
            b = m.bounds
            if isinstance(b, tuple) and len(b) == 2:
                xl, xu = b
            else:
                b = np.asarray(b, float); xl, xu = b[:, 0], b[:, 1]
        xl = np.asarray(xl, float); xu = np.asarray(xu, float)
        n = len(xl)
        im, tx = _as_slice(m.image_slice), _as_slice(m.text_slice)
        mag = getattr(m, "magnitude_mask", None)
        opn = getattr(m, "upper_open", None)
        if mag is None:
            mag = np.zeros(n, bool); mag[im] = True
            if tx.stop > tx.start:
                mag[tx.start] = True
        if opn is None:
            opn = np.zeros(n, bool)
            if tx.stop - tx.start > 1:
                opn[tx.start + 1:tx.stop] = True
        return cls(xl, xu, im, tx, np.asarray(mag, bool), np.asarray(opn, bool),
                   getattr(m, "scale", None), float(getattr(m, "w_img", 1.0)),
                   float(getattr(m, "w_txt", 1.0)))

    # ------------------------------------------------------------ properties
    @property
    def n_var(self) -> int:
        return len(self.xl)

    @property
    def blocks(self) -> list[tuple[str, slice]]:
        """Non-empty blocks in genotype order."""
        out = []
        for name, s in ((IMAGE, self.image_slice), (TEXT, self.text_slice)):
            if s.stop > s.start:
                out.append((name, s))
        return out

    def block(self, name: str) -> slice:
        return self.image_slice if name == IMAGE else self.text_slice

    def zero(self) -> NDArray[np.float64]:
        return np.zeros(self.n_var)

    # ------------------------------------------------------------ operations
    def sanitize(self, x: NDArray) -> NDArray[np.float64]:
        """Clip to the box; open upper bounds stay strictly below xu."""
        x = np.clip(np.asarray(x, dtype=np.float64), self.xl, self.xu)
        if self.upper_open.any():
            top = np.nextafter(self.xu, -np.inf)
            x[..., self.upper_open] = np.minimum(x[..., self.upper_open], top[self.upper_open])
        return x

    def scale(self, g: NDArray, lam: float) -> NDArray[np.float64]:
        """Global scale: magnitude genes times lam (manipulator's definition if given)."""
        if self.scale_fn is not None:
            return np.asarray(self.scale_fn(np.asarray(g, float), float(lam)), dtype=np.float64)
        out = np.array(g, dtype=np.float64, copy=True)
        out[..., self.magnitude_mask] *= lam
        return out

    def scale_block(self, g: NDArray, lam: float, name: str) -> NDArray[np.float64]:
        """Scale one block only; the other block is left as it is."""
        s = self.block(name)
        out = np.array(g, dtype=np.float64, copy=True)
        out[..., s] = self.scale(g, lam)[..., s]
        return out

    def block_magnitude(self, g: NDArray, name: str) -> NDArray[np.float64] | float:
        """Weighted: image w_img * RMS(c); text w_txt * mean of the magnitude genes (t)."""
        g = np.asarray(g, dtype=np.float64)
        s = self.block(name)
        if s.stop <= s.start:
            return np.zeros(g.shape[:-1]) if g.ndim > 1 else 0.0
        mask = self.magnitude_mask[s]
        v = g[..., s][..., mask]
        if v.shape[-1] == 0:
            return np.zeros(g.shape[:-1]) if g.ndim > 1 else 0.0
        if name == IMAGE:
            return self.w_img * np.sqrt(np.mean(v ** 2, axis=-1))
        return self.w_txt * np.mean(np.abs(v), axis=-1)

    def magnitude(self, g: NDArray) -> NDArray[np.float64] | float:
        """L2 norm of the weighted block magnitudes."""
        parts = [self.block_magnitude(g, b) for b, _ in self.blocks]
        if not parts:
            return 0.0
        return np.sqrt(sum(np.asarray(p) ** 2 for p in parts))
