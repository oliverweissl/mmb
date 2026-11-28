"""Continuous image backend on the VQGAN f8-16384 latent (2026-11-28).

prepare(image): z0 = the QUANTISED latent of the seed (ch x h x w = 4 x 32 x 32); its decode is the
codec reconstruction (baseline_image). Edit, decoded WITHOUT re-quantisation:

    z = z0 + w_img * m_max * ||z0|| * (B c) / sqrt(d),    c in [-1, 1]^d,

B = d orthonormal latent directions (latent_bases: dct | dctmid | bump). Since ||B c|| = ||c||,
||z - z0|| / ||z0|| = w_img * m_max * RMS(c): with w_img = 1 and RMS(c) = 1 the relative latent
magnitude is m_max (default 0.2, the edge of the valid range in the pilot, docs/figs/image_pilot).
c = 0 is the unedited seed. scale(c, lam) = lam * c. The optimizer never sees w_img.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms import functional as TF

from mmb import distlock

from .latent_bases import FAMILIES, build_basis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatentImageConfig:
    """Continuous latent image block (image.backend = vqgan_latent).

    :param basis: dct (default) | dctmid | bump.
    :param d: number of latent directions = image genes (multiple of 4; 0 = no image block, modality text_only).
    :param m_max: relative latent magnitude at w_img = 1, RMS(c) = 1.
    :param w_img: image weight (calibration knob; 1.0 = uncalibrated).
    """

    basis: str = "dct"
    d: int = 16
    m_max: float = 0.2
    w_img: float = 1.0

    def __post_init__(self) -> None:
        if self.basis not in FAMILIES:
            raise ValueError(f"image.latent.basis must be one of {FAMILIES}; got {self.basis!r}")
        if self.d < 0 or self.d % 4:
            raise ValueError(f"image.latent.d must be a multiple of 4 (0 = no image genes, text_only); got {self.d}")
        if self.m_max <= 0:
            raise ValueError(f"image.latent.m_max must be > 0; got {self.m_max}")
        if self.w_img < 0:
            raise ValueError(f"image.latent.w_img must be >= 0; got {self.w_img}")


@dataclass(frozen=True)
class LatentManipulationContext:
    """Per-seed state: quantised latent z0 (ch, h, w), its norm, the basis and the reconstruction."""

    z0: torch.Tensor
    z0_norm: float
    basis: torch.Tensor                     # (d, ch, h, w), orthonormal
    basis_name: str
    reconstruction: Image.Image
    target_class: str | None = None
    candidate_strategy: str = "latent"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def genotype_dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def is_continuous(self) -> bool:
        return True

    @property
    def xl(self) -> NDArray[np.float64]:
        return -np.ones(self.genotype_dim)

    @property
    def xu(self) -> NDArray[np.float64]:
        return np.ones(self.genotype_dim)

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        """Legacy integer surface (preflight / context meta only): 2 levels per gene."""
        return np.full(self.genotype_dim, 2, dtype=np.int64)

    def zero_genotype(self) -> NDArray[np.float64]:
        return np.zeros(self.genotype_dim)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(-1.0, 1.0, self.genotype_dim)


class LatentImageManipulator:
    """ImageBackend with a continuous genotype c in [-1, 1]^d (see module docstring)."""

    is_continuous = True

    def __init__(self, model: torch.nn.Module, config: LatentImageConfig | None = None,
                 device: str | torch.device = "cpu", resolution: int = 256,
                 decode_batch_size: int = 1, preset: str = "f8-16384") -> None:
        # float(): a SUT loading in another thread can switch torch's default dtype while the VQGAN is
        # built, leaving some parameters in bf16 (see vqgan.py encode); the latent path runs in fp32.
        self._model = model.eval().float().to(torch.device(device))
        self._device = torch.device(device)
        self._cfg = config or LatentImageConfig()
        self._resolution = int(resolution)
        self._decode_bs = max(1, int(decode_batch_size))
        self._preset = preset
        self.w_img = float(self._cfg.w_img)
        self._basis_cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def from_config(cls, device: str = "cpu", image_config=None) -> "LatentImageManipulator":
        from .loading import load_vqgan
        from .config import ImageConfig
        cfg = image_config or ImageConfig(backend="vqgan_latent")
        model = load_vqgan(cfg.preset)
        return cls(model, cfg.latent, device=device, resolution=cfg.resolution,
                   decode_batch_size=cfg.decode_batch_size, preset=cfg.preset)

    # ------------------------------------------------------------------ properties
    @property
    def config(self) -> LatentImageConfig:
        return self._cfg

    @property
    def m_max(self) -> float:
        return float(self._cfg.m_max)

    @staticmethod
    def scale(c: NDArray, lam: float) -> NDArray[np.float64]:
        """Image block of scale(g, lam): c -> lam * c."""
        return np.asarray(c, dtype=np.float64) * float(lam)

    # ------------------------------------------------------------------ codec
    def preprocess(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        img = TF.resize(img, self._resolution, interpolation=TF.InterpolationMode.LANCZOS, antialias=True)
        return TF.center_crop(img, [self._resolution, self._resolution])

    def _encode_q(self, image: Image.Image) -> torch.Tensor:
        x = (TF.to_tensor(self.preprocess(image)) * 2 - 1).unsqueeze(0).to(self._device)
        with distlock.lock(str(self._device)), torch.no_grad():
            zq, _, _ = self._model.encode(x)
        return zq[0].float()

    def _decode(self, z: torch.Tensor) -> list[Image.Image]:
        out: list[Image.Image] = []
        for s in range(0, len(z), self._decode_bs):
            with distlock.lock(str(self._device)), torch.no_grad():
                x = self._model.decode(z[s:s + self._decode_bs].float())
                x = x.clamp(-1.0, 1.0).add(1.0).mul(0.5).cpu()
            out.extend(TF.to_pil_image(x[i]) for i in range(len(x)))
        return out

    def _basis(self, ch: int, h: int, w: int) -> torch.Tensor:
        key = (self._cfg.basis, self._cfg.d, ch, h, w)
        if key not in self._basis_cache:
            B = (np.zeros((0, ch, h, w), np.float32) if self._cfg.d == 0
                 else build_basis(self._cfg.basis, self._cfg.d, ch, h, w))
            self._basis_cache[key] = torch.from_numpy(B).to(self._device)
        return self._basis_cache[key]

    # ------------------------------------------------------------------ ImageBackend
    def prepare(self, image: Image.Image, target_class: str | None = None,
                origin_class: str | None = None) -> LatentManipulationContext:
        z0 = self._encode_q(image)
        ch, h, w = z0.shape
        B = self._basis(ch, h, w)
        rec = self._decode(z0[None])[0]
        return LatentManipulationContext(z0=z0, z0_norm=float(z0.norm()), basis=B, basis_name=self._cfg.basis,
                                         reconstruction=rec, target_class=target_class,
                                         meta={"m_max": self.m_max, "d": self._cfg.d, "preset": self._preset})

    def latents(self, ctx: LatentManipulationContext, genotypes: NDArray) -> torch.Tensor:
        """Edited latents (n, ch, h, w) for genotypes (n, d)."""
        C = torch.as_tensor(np.atleast_2d(np.asarray(genotypes, dtype=np.float32)), device=self._device)
        if C.shape[1] != ctx.genotype_dim:
            raise ValueError(f"image genotype length {C.shape[1]} != d = {ctx.genotype_dim}")
        if ctx.genotype_dim == 0:
            return ctx.z0[None].repeat(len(C), 1, 1, 1)
        coef = self.w_img * self.m_max * ctx.z0_norm / np.sqrt(ctx.genotype_dim)
        # Row by row (fix 2026-11-28): a batched einsum / matmul on the GPU can differ from the single-row result
        # in the last bit, which flips grey levels after decoding; every latent must be bit-identical however
        # the population is batched (canonical scoring, docs/DETERMINISM_2026-11-28.md).
        B = ctx.basis.reshape(ctx.genotype_dim, -1)
        delta = torch.stack([(C[i:i + 1] @ B)[0] for i in range(len(C))]).reshape(len(C), *ctx.z0.shape)
        return ctx.z0[None] + coef * delta

    def apply_batch(self, ctx: LatentManipulationContext, genotypes: NDArray) -> list[Image.Image]:
        G = np.atleast_2d(np.asarray(genotypes, dtype=np.float64))
        if len(G) == 0:
            return []
        out: list[Image.Image | None] = [None] * len(G)
        zero = ~np.any(G != 0, axis=1)                    # c = 0: the cached reconstruction
        for i in np.flatnonzero(zero):
            out[i] = ctx.reconstruction.copy()
        idx = np.flatnonzero(~zero)
        if idx.size:
            for i, im in zip(idx, self._decode(self.latents(ctx, G[idx]))):
                out[i] = im
        return out  # type: ignore[return-value]

    def apply(self, ctx: LatentManipulationContext, genotype: NDArray) -> Image.Image:
        return self.apply_batch(ctx, np.asarray(genotype)[None])[0]

    def baseline_image(self, ctx: LatentManipulationContext) -> Image.Image:
        return ctx.reconstruction

    def attach_modal_builder(self, builder: Any) -> None:  # no class targets in the latent backend
        return None

    def precompute_targets(self, target_classes: tuple[str, ...]) -> None:
        return None


__all__ = ["LatentImageConfig", "LatentImageManipulator", "LatentManipulationContext"]
