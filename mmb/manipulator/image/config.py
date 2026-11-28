"""Image backend settings (continuous genotype, 2026-11-28).

Re-exported via ``mmb.config.ImageConfig``. The only backend is ``"vqgan_latent"``
(:class:`~mmb.manipulator.image.latent.LatentImageManipulator`): continuous low-rank edits of the quantised VQGAN
latent, decoded without re-quantisation. The discrete codebook-swap and StyleGAN-XL backends were removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .latent import LatentImageConfig

BACKENDS = ("vqgan_latent",)


@dataclass(frozen=True)
class ImageConfig:
    """:param backend: ``"vqgan_latent"`` (the only backend).
    :param preset: VQGAN preset (f8-16384: 4 x 32 x 32 latent).
    :param resolution: codec input resolution.
    :param decode_batch_size: decoder sub-batch size. 1 (default) = canonical: every image decoded alone, so a
        render does not depend on the batch (docs/DETERMINISM_2026-11-28.md).
    :param latent: basis, d, m_max, w_img of the latent edit."""

    backend: str = "vqgan_latent"
    preset: str = "f8-16384"
    resolution: int = 256
    decode_batch_size: int = 1
    latent: LatentImageConfig = field(default_factory=LatentImageConfig)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"image.backend must be one of {BACKENDS}; got {self.backend!r} "
                             "(the codebook and StyleGAN backends were removed 2026-11-28)")


__all__ = ["BACKENDS", "ImageConfig"]
