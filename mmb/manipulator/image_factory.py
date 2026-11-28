"""Backend factory: instantiate the image manipulator from the config.

One backend since 2026-11-28: ``"vqgan_latent"`` (:class:`LatentImageManipulator`).
"""

from __future__ import annotations

from mmb.manipulator.image.config import ImageConfig
from mmb.manipulator.image.latent import LatentImageManipulator
from mmb.manipulator.image_backend import ImageBackend


def build_image_backend(*, image_config: ImageConfig, device: str, **_: object) -> ImageBackend:
    """Construct the image backend selected by ``image_config.backend``.

    Extra keyword arguments (categories, sut, ...) are accepted and ignored, for callers
    written against the former multi-backend signature.
    """
    if image_config.backend == "vqgan_latent":
        return LatentImageManipulator.from_config(device=device, image_config=image_config)
    raise ValueError(f"Unknown image.backend={image_config.backend!r} — expected 'vqgan_latent'.")


__all__ = ["build_image_backend"]
