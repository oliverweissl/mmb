"""Image side: continuous low-rank edits of the VQGAN f8-16384 latent (2026-11-28).

    codec        — encode/decode between PIL images and code grids (diagnostics, pilot)
    loading      — VQGAN presets and weights
    latent       — LatentImageManipulator (the ImageBackend used by the pipeline)
    latent_bases — dct / dctmid / bump latent directions
    config       — ImageConfig
"""

from .codec import VQGANCodec
from .config import ImageConfig
from .latent import LatentImageConfig, LatentImageManipulator, LatentManipulationContext
from .types import CodeGrid

__all__ = ["CodeGrid", "ImageConfig", "LatentImageConfig", "LatentImageManipulator",
           "LatentManipulationContext", "VQGANCodec"]
