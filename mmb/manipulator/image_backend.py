"""Shared protocol for image-manipulation backends.

Since 2026-11-28 the only backend is :class:`mmb.manipulator.image.latent.LatentImageManipulator` (continuous
low-rank edits of the VQGAN latent). The protocol stays so tests and future backends plug into
:class:`VLMManipulator` without branching:

    prepare(image, target_class, origin_class) → context
    apply(ctx, genotype) / apply_batch(ctx, genotypes) → PIL images
    baseline_image(ctx) → the image at the all-zero genotype
    attach_modal_builder / precompute_targets → no-ops for the latent backend
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@runtime_checkable
class ImageManipulationContextLike(Protocol):
    """Subset of context surface that callers depend on.

    :class:`mmb.manipulator.image.latent.LatentManipulationContext` exposes this
    surface; the protocol only enumerates what the runner / VLMManipulator use.
    """

    @property
    def genotype_dim(self) -> int: ...

    @property
    def gene_bounds(self) -> NDArray[np.int64]: ...

    @property
    def target_class(self) -> str | None: ...

    @property
    def candidate_strategy(self) -> str: ...

    def zero_genotype(self) -> NDArray[np.int64]: ...

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]: ...


@runtime_checkable
class ImageBackend(Protocol):
    """Image-manipulation backend surface used by :class:`VLMManipulator` and runners.

    Implementations must be safe to share across worker threads at the model
    level (the underlying generator / codec is loaded once) — per-seed state
    is held in the :class:`ImageManipulationContextLike` returned from
    :meth:`prepare`, which the caller stores. Implementations should not
    mutate shared state from :meth:`apply` / :meth:`apply_batch`.
    """

    def prepare(
        self,
        image: Image.Image,
        target_class: str | None = None,
        origin_class: str | None = None,
    ) -> ImageManipulationContextLike:
        """Encode the seed image and build the per-seed search space.

        ``origin_class`` is the seed's "from" label; required by the
        StyleGAN backend so the pairwise origin-seed cache can be
        looked up. Ignored by VQGAN, which derives its origin from the
        seed image's own latent code.
        """

    def apply(
        self,
        ctx: ImageManipulationContextLike,
        genotype: NDArray[np.int64],
    ) -> Image.Image:
        """Apply one genotype to produce a manipulated image."""

    def apply_batch(
        self,
        ctx: ImageManipulationContextLike,
        genotypes: NDArray[np.int64],
    ) -> list[Image.Image]:
        """Apply N genotypes in one forward (where the backend supports it)."""

    def baseline_image(self, ctx: ImageManipulationContextLike) -> Image.Image:
        """Return the image at the all-zeros genotype.

        This is the reference point for :class:`MatrixDistance` (origin
        side of the κ-interpolation, or the original seed image's
        codec roundtrip for VQGAN). Backends should return it without
        going through the batched manipulate path so callers don't have
        to fabricate a batch of one for what is conceptually a single
        cached image. StyleGAN: the precheck-accepted ``ctx.origin_image``.
        VQGAN: ``decode(ctx.original_grid)`` (or equivalent).
        """

    def attach_modal_builder(self, builder: Any) -> None:
        """Late-bind a class-target builder used by :meth:`precompute_targets`.

        VQGAN: a :class:`ModalTargetBuilder` providing modal codeword grids.
        StyleGAN: the manipulator owns the builder internally; this method
        is a no-op there but kept for ABI symmetry so the runner does not
        branch on backend identity.
        """

    def precompute_targets(self, target_classes: tuple[str, ...]) -> None:
        """Pre-populate any per-class caches needed for the run."""


__all__ = ["ImageBackend", "ImageManipulationContextLike"]
