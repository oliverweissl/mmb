"""Multi-modal VLM manipulator wrapping image + composite-text sub-manipulators.

Bridges the two-phase (prepare / apply) lifecycle of the individual
manipulators with SMOO's ``Manipulator`` interface. The optimizer works
with a single concatenated genotype ``[image_genes | text_genes]`` and
this class splits and dispatches.

The image side is duck-typed via :class:`ImageBackend`
(:class:`mmb.manipulator.image.latent.LatentImageManipulator` since 2026-11-28).

Real-coded genotype contract (2026-11-28), used by the search
(:meth:`mmb.optimizer.genotype.GenotypeSpace.from_manipulator`): ``xl``,
``xu``, ``image_slice``, ``text_slice``, ``scale(g, lam)``, ``w_img``,
``w_txt`` (settable, forwarded to the sub-manipulators), ``magnitude_mask``,
``upper_open``. Genotypes stay float64 end to end when a sub-manipulator is
continuous (latent image backend, question_pool text); legacy integer
sub-manipulators still receive int64 genes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from mmb.core import Manipulator

from .image_backend import ImageBackend, ImageManipulationContextLike
from .text.composite import CompositeTextManipulator

if TYPE_CHECKING:
    from .text.composite import CompositeManipulationContext


class VLMManipulator(Manipulator):
    """Multi-modal manipulator wrapping image + composite-text sub-manipulators.

    Lifecycle::

        manipulator = VLMManipulator(image_manip, text_manip)
        manipulator.prepare(seed_image, seed_text)

        # In optimizer loop:
        images, texts = manipulator.manipulate(candidates=None, weights=genotypes)

    The ``image_manipulator`` argument is typed as
    :class:`ImageBackend` (a :class:`typing.Protocol`); either the VQGAN
    :class:`ImageManipulator` or the StyleGAN
    :class:`StyleGANImageManipulator` may be passed.
    """

    def __init__(
        self,
        image_manipulator: ImageBackend,
        text_manipulator: CompositeTextManipulator,
    ) -> None:
        self._image: ImageBackend = image_manipulator
        self._text = text_manipulator
        self._image_ctx: ImageManipulationContextLike | None = None
        self._text_ctx: CompositeManipulationContext | None = None

    # -- lifecycle -----------------------------------------------------------

    def prepare(
        self,
        image: Image.Image,
        text: str,
        exclude_words: frozenset[str] | None = None,
        target_class: str | None = None,
        origin_class: str | None = None,
    ) -> None:
        """Prepare both manipulators for a seed (image, text) pair.

        Call once per seed.

        :param image: Seed PIL image.
        :param text: Seed prompt text.
        :param exclude_words: Words to protect from text mutation
            (case-insensitive). Typically the category labels so the
            optimizer cannot trivially remove the correct answer.
        :param target_class: Seed's target class; recorded in the image
            context's metadata (the latent backend does not use it).
        :param origin_class: Seed's "from" class; ignored by the latent backend.
        """
        self._image_ctx = self._image.prepare(
            image, target_class=target_class, origin_class=origin_class,
        )
        self._text_ctx = self._text.prepare(text, exclude_words=exclude_words)

    # -- properties ----------------------------------------------------------

    @property
    def is_prepared(self) -> bool:
        return self._image_ctx is not None and self._text_ctx is not None

    @property
    def genotype_dim(self) -> int:
        return self._image_ctx.genotype_dim + self._text_ctx.genotype_dim

    @property
    def image_dim(self) -> int:
        return self._image_ctx.genotype_dim

    @property
    def text_dim(self) -> int:
        return self._text_ctx.genotype_dim

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        return np.concatenate([
            self._image_ctx.gene_bounds,
            self._text_ctx.gene_bounds,
        ])

    # -- real-coded genotype contract ----------------------------------------

    @property
    def image_continuous(self) -> bool:
        return bool(getattr(self._image, "is_continuous", False))

    @property
    def text_continuous(self) -> bool:
        return self._text_ctx is not None and (self._text_ctx.genotype_dim == 0
                                               or bool(getattr(self._text_ctx, "is_continuous", False)))

    @property
    def is_continuous(self) -> bool:
        """True when both blocks take real-valued genes (an empty text block counts as continuous)."""
        return self.image_continuous and self.text_continuous

    @property
    def image_slice(self) -> slice:
        return slice(0, self.image_dim)

    @property
    def text_slice(self) -> slice:
        return slice(self.image_dim, self.genotype_dim)

    @property
    def xl(self) -> NDArray[np.float64]:
        img = (np.asarray(self._image_ctx.xl, float) if hasattr(self._image_ctx, "xl")
               else np.zeros(self.image_dim))
        txt = np.asarray(self._text_ctx.xl, float) if self.text_dim else np.empty(0)
        return np.concatenate([img, txt])

    @property
    def xu(self) -> NDArray[np.float64]:
        img = (np.asarray(self._image_ctx.xu, float) if hasattr(self._image_ctx, "xu")
               else np.asarray(self._image_ctx.gene_bounds, float) - 1.0)
        txt = np.asarray(self._text_ctx.xu, float) if self.text_dim else np.empty(0)
        return np.concatenate([img, txt])

    def _text_masks(self) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """(magnitude, upper_open) over the text block: per continuous operator block the first gene is the
        magnitude (t), the others are choices (u, upper-open); integer genes are magnitudes (rank = strength)."""
        ctx = self._text_ctx
        mag = getattr(ctx, "magnitude_mask", None)
        opn = getattr(ctx, "upper_open", None)
        if mag is not None and opn is not None:
            return np.asarray(mag, bool), np.asarray(opn, bool)
        cont = tuple(getattr(ctx, "op_continuous", ()))
        mags, opns = [], []
        for i, d in enumerate(ctx.op_gene_dims):
            if d == 0:
                continue
            is_c = bool(cont[i]) if i < len(cont) else False
            m = np.zeros(d, bool); o = np.zeros(d, bool)
            if is_c:
                m[0] = True; o[1:] = True
            else:
                m[:] = True
            mags.append(m); opns.append(o)
        if not mags:
            return np.empty(0, bool), np.empty(0, bool)
        return np.concatenate(mags), np.concatenate(opns)

    @property
    def magnitude_mask(self) -> NDArray[np.bool_]:
        return np.concatenate([np.ones(self.image_dim, bool), self._text_masks()[0]])

    @property
    def upper_open(self) -> NDArray[np.bool_]:
        return np.concatenate([np.zeros(self.image_dim, bool), self._text_masks()[1]])

    def scale(self, g: NDArray, lam: float) -> NDArray[np.float64]:
        """scale(g, lam): image block via the backend (c -> lam c), text block via the composite
        (t -> lam t, u unchanged). Accepts one genotype or a batch."""
        G = np.asarray(g, dtype=np.float64)
        one = G.ndim == 1
        G = np.atleast_2d(G).copy()
        d = self.image_dim
        img_scale = getattr(self._image, "scale", None)
        for r in range(len(G)):
            G[r, :d] = img_scale(G[r, :d], lam) if img_scale is not None else lam * G[r, :d]
            if self.text_dim:
                G[r, d:] = self._text.scale(G[r, d:], lam, self._text_ctx)
        return G[0] if one else G

    @property
    def w_img(self) -> float:
        return float(getattr(self._image, "w_img", 1.0))

    @w_img.setter
    def w_img(self, value: float) -> None:
        if not hasattr(self._image, "w_img"):
            raise AttributeError(f"{type(self._image).__name__} has no image weight w_img")
        self._image.w_img = float(value)

    @property
    def w_txt(self) -> float:
        try:
            return float(self._text.w_txt)
        except AttributeError:
            return 1.0

    @w_txt.setter
    def w_txt(self, value: float) -> None:
        self._text.w_txt = float(value)

    @property
    def image_manipulator(self) -> ImageBackend:
        """Underlying image sub-manipulator. Exposed for samplers that
        need access to backend-specific structures (e.g. VQGAN codebook
        for embedding-FPS init). Callers must isinstance-check before
        using backend-specific surfaces."""
        return self._image

    @property
    def image_context(self) -> ImageManipulationContextLike:
        return self._image_ctx

    def baseline_image(self) -> Image.Image:
        """Return the image at the all-zeros genotype.

        Delegates to the underlying image backend's
        :meth:`ImageBackend.baseline_image`. StyleGAN returns the
        precheck-cached origin image (no synthesis); VQGAN runs a
        single decode (no batched manipulate needed).
        """
        if self._image_ctx is None:
            raise RuntimeError("Call prepare() before baseline_image().")
        return self._image.baseline_image(self._image_ctx)

    @property
    def text_context(self) -> CompositeManipulationContext:
        return self._text_ctx

    # -- SMOO Manipulator interface ------------------------------------------

    def manipulate(self, candidates=None, *, weights, **kwargs):
        """Apply genotypes to produce mutated (images, texts) pairs.

        :param candidates: Unused (contexts stored from ``prepare()``).
        :param weights: ``NDArray`` of shape ``(pop_size, genotype_dim)``
            with integer genotypes. The first ``image_dim`` genes drive
            the image; the remaining ``text_dim`` drive the text.
        :returns: ``(images, texts)`` lists.
        """
        if not self.is_prepared:
            raise RuntimeError("Call prepare() before manipulate().")

        # Real-coded blocks keep float64 genes; legacy integer blocks get int64.
        weights = np.atleast_2d(np.asarray(weights))
        img_genes = weights[:, : self.image_dim]
        txt_genes = weights[:, self.image_dim:]
        img_genes = (img_genes.astype(np.float64) if self.image_continuous
                     else img_genes.astype(np.int64))
        # the composite casts the blocks of its integer operators itself
        txt_genes = (txt_genes.astype(np.float64) if self.text_continuous and self.text_dim
                     else txt_genes.astype(np.int64))

        # One VQGAN forward for the whole population (was N batch-1 calls).
        images = self._image.apply_batch(self._image_ctx, img_genes)
        # Text apply is a candidate-table lookup — keep sequential.
        texts = [
            self._text.apply(self._text_ctx, txt_genes[i])
            for i in range(len(weights))
        ]
        return images, texts

    def get_images(self, z):
        raise NotImplementedError(
            "VLMManipulator produces (images, texts) via manipulate(). "
            "Use manipulate() instead."
        )

    # -- genotype helpers ----------------------------------------------------

    def zero_genotype(self) -> NDArray:
        cont = self.image_continuous or (self.text_dim and self.text_continuous)
        return np.zeros(self.genotype_dim, dtype=np.float64 if cont else np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]:
        return np.concatenate([
            self._image_ctx.random_genotype(rng),
            self._text_ctx.random_genotype(rng),
        ])
