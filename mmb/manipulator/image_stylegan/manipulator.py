"""StyleGAN-XL image manipulator (Protocol-compatible image backend).

The class :class:`StyleGANImageManipulator` is the parallel of
:class:`mmb.manipulator.image.ImageManipulator` for the StyleGAN backend.
Lifecycle mirrors VQGAN's two-phase API exactly:

    prepare(image, target_class) → StyleGANManipulationContext
    apply(ctx, genotype)         → PIL.Image
    apply_batch(ctx, genotypes)  → list[PIL.Image]

Internally it wraps a SMOO :class:`StyleGANManipulator` plus a
:class:`StyleGANClassTargetBuilder` that owns the per-class modal-w
and accepted-origin-seed caches (L1 dict + L2 Redis bytes).

Critical: this backend ignores the *seed image* (it cannot invert it
into w-space). The synthetic origin used for the run is the image that
the SUT precheck accepted; the input ``image`` is only retained for
the seed-pipeline upstream API parity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from mmb.manipulator.image.manipulator import ImageConfig, StyleGANConfig

from .class_target import StyleGANClassTargetBuilder, tensor_to_pil
from .types import StyleGANManipulationContext

logger = logging.getLogger(__name__)


class StyleGANImageManipulator:
    """StyleGAN-XL backend implementing :class:`ImageBackend`.

    Construction requires a fully-loaded SMOO ``StyleGANManipulator``
    plus a class-target builder that already has the SUT wired in. The
    runner is responsible for constructing both before reaching the
    per-seed loop.

    :param smoo_manipulator: A SMOO
        :class:`smoo.manipulator.style_gan_manipulator.StyleGANManipulator`
        wrapping the generator.
    :param class_target_builder: Owns modal-w + accepted-origin caches.
    :param config: Image-config dataclass.
    :param device: Torch device for context tensors.
    :param categories: Run-wide SUT contrast set, kept for trace
        provenance (the class_target_builder already holds it).
    """

    def __init__(
        self,
        *,
        smoo_manipulator: Any,
        class_target_builder: StyleGANClassTargetBuilder,
        config: ImageConfig,
        device: torch.device,
        categories: tuple[str, ...],
    ) -> None:
        if config.backend != "stylegan_xl":
            raise ValueError(
                f"StyleGANImageManipulator requires image.backend="
                f"'stylegan_xl'; got {config.backend!r}"
            )
        self._smoo = smoo_manipulator
        self._builder = class_target_builder
        self._config = config
        self._stylegan_cfg: StyleGANConfig = config.stylegan
        self._device = device
        self._categories = tuple(categories)
        self._num_ws = int(self._infer_num_ws())

    # ------------------------------------------------------------------
    # Generator introspection
    # ------------------------------------------------------------------

    def _infer_num_ws(self) -> int:
        """Read ``num_ws`` off the underlying generator.

        SMOO's ``StyleGANManipulator`` stores the generator on
        ``_generator``. Newer NVlabs StyleGAN-XL generators always
        expose ``num_ws`` on the synthesis network and on ``G_ema``
        itself.
        """
        gen = getattr(self._smoo, "_generator", None)
        if gen is None:
            raise RuntimeError(
                "Underlying SMOO StyleGANManipulator does not expose "
                "_generator; cannot read num_ws."
            )
        if hasattr(gen, "num_ws"):
            return int(gen.num_ws)
        if hasattr(gen, "synthesis") and hasattr(gen.synthesis, "num_ws"):
            return int(gen.synthesis.num_ws)
        raise RuntimeError(
            "StyleGAN generator does not expose num_ws — cannot derive "
            "genome dimension."
        )

    @property
    def num_ws(self) -> int:
        """Number of style layers = genome dimension."""
        return self._num_ws

    @property
    def config(self) -> ImageConfig:
        return self._config

    @property
    def class_target_builder(self) -> StyleGANClassTargetBuilder:
        return self._builder

    # ------------------------------------------------------------------
    # ImageBackend protocol — modal-target lifecycle
    # ------------------------------------------------------------------

    def attach_modal_builder(self, builder: Any) -> None:
        """No-op for symmetry with the VQGAN backend.

        StyleGAN already owns its class-target builder internally; the
        runner does not need (or get) to swap it in. This method exists
        only so the :class:`ImageBackend` protocol is satisfied without
        backend-specific branches in the runner.
        """
        return

    def precompute_targets(
        self,
        target_classes: tuple[str, ...],
        origin_pairs: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        """Pre-populate the modal-w cache (targets) and origin-seed cache.

        ``target_classes`` are the L0 names appearing in seed metadata's
        ``target_class_concrete`` (or ``class_b`` for gap-filter seeds) —
        these drive modal-w precompute.

        ``origin_pairs`` is the explicit list of ``(origin_class,
        target_class)`` pairs the run will evaluate. The pairwise precheck
        finds, for each pair, the smallest ``seed_int`` whose synthetic
        origin-class image gets a higher SUT log-prob for the origin than
        for the target — guaranteeing the κ=0 endpoint sits on the origin
        side of *that specific* decision boundary. When ``origin_pairs``
        is omitted the precheck is skipped (modal-w only).

        For StyleGAN the precompute is more substantial than for VQGAN:
        the precheck may run dozens of SUT calls per pair on a cold cache.
        After this call returns, every per-seed :meth:`prepare` is
        L1-hit-and-tensor-copy.
        """
        if not target_classes and not origin_pairs:
            return
        logger.info(
            "StyleGAN precompute: %d target class(es), %d origin pair(s)",
            len(target_classes), len(origin_pairs) if origin_pairs else 0,
        )
        # Build modal-w first — fast, no SUT calls.
        self._builder.populate_modal_w(target_classes)
        # Then pairwise origin precheck — slow on cold cache.
        if origin_pairs:
            self._builder.populate_origins(origin_pairs)

    # ------------------------------------------------------------------
    # ImageBackend protocol — per-seed lifecycle
    # ------------------------------------------------------------------

    def prepare(
        self,
        image: Image.Image,
        target_class: str | None = None,
        origin_class: str | None = None,
    ) -> StyleGANManipulationContext:
        """Build a per-seed manipulation context.

        StyleGAN cannot invert the supplied ``image`` to w-space, so the
        seed image is ignored for w computation; the synthetic origin
        accepted by the precheck is used instead. The ``image`` argument
        is retained on the protocol so the upstream
        :class:`VLMManipulator.prepare` call site does not need to
        branch on backend.

        :param image: The seed image (ignored for w computation; held
            externally by the upstream caller for trace purposes).
        :param target_class: Required. The class κ→1 interpolates toward.
        :param origin_class: Optional. The class the accepted origin
            seed must match. Defaults to ``target_class`` when not given
            (consistent with the "anchor is the seed's class" usage in
            roster-mode runs). Pass explicitly when origin/anchor labels
            diverge from the target.
        :raises ValueError: If ``target_class`` is not provided.
        """
        if target_class is None:
            raise ValueError(
                "StyleGANImageManipulator.prepare() requires target_class."
            )
        if origin_class is None or origin_class == target_class:
            raise ValueError(
                "StyleGANImageManipulator.prepare() requires an origin_class "
                f"distinct from target_class={target_class!r}; got "
                f"origin_class={origin_class!r}. The pairwise precheck has no "
                "boundary to test when origin and target coincide."
            )

        target_w = self._builder.ensure_modal_w(target_class).to(self._device)
        _, origin_w, origin_image = self._builder.ensure_origin(
            origin_class, target_class,
        )
        origin_w = origin_w.to(self._device)

        return StyleGANManipulationContext(
            origin_w=origin_w,
            target_w=target_w,
            origin_image=origin_image,
            origin_class=origin_class,
            target_class=target_class,
            kappa_quant_levels=int(self._stylegan_cfg.kappa_quant_levels),
            num_ws=self._num_ws,
        )

    # ------------------------------------------------------------------
    # Apply (single + batch)
    # ------------------------------------------------------------------

    def apply(
        self,
        ctx: StyleGANManipulationContext,
        genotype: NDArray[np.int64],
    ) -> Image.Image:
        """Single-image apply — unsupported on this backend.

        SMOO's :meth:`StyleGANManipulator.manipulate` asserts via
        ``len(cond[1])``, which IndexErrors at batch=1. Padding to two and
        discarding a render is wasteful; the live code paths always have
        batches available (optimizer generations, Pareto save, etc.).
        Callers needing the κ=0 image should use :meth:`baseline_image`;
        general batched needs go through :meth:`apply_batch` with the
        full batch.
        """
        raise NotImplementedError(
            "StyleGANImageManipulator.apply() is unsupported (SMOO's "
            "manipulate requires batch >= 2). Use apply_batch with the "
            "full batch, or baseline_image() for the κ=0 reference."
        )

    def baseline_image(self, ctx: StyleGANManipulationContext) -> Image.Image:
        """Return the κ=0 reference image without re-rendering.

        The precheck already produced the synthetic origin image and
        stored it on the context; the all-zeros genotype interpolates
        from origin_w (with weight 1) to target_w (with weight 0), which
        is exactly the origin. No synthesis call needed.
        """
        return ctx.origin_image

    def apply_batch(
        self,
        ctx: StyleGANManipulationContext,
        genotypes: NDArray[np.int64],
    ) -> list[Image.Image]:
        """Apply ``B`` quantised-κ genotypes in one synthesis forward.

        Per-layer κ = ``gene[l] / kappa_quant_levels`` ∈ ``[0, 1]``.
        Construction follows SMOO's interpolation contract: build a
        ``MixCandidateList`` with one w0 candidate (origin) and one wn
        candidate (target), then call ``manipulate`` with
        ``cond = zeros(batch, num_ws)`` (single wn slot) and
        ``weights = κ``.
        """
        if len(genotypes) == 0:
            return []

        # Per-layer lerp between origin and target W+ latents:
        #   w_l = (1 - kappa_l) * w_origin + kappa_l * w_target
        Q = ctx.kappa_quant_levels
        if Q < 1:
            raise ValueError(f"kappa_quant_levels must be >= 1, got {Q}")
        kappa = torch.as_tensor(
            np.asarray(genotypes, dtype=np.float32) / float(Q), device=self._device
        )[..., None]  # (B, num_ws, 1)
        w0 = ctx.origin_w.to(self._device)  # (1, num_ws, w_dim)
        wn = ctx.target_w.to(self._device)
        w = w0 + kappa * (wn - w0)
        batch_size = len(genotypes)
        # Returned shape is (B, C, H, W) in [0, 1].
        imgs_tensor = self._smoo.get_images(w)
        return [tensor_to_pil(imgs_tensor[i]) for i in range(batch_size)]


__all__ = ["StyleGANImageManipulator"]
