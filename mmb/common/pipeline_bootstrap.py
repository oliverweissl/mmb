"""Shared component-init + seed-gen + backend-precompute helpers.

The runner (``run_boundary_test.py``) needs to:

1. Load text manipulator, image manipulator (vqgan_latent), and SUT in parallel.
2. Generate the seed pool (``gap_filter``, ``roster`` + combinatorial
   abstraction expansion, ``refcocoplus``, ``slot_items``).

(The backend precompute step for the cone filter and StyleGAN-XL was removed with
those backends, 2026-11-28.)

All helpers take the canonical :class:`~mmb.config.ExperimentConfig`.
Pipelines that use their own config dataclass (PDQ, boundary-pair)
project to ``ExperimentConfig`` before calling.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

from .roster_seed_generator import roster_seeds
from .seed_generator import generate_seeds
from .combinatorial_pair_generator import combinatorial_pairs
from .refcocoplus_seed_generator import refcocoplus_seeds
from .slot_items_seed_generator import slot_items_seeds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared component bundle
# ---------------------------------------------------------------------------


@dataclass
class SharedComponents:
    """All process-wide singletons needed by the per-seed loop.

    Threads share these objects by reference; per-worker wrappers
    (VLMSUT thread wrapper, VLMManipulator, …) hold thin per-seed
    state on top.
    """

    sut: "VLMSUT"  # noqa: F821 — forward, resolved at runtime
    image_manip: "ImageBackend"  # noqa: F821
    text_manip: "CompositeTextManipulator"  # noqa: F821
    data_source: "ImageNetCache"  # noqa: F821


# ---------------------------------------------------------------------------
# Component init
# ---------------------------------------------------------------------------


def init_shared_components(
    exp_cfg: "ExperimentConfig",  # noqa: F821
    data_source: "ImageNetCache",  # noqa: F821
) -> SharedComponents:
    """Parallel load text manipulator, image manipulator, and SUT.

    Three load futures run on a ``ThreadPoolExecutor`` so the heavy
    HuggingFace weight loads overlap.  The SUT future is resolved
    before the image manipulator if the image backend is StyleGAN-XL
    (StyleGAN's origin-precheck needs the SUT).

    Categories are expected to be already resolved on *exp_cfg* — the
    caller is responsible for invoking
    :func:`mmb.config.resolve_categories` against the data source's
    label set before this point.

    :param exp_cfg: Canonical experiment config with resolved categories.
    :param data_source: Open :class:`ImageNetCache`.
    :returns: :class:`SharedComponents` with all three model bundles loaded.
    """
    # Local imports to avoid circular deps at module-import time:
    # mmb.manipulator.* and mmb.sut import from mmb.common in some
    # paths; deferring these breaks the cycle.
    from mmb.manipulator.image_backend import ImageBackend
    from mmb.manipulator.image_factory import build_image_backend
    from mmb.manipulator.text.composite import CompositeTextManipulator
    from mmb.sut import VLMSUT

    pool = ThreadPoolExecutor(max_workers=3)

    logger.info(
        "Composite text manipulator starting (profile=%s)...",
        exp_cfg.text.composite.profile,
    )
    text_fut: Future[CompositeTextManipulator] = pool.submit(
        CompositeTextManipulator.from_config,
        text_config=exp_cfg.text,
        device=exp_cfg.device,
        redis_url=exp_cfg.sut.redis_url,
    )

    # The VQGAN loads in parallel with the SUT (load_vqgan builds it in fp32 before loading the
    # weights, so a concurrent change of torch's default dtype cannot round them).
    logger.info(
        "Image manipulator starting (%s)...  preset=%s basis=%s d=%d m_max=%g",
        exp_cfg.image.backend, exp_cfg.image.preset, exp_cfg.image.latent.basis,
        exp_cfg.image.latent.d, exp_cfg.image.latent.m_max,
    )
    image_fut: Future[ImageBackend] = pool.submit(
        build_image_backend, image_config=exp_cfg.image, device=exp_cfg.device,
    )

    sut_device = (
        exp_cfg.sut.ov_device
        if exp_cfg.sut.backend == "openvino"
        else exp_cfg.device
    )
    logger.info(
        "SUT starting...  %s on %s", exp_cfg.sut.model_id, sut_device,
    )
    sut_fut: Future[VLMSUT] = pool.submit(VLMSUT, exp_cfg)

    sut: VLMSUT = sut_fut.result()
    logger.info("SUT loaded")

    text_manip: CompositeTextManipulator = text_fut.result()
    logger.info("Text manipulator loaded")

    image_manip: ImageBackend = image_fut.result()
    logger.info("Image manipulator loaded")

    pool.shutdown(wait=False)

    return SharedComponents(
        sut=sut,
        image_manip=image_manip,
        text_manip=text_manip,
        data_source=data_source,
    )


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------


def prepare_pipeline_seeds(
    components: SharedComponents,
    exp_cfg: "ExperimentConfig",  # noqa: F821
) -> Sequence:
    """Generate seeds per the config's seed mode.

    * ``gap_filter`` — :func:`mmb.common.generate_seeds`: scans
      ImageNet validation, returns seeds whose VLM logprob-gap meets
      the threshold.
    * ``roster`` — :func:`mmb.common.roster_seeds` collects anchors
      per explicit class, then :func:`mmb.common.combinatorial_pairs`
      expands across abstraction-level cells.
    * ``refcocoplus`` — two-referent grounding items (box-string candidates).
    * ``slot_items`` — :func:`mmb.common.slot_items_seeds`: explicit
      ``(image, carrier, fillers)`` items, one seed each.

    :param components: Shared components (uses ``sut`` + ``data_source``).
    :param exp_cfg: Canonical experiment config.
    :returns: List of :class:`SeedTriple` objects (possibly empty).
    """
    # When PMI is on but not meant to govern seed selection, freeze the
    # generator's anchor scoring to raw so the chosen photos match a raw
    # run exactly (PMIConfig.apply_to_seedgen; Exp-104 Phase B). No-op when
    # PMI is disabled. ``force_raw`` is a scoped override on the SUT.
    if exp_cfg.pmi.enabled and not exp_cfg.pmi.apply_to_seedgen:
        logger.info(
            "PMI enabled but apply_to_seedgen=False → seed generation uses "
            "raw scoring (anchor selection identical to a raw run)."
        )
        with components.sut.force_raw():
            return _generate_seeds_for_mode(components, exp_cfg)
    return _generate_seeds_for_mode(components, exp_cfg)


def _generate_seeds_for_mode(
    components: SharedComponents,
    exp_cfg: "ExperimentConfig",  # noqa: F821
) -> Sequence:
    """Dispatch to the configured seed-generation mode."""
    if exp_cfg.seeds.mode == "roster":
        if exp_cfg.seeds.roster is None:
            raise ValueError(
                "seeds.mode='roster' requires a seeds.roster config block."
            )
        logger.info(
            "Generating seeds (roster: %d classes × %d seeds, "
            "combinatorial abstraction expansion)",
            len(exp_cfg.seeds.roster.class_list),
            exp_cfg.seeds.roster.seeds_per_class,
        )
        seed_images = roster_seeds(components.sut, exp_cfg, components.data_source)
        return combinatorial_pairs(
            seed_images,
            exp_cfg.seeds.roster.class_list,
            exp_cfg.seeds.roster.abstraction,
        )
    if exp_cfg.seeds.mode == "gap_filter":
        logger.info("Generating seeds (gap_filter)")
        return generate_seeds(components.sut, exp_cfg, components.data_source)
    if exp_cfg.seeds.mode == "refcocoplus":
        logger.info("Generating seeds (refcocoplus)")
        return refcocoplus_seeds(components.sut, exp_cfg, components.data_source)
    if exp_cfg.seeds.mode == "slot_items":
        logger.info("Generating seeds (slot_items)")
        return slot_items_seeds(components.sut, exp_cfg, components.data_source)
    raise ValueError(  # pragma: no cover — guarded by SeedConfig.__post_init__
        f"Unknown seeds.mode={exp_cfg.seeds.mode!r}"
    )


__all__ = [
    "SharedComponents",
    "init_shared_components",
    "prepare_pipeline_seeds",
]
