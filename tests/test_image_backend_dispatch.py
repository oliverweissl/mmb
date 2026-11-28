"""Image backend dispatch after the 2026-11-28 trim: ImageConfig accepts only vqgan_latent; the latent
backend satisfies the ImageBackend protocol; build_context_meta records the latent fields."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import make_vlm_manipulator, write_question_pool
from mmb.common.seed_context import build_context_meta
from mmb.manipulator.image.config import ImageConfig
from mmb.manipulator.image.latent import LatentImageManipulator
from mmb.manipulator.image_backend import ImageBackend
from mmb.manipulator.image_factory import build_image_backend


def test_default_backend_is_latent():
    assert ImageConfig().backend == "vqgan_latent"


@pytest.mark.parametrize("name", ["vqgan_codebook", "stylegan_xl", "bogus"])
def test_removed_backends_rejected(name):
    with pytest.raises(ValueError, match="vqgan_latent"):
        ImageConfig(backend=name)


def test_dacite_yaml_ignores_legacy_keys():
    from mmb.runners.run_boundary_test import load_config
    cfg = load_config({"image": {"backend": "vqgan_latent", "latent": {"d": 32, "basis": "bump"},
                                 "region_block": 4, "cone_filter": {"enabled": False}}})
    assert cfg.image.latent.d == 32 and cfg.image.latent.basis == "bump"


def test_latent_backend_satisfies_protocol():
    from conftest import make_latent_manipulator
    assert isinstance(make_latent_manipulator(), ImageBackend)
    assert issubclass(LatentImageManipulator, object)


def test_factory_builds_latent(monkeypatch):
    import mmb.manipulator.image.loading as loading
    from conftest import FakeVQ
    monkeypatch.setattr(loading, "load_vqgan", lambda preset: FakeVQ())
    m = build_image_backend(image_config=ImageConfig(), device="cpu")
    assert isinstance(m, LatentImageManipulator) and m.config.d == 16


def test_context_meta_latent(tmp_path):
    q = "Is there a dog in the image?"
    m = make_vlm_manipulator(write_question_pool(tmp_path / "p.json", [q]), d=8)
    m.prepare(Image.fromarray(np.zeros((64, 64, 3), np.uint8)), q)
    meta = build_context_meta(m)
    assert meta["image_backend"] == "vqgan_latent" and meta["image_basis"] == "dct" and meta["image_d"] == 8
    assert meta["image_latent_meta"]["m_max"] == 0.2
