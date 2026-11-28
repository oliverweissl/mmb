"""Tests for VLMManipulator over the continuous blocks (fake VQ latent backend + question-pool text). CPU only."""

import numpy as np
import pytest
from PIL import Image

from conftest import make_vlm_manipulator, write_question_pool

Q = "Is there a dog in the image?"


@pytest.fixture()
def pool(tmp_path):
    return write_question_pool(tmp_path / "pool.json", [Q])


@pytest.fixture()
def img():
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))


@pytest.fixture()
def vlm(pool, img):
    m = make_vlm_manipulator(pool, d=8)
    m.prepare(img, Q)
    return m


def test_dims_and_slices(vlm):
    assert vlm.image_dim == 8 and vlm.text_dim == 2 and vlm.genotype_dim == 10
    assert vlm.image_slice == slice(0, 8) and vlm.text_slice == slice(8, 10)
    assert vlm.is_continuous


def test_unprepared_raises(pool):
    m = make_vlm_manipulator(pool)
    assert not m.is_prepared
    with pytest.raises(RuntimeError, match="prepare"):
        m.manipulate(candidates=None, weights=np.zeros((1, 10)))


def test_zero_genotype_is_seed(vlm):
    g = vlm.zero_genotype()
    assert g.dtype == np.float64 and not g.any()
    ims, txts = vlm.manipulate(candidates=None, weights=g[None])
    assert txts[0] == Q
    assert np.array_equal(np.asarray(ims[0]), np.asarray(vlm.baseline_image()))


def test_blocks_act_independently(vlm):
    G = np.zeros((3, 10))
    G[1, :8] = 0.5
    G[2, 8:] = [0.3, 0.5]
    ims, txts = vlm.manipulate(candidates=None, weights=G)
    assert txts[1] == Q and txts[2] != Q
    assert not np.array_equal(np.asarray(ims[0]), np.asarray(ims[1]))
    assert np.array_equal(np.asarray(ims[0]), np.asarray(ims[2]))


def test_batched_manipulate_length(vlm):
    ims, txts = vlm.manipulate(candidates=None, weights=np.random.default_rng(1).uniform(0, 1, (5, 10)))
    assert len(ims) == len(txts) == 5
