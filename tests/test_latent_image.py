"""Continuous latent image backend (vqgan_latent) and the VLMManipulator real-coded contract. CPU, fake VQ model."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from mmb.manipulator.image.latent import LatentImageConfig, LatentImageManipulator
from mmb.manipulator.image.latent_bases import build_basis
from mmb.manipulator.image.config import ImageConfig
from mmb.manipulator.text.operators.question_pool import QuestionPoolOperator, load_pool
from mmb.manipulator.vlm_manipulator import VLMManipulator
from mmb.optimizer.genotype import GenotypeSpace

D_POOL = (0.0, 0.012, 0.03, 0.045, 0.051, 0.08, 0.12, 0.2)
Q = "Is there a dog in the image?"


class FakeVQ(torch.nn.Module):
    """encode: 8x average pool -> (n, 4, 32, 32) (4th channel = mean); decode: upsample back, tanh."""

    def encode(self, x):
        p = F.avg_pool2d(x, 8)
        z = torch.cat([p, p.mean(1, keepdim=True)], 1)
        return z, None, None

    def decode(self, z):
        return torch.tanh(F.interpolate(z[:, :3], scale_factor=8, mode="nearest"))


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))


@pytest.fixture
def latent():
    return LatentImageManipulator(FakeVQ(), LatentImageConfig(d=16, m_max=0.2), device="cpu", decode_batch_size=3)


@pytest.fixture
def pool(tmp_path):
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"meta": {"d_max": 0.25, "n_bands": 10}, "questions": {Q: {"pool": [
        {"text": f"Q{i} dog image?", "dist": float(x)} for i, x in enumerate(D_POOL[1:], 1)]}}}))
    load_pool.cache_clear()
    yield str(p)
    load_pool.cache_clear()


def test_config_validation_and_backend_name():
    assert ImageConfig(backend="vqgan_latent").latent == LatentImageConfig()
    for bad in (dict(d=6), dict(d=-4), dict(basis="pca"), dict(m_max=0.0)):
        with pytest.raises(ValueError):
            LatentImageConfig(**bad)


@pytest.mark.parametrize("fam", ["dct", "dctmid", "bump"])
def test_basis_orthonormal(fam):
    B = build_basis(fam, 16).reshape(16, -1)
    assert np.allclose(B @ B.T, np.eye(16), atol=1e-5)


def test_edit_formula_relative_magnitude(latent, img):
    ctx = latent.prepare(img)
    assert ctx.genotype_dim == 16 and np.all(ctx.xl == -1) and np.all(ctx.xu == 1)
    rng = np.random.default_rng(1)
    C = rng.uniform(-1, 1, (5, 16))
    Z = latent.latents(ctx, C)
    rel = ((Z - ctx.z0[None]).flatten(1).norm(dim=1) / ctx.z0_norm).numpy()
    rms = np.sqrt((C ** 2).mean(1))
    assert np.allclose(rel, 0.2 * rms, rtol=1e-4)            # = w_img * m_max * RMS(c)
    latent.w_img = 0.5
    rel2 = ((latent.latents(ctx, C) - ctx.z0[None]).flatten(1).norm(dim=1) / ctx.z0_norm).numpy()
    assert np.allclose(rel2, 0.1 * rms, rtol=1e-4)
    assert np.allclose(latent.scale(C[0], 0.3), 0.3 * C[0])


def test_zero_is_reconstruction_and_batching(latent, img):
    ctx = latent.prepare(img)
    rec = np.asarray(latent.baseline_image(ctx))
    G = np.vstack([np.zeros(16), np.random.default_rng(2).uniform(-1, 1, (6, 16))])
    ims = latent.apply_batch(ctx, G)
    assert len(ims) == 7 and np.array_equal(np.asarray(ims[0]), rec)
    assert not np.array_equal(np.asarray(ims[1]), rec)
    for i in range(1, 7):                                     # chunked batch == single decodes
        assert np.array_equal(np.asarray(ims[i]), np.asarray(latent.apply(ctx, G[i])))


def _vlm(latent, pool):
    import spacy
    from mmb.manipulator.text.composite import CompositeTextManipulator
    cm = CompositeTextManipulator(spacy.blank("en"), [QuestionPoolOperator(pool=pool)])
    return VLMManipulator(latent, cm)


def test_vlm_contract(latent, pool, img):
    m = _vlm(latent, pool)
    m.prepare(img, Q)
    assert m.is_continuous and m.image_slice == slice(0, 16) and m.text_slice == slice(16, 18)
    assert np.array_equal(m.xl, np.r_[-np.ones(16), 0, 0]) and np.array_equal(m.xu, np.ones(18))
    assert m.magnitude_mask.tolist() == [True] * 17 + [False]
    assert m.upper_open.tolist() == [False] * 17 + [True]
    g = np.r_[np.full(16, 0.5), 0.8, 0.3]
    s = m.scale(g, 0.5)
    assert np.allclose(s[:16], 0.25) and s[16] == pytest.approx(0.4) and s[17] == pytest.approx(0.3)
    assert np.allclose(m.scale(np.stack([g, g]), 0.5), np.stack([s, s]))
    m.w_img, m.w_txt = 0.7, 1.5
    assert latent.w_img == 0.7 and m.w_img == 0.7 and m.w_txt == 1.5
    sp = GenotypeSpace.from_manipulator(m)
    assert sp.w_img == 0.7 and sp.w_txt == 1.5 and sp.n_var == 18
    assert sp.block_magnitude(g, "text") == pytest.approx(1.5 * 0.8)
    assert m.zero_genotype().dtype == np.float64


def test_vlm_manipulate_keeps_floats(latent, pool, img):
    m = _vlm(latent, pool)
    m.prepare(img, Q)
    G = np.zeros((3, 18))
    G[1, 16:] = [0.2, 0.5]                                    # text only (int cast would give t = 0)
    G[2, :16] = 0.4                                           # image only (int cast would give c = 0)
    ims, txts = m.manipulate(candidates=None, weights=G)
    assert txts[0] == Q and txts[1].startswith("Q") and txts[1] != Q and txts[2] == Q
    assert np.array_equal(np.asarray(ims[0]), np.asarray(ims[1]))
    assert not np.array_equal(np.asarray(ims[0]), np.asarray(ims[2]))


def test_tester_end_to_end_with_latent_backend(latent, pool, img, tmp_path):
    """MMBA through the real VLMBoundaryTester on the latent backend + question_pool text."""
    from mmb.config import ExperimentConfig, SeedTriple
    from mmb.evolutionary.vlm_boundary_tester import VLMBoundaryTester
    from mmb.objectives import CriterionCollection, build_criteria
    from mmb.optimizer.mmba import MMBAOptimizer

    class SUT:
        text_embedder = None
        device_str = "cpu"
        last_call_cached = False
        cache_stats = {"hits": 0, "misses": 0}

        def __init__(self):
            self.ref = None

        def process_input(self, image, text=None, categories=None):
            a = np.asarray(image, dtype=np.float32)
            if self.ref is None:
                self.ref = a
            dev = float(np.abs(a - self.ref).mean()) / 255.0 + (0.02 if text and text.startswith("Q") else 0.0)
            return torch.tensor([-0.1 - 40 * dev, -1.0, -5.0])

    cfg = ExperimentConfig(categories=("cat_a", "cat_b", "cat_c"), prompt_template=Q,
                           answer_format=" from: {categories}.", generations=6, pop_size=10,
                           save_dir=tmp_path / "runs", name="lat")
    crit = build_criteria("joint", "logprob_gap", True)
    objs = CriterionCollection(*[c() for c in crit])
    opt = MMBAOptimizer(space=None, num_objectives=objs.num_objectives, pop_size=10, seed=0)
    seed = SeedTriple(image=img, class_a="cat_a", class_b="cat_b")
    VLMBoundaryTester(sut=SUT(), manipulator=_vlm(latent, pool), optimizer=opt, objectives=objs,
                      config=cfg).test([seed])
    run = next((tmp_path / "runs").iterdir())
    stats = json.load(open(run / "stats.json"))
    assert stats["genotype"] == "real" and stats["image_slice"] == [0, 16] and stats["text_slice"] == [16, 18]
    ctx = json.load(open(run / "context.json"))
    assert ctx["image_backend"] == "vqgan_latent" and ctx["image_d"] == 16
    if (run / "min_cross.json").exists():
        bp = json.load(open(run / "boundary_pair.json"))
        assert bp["pair_source"] == "own_phase2" and bp["on"]["g"] < 0 <= bp["off"]["g"]


def test_text_only_modality_drops_image_block(img):
    from mmb.config import ExperimentConfig, apply_modality
    exp = apply_modality(ExperimentConfig(modality="text_only"))
    assert exp.image.latent.d == 0
    m = LatentImageManipulator(FakeVQ(), exp.image.latent, device="cpu")
    ctx = m.prepare(img)
    assert ctx.genotype_dim == 0
    ims = m.apply_batch(ctx, np.zeros((3, 0)))
    assert len(ims) == 3 and np.array_equal(np.asarray(ims[0]), np.asarray(ctx.reconstruction))
