"""Tests for VLMBoundaryTester.

Uses the continuous blocks over fakes (fake VQ latent backend + question-pool text over a
temporary pool) and a fake SUT — no real VLM or VQGAN is loaded.  The full optimisation loop runs with tiny parameters
(pop_size=5, generations=3) to verify wiring and data flow.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from mmb.manipulator.vlm_manipulator import VLMManipulator
from mmb.objectives import (
    CriterionCollection,
    MatrixDistance,
    TargetedBalance,
    TextEmbeddingDistance,
)
from mmb.optimizer.real_pymoo_optimizer import RealPymooOptimizer
from mmb.config import ExperimentConfig, SeedTriple
from mmb.evolutionary.vlm_boundary_tester import (
    VLMBoundaryTester,
    pil_to_tensor,
)

from conftest import make_vlm_manipulator, write_question_pool


# ---------------------------------------------------------------------------
# Test-local constants and fakes
# ---------------------------------------------------------------------------

CATEGORIES = ("cat_a", "cat_b", "cat_c")
PROMPT_TEMPLATE = "Classify this"
ANSWER_FORMAT = " from: {categories}."


class FakeSUT:
    """Returns deterministic logprobs favouring cat_a."""

    text_embedder = None
    device_str = "cpu"
    last_call_cached = False
    cache_stats = {"hits": 0, "misses": 0}

    def process_input(self, image, text=None, categories=None):
        return torch.tensor([-0.1, -2.0, -5.0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manipulator(tmp_path):
    """VLMManipulator: latent image block (d=8) + question-pool text block (2 genes)."""
    return make_vlm_manipulator(write_question_pool(tmp_path / "pool.json", [PROMPT_TEMPLATE]), d=8)


class CrossingSUT(FakeSUT):
    """Prefers cat_b once the image moves far enough from the first image it scores (the anchor)."""

    def __init__(self):
        self.ref = None

    def process_input(self, image, text=None, categories=None):
        a = np.asarray(image, dtype=np.float32)
        if self.ref is None:
            self.ref = a
        dev = float(np.abs(a - self.ref).mean()) / 255.0
        return torch.tensor([-0.1 - 60.0 * dev, -1.0, -5.0])


@pytest.fixture()
def objectives():
    """3 live batched objectives (MatrixDistance, TextEmbeddingDistance, TargetedBalance)."""
    return CriterionCollection(
        MatrixDistance(),
        TextEmbeddingDistance(),
        TargetedBalance(),
    )


@pytest.fixture()
def config(tmp_path):
    return ExperimentConfig(
        categories=CATEGORIES,
        prompt_template=PROMPT_TEMPLATE,
        answer_format=ANSWER_FORMAT,
        generations=3,
        pop_size=5,
        save_dir=tmp_path / "runs",
        name="test_exp",
    )


@pytest.fixture()
def seed():
    return SeedTriple(
        image=Image.new("RGB", (8, 8), (128, 64, 32)),
        class_a="cat_a",
        class_b="cat_b",
    )


# ---------------------------------------------------------------------------
# TestUtilities
# ---------------------------------------------------------------------------


class TestUtilities:
    def testpil_to_tensor_shape(self):
        img = Image.new("RGB", (16, 8))
        t = pil_to_tensor(img)
        assert t.shape == (3, 8, 16)  # C, H, W

    def testpil_to_tensor_range(self):
        img = Image.new("RGB", (4, 4), (255, 128, 0))
        t = pil_to_tensor(img)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def testpil_to_tensor_grayscale(self):
        img = Image.new("L", (4, 4))
        t = pil_to_tensor(img)
        assert t.shape == (1, 4, 4)


# ---------------------------------------------------------------------------
# TestVLMBoundaryTester
# ---------------------------------------------------------------------------


class TestVLMBoundaryTester:
    def _make_tester(self, config, manipulator, objectives):
        optimizer = RealPymooOptimizer(
            space=None,
            num_objectives=3,
            pop_size=config.pop_size,
        )
        return VLMBoundaryTester(
            sut=FakeSUT(),
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            config=config,
        )

    def test_full_loop_runs(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

    def test_trace_parquet_exists(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dirs = list(config.save_dir.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "trace.parquet").exists()

    def test_trace_row_count(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        import pandas as pd

        run_dir = next(config.save_dir.iterdir())
        df = pd.read_parquet(run_dir / "trace.parquet")
        expected = config.generations * config.pop_size
        assert len(df) == expected

    def test_trace_has_all_fitness_columns(
        self, config, manipulator, objectives, seed,
    ):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        import pandas as pd

        run_dir = next(config.save_dir.iterdir())
        df = pd.read_parquet(run_dir / "trace.parquet")

        fitness_cols = [c for c in df.columns if c.startswith("fitness_")]
        assert len(fitness_cols) == 3

    def test_trace_has_required_columns(
        self, config, manipulator, objectives, seed,
    ):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        import pandas as pd

        run_dir = next(config.save_dir.iterdir())
        df = pd.read_parquet(run_dir / "trace.parquet")

        required = {
            "seed_id", "generation", "individual",
            "genotype", "logprobs", "decoded_text",
            "predicted_class", "p_class_a", "p_class_b",
        }
        assert required.issubset(set(df.columns))

    def test_stats_json(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        with open(run_dir / "stats.json") as f:
            stats = json.load(f)

        assert stats["class_a"] == "cat_a"
        assert stats["class_b"] == "cat_b"
        assert stats["generations"] == config.generations
        assert stats["pop_size"] == config.pop_size

    def test_context_json(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        with open(run_dir / "context.json") as f:
            ctx = json.load(f)

        assert ctx["image_backend"] == "vqgan_latent"
        assert "text_original_tokens" in ctx
        assert "text_op_order" in ctx

    def test_origin_image_saved(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        assert (run_dir / "origin.png").exists()

    def test_pareto_candidates_saved(
        self, config, manipulator, objectives, seed,
    ):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        pareto_pngs = list(run_dir.glob("pareto_*.png"))
        pareto_jsons = list(run_dir.glob("pareto_*.json"))
        assert len(pareto_pngs) > 0
        assert len(pareto_pngs) == len(pareto_jsons)

    def test_two_seeds_produce_two_dirs(
        self, config, manipulator, objectives, seed,
    ):
        seed2 = SeedTriple(
            image=Image.new("RGB", (8, 8), (0, 0, 0)),
            class_a="cat_a",
            class_b="cat_c",
        )
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed, seed2])

        run_dirs = list(config.save_dir.iterdir())
        assert len(run_dirs) == 2


class TestBaselineArms:
    """E1/E2 arms run end-to-end through the tester."""

    @pytest.mark.parametrize(
        "algorithm,objective,proximity_only",
        [
            ("ga", "logprob_gap", True),
            ("agemoea2", "softmax_balance", False),
            ("random", "logprob_gap", False),
            ("nsga2", "logprob_gap", False),
        ],
    )
    def test_arm_runs(self, config, manipulator, seed, algorithm, objective, proximity_only):
        from mmb.objectives import build_criteria

        crit = build_criteria("joint", objective, proximity_only)
        objs = CriterionCollection(*[c() for c in crit])
        optimizer = RealPymooOptimizer(
            space=None,
            num_objectives=objs.num_objectives,
            pop_size=config.pop_size,
            algorithm=algorithm,
        )
        VLMBoundaryTester(
            sut=FakeSUT(), manipulator=manipulator, optimizer=optimizer,
            objectives=objs, config=config,
        ).test([seed])
        assert len(list(config.save_dir.rglob("trace.parquet"))) == 1


# ---------------------------------------------------------------------------
# Real-coded genotype integration (2026-11-28)
# ---------------------------------------------------------------------------


class TestRealCodedIntegration:
    def _run(self, config, manipulator, optimizer, objectives, seed, sut=None):
        VLMBoundaryTester(sut=sut or FakeSUT(), manipulator=manipulator, optimizer=optimizer,
                          objectives=objectives, config=config).test([seed])
        return next(config.save_dir.iterdir())

    def test_float_genotypes_in_trace_and_stats(self, config, manipulator, objectives, seed):
        import pandas as pd
        opt = RealPymooOptimizer(space=None, num_objectives=3, pop_size=config.pop_size, seed=0)
        run_dir = self._run(config, manipulator, opt, objectives, seed)
        df = pd.read_parquet(run_dir / "trace.parquet")
        G = np.stack(df["genotype"].to_numpy())
        assert G.shape == (config.generations * config.pop_size, 10) and G.dtype == np.float64
        assert (G >= manipulator.xl).all() and (G <= manipulator.xu).all() and (G[:, 9] < 1).all()
        stats = json.load(open(run_dir / "stats.json"))
        assert stats["genotype"] == "real" and stats["image_slice"] == [0, 8] and stats["w_txt"] == 1.0
        assert stats["anchor_g"] > 0   # float all-zero anchor scored

    def test_legacy_sampler_rejected(self):
        from mmb.config import SamplingConfig
        with pytest.raises(ValueError, match="stratified_magnitude"):
            SamplingConfig(mode="sparse_multitier")

    def test_mmba_end_to_end_boundary_pair(self, config, manipulator, objectives, seed):
        import dataclasses
        from mmb.optimizer.mmba import MMBAOptimizer
        cfg = dataclasses.replace(config, generations=6, pop_size=10)
        opt = MMBAOptimizer(space=None, num_objectives=3, pop_size=10, seed=0,
                            eps=cfg.optimizer.mmba.eps)
        run_dir = self._run(cfg, manipulator, opt, objectives, seed, sut=CrossingSUT())
        nb = json.load(open(run_dir / "min_cross_neighbour.json"))
        on = json.load(open(run_dir / "min_cross.json"))
        assert nb["pair_source"] == "own_phase2" and nb["kind"] in ("continuous", "text_step")
        assert on["within_budget"] and on["eval_idx"] is not None
        bp = json.load(open(run_dir / "boundary_pair.json"))
        assert bp["on"]["g"] < 0 <= bp["off"]["g"] and bp["gap"] <= cfg.optimizer.mmba.eps
        # canonical search (the default): the confirmation reproduces the trace exactly
        assert bp["search_scoring"] == "canonical" and bp["canonical_consistent"] is True
        assert on["g_canonical"] == pytest.approx(on["g"], abs=1e-6) and on["confirmed"]
        assert json.load(open(run_dir / "stats.json"))["search_scoring"] == "canonical"
        assert on["g"] < 0 <= nb["g"]            # anchor prefers cat_a (g > 0)
        assert nb["magnitude"] < json.load(open(run_dir / "boundary_pair.json"))["on"]["magnitude"]

    @pytest.mark.parametrize("algorithm", ["ga", "random"])
    def test_posthoc_pair_for_other_arms(self, config, manipulator, seed, algorithm):
        import dataclasses
        from mmb.objectives import build_criteria
        cfg = dataclasses.replace(config, generations=3, pop_size=10)
        crit = build_criteria("joint", "logprob_gap", True)
        objs = CriterionCollection(*[c() for c in crit])
        opt = RealPymooOptimizer(space=None, num_objectives=objs.num_objectives, pop_size=10,
                                 algorithm=algorithm, seed=1)
        run_dir = self._run(cfg, manipulator, opt, objs, seed, sut=CrossingSUT())
        if not (run_dir / "min_cross.json").exists():
            pytest.skip("no crossing in this short run")
        nb = json.load(open(run_dir / "min_cross_neighbour.json"))
        assert nb["pair_source"] == "posthoc_bisection" and nb["gap"] <= cfg.optimizer.mmba.eps
        assert nb["nb_extra_calls"] > 0
        on = json.load(open(run_dir / "min_cross.json"))
        assert on["within_budget"] and on["g"] < 0      # min_cross stays the in-budget crossing
        assert (run_dir / "boundary_on.png").exists()


class BatchArtefactSUT(FakeSUT):
    """Crosses ONLY on the batched path (a stand-in for batch-composition-dependent bf16 scoring);
    the canonical single path never crosses."""

    def __init__(self):
        self.ref = None

    def _dev(self, image):
        a = np.asarray(image, dtype=np.float32)
        if self.ref is None:
            self.ref = a
        return float(np.abs(a - self.ref).mean()) / 255.0

    def process_input(self, image, text=None, categories=None):
        self._dev(image)
        return torch.tensor([-0.1, -1.0, -5.0])

    def process_batch(self, images, texts, categories):
        return torch.stack([torch.tensor([-0.1 - 60.0 * self._dev(im), -1.0, -5.0]) for im in images])


def test_batch_artefact_crossing_is_not_confirmed(config, manipulator, objectives, seed):
    """A crossing that only exists on the batched path is reported unconfirmed, with no boundary pair."""
    import dataclasses
    from mmb.optimizer.mmba import MMBAOptimizer
    cfg = dataclasses.replace(config, generations=6, pop_size=10,
                              sut=dataclasses.replace(config.sut, batched_scoring=True))
    opt = MMBAOptimizer(space=None, num_objectives=3, pop_size=10, seed=0)
    VLMBoundaryTester(sut=BatchArtefactSUT(), manipulator=manipulator, optimizer=opt, objectives=objectives,
                      config=cfg).test([seed])
    run = next(cfg.save_dir.iterdir())
    assert (run / "min_cross.json").exists()
    mc = json.load(open(run / "min_cross.json"))
    bp = json.load(open(run / "boundary_pair.json"))
    assert mc["within_budget"] and mc["g"] < 0 and mc["g_canonical"] > 0 and not mc["confirmed"]
    assert bp["pair_source"] == "none" and bp["confirmed"] is False and bp["nb_extra_calls"] >= 3
    assert bp["search_scoring"] == "batched" and bp["canonical_consistent"] is None


def test_non_canonical_scoring_warns_loudly(config, manipulator, objectives, caplog):
    import dataclasses
    import logging
    cfg = dataclasses.replace(config, sut=dataclasses.replace(config.sut, batched_scoring=True))
    with caplog.at_level(logging.WARNING):
        VLMBoundaryTester(sut=FakeSUT(), manipulator=manipulator,
                          optimizer=RealPymooOptimizer(space=None, num_objectives=3, pop_size=5),
                          objectives=objectives, config=cfg)
    assert "NON-CANONICAL SEARCH SCORING" in caplog.text


def test_canonical_defaults():
    from mmb.config import ExperimentConfig
    c = ExperimentConfig()
    assert c.sut.batched_scoring is False and c.image.decode_batch_size == 1
