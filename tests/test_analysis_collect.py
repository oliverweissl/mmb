"""Tests for mmb.analysis.collect: hand-computed crossing / first_cross_eval cases."""
import json

import numpy as np
import pandas as pd
import pytest

from mmb.analysis import collect as col
from test_analysis_fixtures import ScriptedSUT, write_run


# ---------------------------------------------------------------------------
# Pure function, hand-computed
# ---------------------------------------------------------------------------


def lp2(g):
    return [-0.5 + g / 2, -0.5 - g / 2]


class TestEvaluateTrace:
    def test_first_crossing_and_counts(self):
        # pop 3, 3 generations: evaluation order is generation-major
        g = [2.0, 1.5, 0.5,   0.3, -0.2, 0.8,   0.1, 1.0, -3.0]
        gen = np.repeat([0, 1, 2], 3)
        r = col.evaluate_trace(np.array([lp2(x) for x in g]), gen)
        assert r["crossed"] is True
        assert r["first_cross_eval"] == 5          # 1-based: 5th evaluation (generation 1, individual 1)
        assert r["first_cross_gen"] == 1
        assert r["crossed_at_gen0"] is False
        assert r["n_cross_evals"] == 2             # -0.2 and -3.0
        assert r["floor_nats"] == pytest.approx(0.1)
        assert r["reached_band"] is False
        assert r["label_flip"] is True
        assert r["anchor_source"] == "assumed_class_a"

    def test_gen0_crossing_with_zero_genotype_anchor(self):
        # anchor evaluation (row 1) is negative, so a positive g is a crossing
        g = [0.5, -1.0, -0.2, -0.7]
        r = col.evaluate_trace(np.array([lp2(x) for x in g]), np.array([0, 0, 1, 1]), anchor_row=1)
        assert r["anchor_source"] == "zero_genotype"
        assert r["crossed"] and r["first_cross_eval"] == 1 and r["crossed_at_gen0"] is True
        assert r["n_cross_evals"] == 1
        assert r["floor_nats"] == pytest.approx(0.2)

    def test_never_crossed_and_band(self):
        g = [3.0, 1.0, 0.005, 2.0]
        r = col.evaluate_trace(np.array([lp2(x) for x in g]), np.array([0, 0, 1, 1]))
        assert r["crossed"] is False and r["first_cross_eval"] is None
        assert r["floor_nats"] == pytest.approx(0.005)
        assert r["reached_band"] is True            # inside the band without crossing
        assert r["label_flip"] is False

    def test_exact_tie_is_not_a_crossing(self):
        r = col.evaluate_trace(np.array([lp2(x) for x in [1.0, 0.0, 0.4]]), np.array([0, 0, 0]))
        assert r["crossed"] is False and r["floor_nats"] == 0.0

    def test_label_flip_via_third_candidate_without_pair_crossing(self):
        # pair = candidates (0, 1); row 1 has candidate 2 as argmax while a stays ahead of b
        lp = np.array([[-0.2, -1.0, -3.0], [-1.0, -2.0, -0.1]])
        r = col.evaluate_trace(lp, np.array([0, 0]))
        assert r["crossed"] is False
        assert r["label_flip"] is True

    def test_nondefault_target_positions(self):
        # class_a at position 2, class_b at position 0
        lp = np.array([[-3.0, -9.0, -0.5], [-0.4, -9.0, -2.0]])
        r = col.evaluate_trace(lp, np.array([0, 0]), target_classes=(2, 0))
        # g = lp[2] - lp[0]: 2.5, -1.6 -> crossing at eval 2
        assert r["crossed"] and r["first_cross_eval"] == 2

    def test_cache_hits_count_for_first_cross_but_not_nocache(self):
        g = [1.0, 1.0, 1.0, -1.0, -1.0]
        hits = [False, True, False, False, True]
        r = col.evaluate_trace(np.array([lp2(x) for x in g]), np.zeros(5, int), cache_hit=np.array(hits))
        assert r["first_cross_eval"] == 4           # cache hits counted
        assert r["first_cross_eval_nocache"] == 3   # evaluations 1, 3, 4 are fresh; the 4th is the 3rd fresh

    def test_anchor_from_stats(self):
        r = col.evaluate_trace(np.array([lp2(x) for x in [-1.0, -2.0, 0.5]]), np.zeros(3, int), anchor_g=-0.3)
        assert r["anchor_source"] == "reference_eval" and r["first_cross_eval"] == 3

    def test_reference_eval_beats_zero_genotype_row(self):
        # zero-genotype row says anchor +, reference eval says anchor -: the reference wins
        g = [0.5, 1.0, -0.2]
        r = col.evaluate_trace(np.array([lp2(x) for x in g]), np.zeros(3, int), anchor_row=0, anchor_g=-0.3)
        assert r["anchor_source"] == "reference_eval" and r["first_cross_eval"] == 1

    def test_reference_argmax_used_for_label_flip(self):
        lp = np.array([[-0.2, -1.0, -3.0], [-1.0, -2.0, -0.1]])
        r = col.evaluate_trace(lp, np.zeros(2, int), anchor_g=0.8, anchor_argmax=2)   # anchor answer = candidate 2
        assert r["label_flip"] is True and r["crossed"] is False     # row 0 differs from candidate 2


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


class TestLoadRun:
    def test_row_for_hand_computed_run(self, tmp_path):
        g = [2.0, 1.5, 0.5, 0.3, -0.2, 0.8, 0.1, 1.0, -3.0]
        img = [0.1 * (i + 1) for i in range(9)]
        txt = [0.01 * (i + 1) for i in range(9)]
        run = write_run(tmp_path, "whatever_dir_name", g, pop=3, img_mag=img, txt_mag=txt)
        row = col.load_run(run, tmp_path)
        assert row["crossed"] and row["first_cross_eval"] == 5 and row["first_cross_gen"] == 1
        assert row["evals_total"] == 9 and row["budget"] == 9 and row["complete"] is True
        assert row["floor_nats"] == pytest.approx(0.1)
        assert row["img_mag_at_cross"] == pytest.approx(0.5)     # 5th evaluation
        assert row["txt_mag_at_cross"] == pytest.approx(0.05)
        assert row["wall_s"] == 12.5
        assert row["task"] == "imagenet_pair" and row["sut"] == "llava-v1.6-mistral-7b-hf"
        assert row["cell_id"] == "junco|ostrich|La0-Lt1|p0"

    def test_censored_at_budget_when_never_crossed(self, tmp_path):
        run = write_run(tmp_path, "r", [3, 2, 1, 4, 5, 6], pop=2)
        row = col.load_run(run, tmp_path)
        assert row["crossed"] is False
        assert row["first_cross_eval"] == 6 == row["budget"]     # pop 2 x 3 generations
        assert np.isnan(row["img_mag_at_cross"])

    def test_zero_genotype_row_sets_anchor_sign(self, tmp_path):
        # evaluation 2 is the unmodified seed and has g < 0 -> anchor is negative
        geno = [[1, 1], [0, 0], [1, 0], [0, 1]]
        run = write_run(tmp_path, "r", [0.5, -1.0, -0.4, 0.2], pop=2, genotypes=geno)
        row = col.load_run(run, tmp_path)
        assert row["anchor_source"] == "zero_genotype"
        assert row["crossed"] and row["first_cross_eval"] == 1 and row["crossed_at_gen0"]

    def test_incomplete_run_is_flagged(self, tmp_path):
        run = write_run(tmp_path, "r", [1.0, 2.0, 3.0], pop=3, generations=4)  # 3 of 12 evaluations
        row = col.load_run(run, tmp_path)
        assert row["evals_total"] == 3 and row["budget"] == 12 and row["complete"] is False


class TestMetadata:
    def test_arm_comes_from_config_not_from_path(self, tmp_path):
        cfg = {"name": "e1_base_a1", "seed": 7, "optimizer": {"algorithm": "random"},
               "sut": {"model_id": "x/llava", "load_in_8bit": True}}
        run = write_run(tmp_path, "a7_image_only_looking_name", [1, 2, -1, 3], pop=2, config=cfg,
                        stats_extra={"model_id": "x/llava"})
        row = col.load_run(run, tmp_path)
        assert row["arm"] == "a1_random_tiered" and row["rng_seed"] == 7
        assert row["exp_id"] == "e1_base_a1" and row["sut"] == "llava-int8"

    @pytest.mark.parametrize("cfg, arm", [
        ({}, None),
        ({"optimizer": {"algorithm": "agemoea2"}}, "a0_directed"),
        ({"optimizer": {"algorithm": "random", "sampling": {"mode": "uniform"}}}, "a2_random_uniform"),
        ({"optimizer": {"algorithm": "ga", "proximity_only": True}}, "a3_ga_single"),
        ({"optimizer": {"algorithm": "nsga2"}}, "a4_nsga2"),
        ({"optimizer": {"algorithm": "agemoea2", "boundary_objective": "softmax_balance"}}, "a5_mimicry_objective"),
        ({"modality": "text_only", "optimizer": {"algorithm": "agemoea2"}}, "a6_text_only"),
        ({"modality": "image_only"}, "a7_image_only"),
        ({"optimizer": {"algorithm": "deepjanus"}}, "a9_deepjanus"),
    ])
    def test_derive_arm(self, cfg, arm):
        assert col.derive_arm(cfg) == arm

    def test_config_found_in_parent_directory(self, tmp_path):
        (tmp_path / "config.yaml").write_text("optimizer:\n  algorithm: nsga2\nseed: 3\n")
        run = write_run(tmp_path / "sub", "r", [1, -1], pop=2)
        row = col.load_run(run, tmp_path)
        assert row["arm"] == "a4_nsga2" and row["rng_seed"] == 3

    def test_stats_keys_beat_config_and_defaults_fill_gaps(self, tmp_path):
        cfg = {"optimizer": {"algorithm": "nsga2"}}
        run = write_run(tmp_path, "r", [1, -1], pop=2, config=cfg, stats_extra={"arm": "custom", "rng_seed": 11})
        row = col.load_run(run, tmp_path, defaults={"arm": "ignored", "exp_id": "E9"})
        assert row["arm"] == "custom" and row["rng_seed"] == 11 and row["exp_id"] == "E9"

    def test_default_arm_when_nothing_else_known(self, tmp_path):
        run = write_run(tmp_path, "r", [1, -1], pop=2)
        assert col.load_run(run, tmp_path)["arm"] is None
        assert col.load_run(run, tmp_path, defaults={"arm": "a1"})["arm"] == "a1"

    def test_run_meta_and_reference_anchor_from_stats(self, tmp_path):
        rm = {"algorithm": "random", "sampling_mode": "uniform", "modality": "joint", "rng_seed": 5,
              "boundary_objective": "logprob_gap", "proximity_only": False, "eval_budget": 8,
              "sut_model_id": "x/internvl", "exp_id": "E3", "task": "pope"}
        snap = {"optimizer": {"algorithm": "nsga2"}, "seed": 99}          # must lose against run_meta
        run = write_run(tmp_path, "r", [1.0, 2.0, 3.0, 4.0], pop=2,
                        stats_extra={"run_meta": rm, "config_snapshot": snap, "anchor_g": -0.5,
                                     "anchor_argmax": "sparrow"})
        row = col.load_run(run, tmp_path)
        assert row["arm"] == "a2_random_uniform" and row["rng_seed"] == 5
        assert row["exp_id"] == "E3" and row["task"] == "pope"
        assert row["budget"] == 8                                       # run_meta eval_budget
        assert row["anchor_source"] == "reference_eval"
        assert row["crossed"] and row["first_cross_eval"] == 1          # anchor is negative, g > 0 crosses
        assert row["label_flip"]                                        # argmax 'junco' != anchor 'sparrow'

    def test_probe_and_prior_columns_hand_computed(self, tmp_path):
        # generation 0 = first 3 evaluations: |g| = 0.6, 1.5, 0.9 -> median 0.9, min 0.6
        run = write_run(tmp_path, "r", [0.6, -1.5, 0.9, 3.0, 2.0, 4.0], pop=3,
                        stats_extra={"anchor_g": 1.2, "prior_g": -0.7, "class_a": "green iguana", "class_b": "boa"})
        row = col.load_run(run, tmp_path)
        assert row["probe_gen0"] == pytest.approx(0.9) and row["min_gen0"] == pytest.approx(0.6)
        assert row["anchor_g"] == pytest.approx(1.2) and row["prior_g"] == pytest.approx(-0.7)
        assert (row["answer_words_a"], row["answer_words_b"]) == (2, 1)
        assert row["first_cross_eval"] == 2                      # -1.5 against a positive anchor

    def test_prior_missing_is_nan_and_assumed_anchor_is_nan(self, tmp_path):
        run = write_run(tmp_path, "r", [1.0, 2.0], pop=2)
        row = col.load_run(run, tmp_path)
        assert np.isnan(row["prior_g"]) and np.isnan(row["anchor_g"]) and row["anchor_source"] == "assumed_class_a"

    def test_cell_id_fallback_without_roster_metadata(self):
        assert col.derive_cell_id({"class_a": "a", "class_b": "b", "seed_idx": 4}) == "a|b|i4"


class TestCollectCli:
    def test_cli_writes_expected_columns(self, tmp_path):
        root = tmp_path / "runs"
        write_run(root / "x", "r1", [1, 2, -1, 3], pop=2, config={"optimizer": {"algorithm": "random"}, "seed": 1})
        write_run(root / "y", "r2", [1, 2, 3, 4], pop=2, config={"optimizer": {"algorithm": "agemoea2"}, "seed": 1})
        out = tmp_path / "res" / "e.csv"
        assert col.main([str(root), "--out", str(out), "--exp-id", "E1"]) == 0
        df = pd.read_csv(out)
        assert list(df.columns) == col.COLUMNS
        assert col.COLUMNS[:17] == col.CORE_COLUMNS
        assert sorted(df["arm"]) == ["a0_directed", "a1_random_tiered"]
        assert set(df["exp_id"]) == {"E1"}
        assert df.set_index("arm").loc["a1_random_tiered", "first_cross_eval"] == 3


# ---------------------------------------------------------------------------
# Real tester with a scripted SUT, checked against the schedule
# ---------------------------------------------------------------------------


class TestWithRealTester:
    def test_collect_matches_schedule(self, tmp_path):
        from PIL import Image

        from mmb.config import ExperimentConfig, SeedTriple
        from mmb.evolutionary.vlm_boundary_tester import VLMBoundaryTester
        from mmb.manipulator.vlm_manipulator import VLMManipulator
        from mmb.objectives import CriterionCollection, MatrixDistance, TargetedBalance, TextEmbeddingDistance
        from mmb.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
        from conftest import FakeCompositeTextManipulator, FakeImageManipulator

        # calls 0 and 1 are the anchor and content-free prior reference evaluations (outside the budget);
        # then pop 5 x 3 generations = 15 evaluations, of which the 8th (schedule index 9) is the only
        # negative contrast
        schedule = [2.0, 0.5] + [1.0 + 0.1 * i for i in range(15)]
        schedule[9] = -0.4
        cfg = ExperimentConfig(
            categories=("cat_a", "cat_b", "cat_c"), prompt_template="Classify this",
            answer_format=" from: {categories}.", generations=3, pop_size=5,
            save_dir=tmp_path / "runs", name="e1_test",
            tags={"exp_id": "E1", "arm": "a4_nsga2"},
        )
        tester = VLMBoundaryTester(
            sut=ScriptedSUT(schedule),
            manipulator=VLMManipulator(image_manipulator=FakeImageManipulator(),
                                       text_manipulator=FakeCompositeTextManipulator()),
            optimizer=DiscretePymooOptimizer(gene_bounds=np.ones(1, dtype=np.int64) * 2,
                                             num_objectives=3, pop_size=5),
            objectives=CriterionCollection(MatrixDistance(), TextEmbeddingDistance(), TargetedBalance()),
            config=cfg,
        )
        tester.test([SeedTriple(image=Image.new("RGB", (8, 8)), class_a="cat_a", class_b="cat_b")])

        df = col.collect(tmp_path / "runs", defaults={"arm": "ignored"})
        assert len(df) == 1
        row = df.iloc[0]
        tr = pd.read_parquet(next((tmp_path / "runs").iterdir()) / "trace.parquet")
        assert row["anchor_source"] == "reference_eval"
        assert row["arm"] == "a4_nsga2" and row["exp_id"] == "E1"      # run_meta tags beat defaults
        assert bool(row["crossed"]) and row["first_cross_eval"] == 8
        assert row["first_cross_gen"] == 1 and not bool(row["crossed_at_gen0"])
        assert row["floor_nats"] == pytest.approx(0.4)
        assert row["anchor_g"] == pytest.approx(2.0) and row["prior_g"] == pytest.approx(0.5)
        g0 = np.abs(tr.loc[tr["generation"] == 0, "logprobs"].map(lambda v: v[0] - v[1]).to_numpy())
        assert row["probe_gen0"] == pytest.approx(np.median(g0)) and row["min_gen0"] == pytest.approx(g0.min())
        # the optimizer may drop duplicates in this 1-gene space, so a generation can hold < 5 rows
        assert row["evals_total"] == len(tr) <= 15 and row["budget"] == 15
        assert bool(row["complete"]) is True
