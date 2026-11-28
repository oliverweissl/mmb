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

    def test_budget_bound_trace(self, tmp_path):
        # 6 evaluations in the trace, budget 4 (run_meta): the crossing at evaluation 5 is outside the budget
        run = write_run(tmp_path, "r", [1.0, 2.0, 0.5, 3.0, -1.0, 0.01], pop=2,
                        stats_extra={"run_meta": {"eval_budget": 4}})
        row = col.load_run(run, tmp_path)
        assert row["budget"] == 4 and row["evals_total"] == 6 and row["evals_in_budget"] == 4
        assert row["crossed"] is False and row["first_cross_eval"] == 4
        assert row["floor_nats"] == pytest.approx(0.5)               # 0.01 is outside the budget

    def test_cell_key_wins(self, tmp_path):
        run = write_run(tmp_path, "r", [1.0, -1.0], pop=2,
                        stats_extra={"seed_metadata": {"cell_key": "pope|COCO_1|dog|no"}})
        assert col.load_run(run, tmp_path)["cell_id"] == "pope|COCO_1|dog|no"

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
        cfg = {"name": "e1_base_a9", "seed": 7, "optimizer": {"algorithm": "deepjanus"},
               "sut": {"model_id": "x/llava", "load_in_8bit": True}}
        run = write_run(tmp_path, "a7_image_only_looking_name", [1, 2, -1, 3], pop=2, config=cfg,
                        stats_extra={"model_id": "x/llava"})
        row = col.load_run(run, tmp_path)
        assert row["arm"] == "a9_deepjanus" and row["rng_seed"] == 7
        assert row["exp_id"] == "e1_base_a9" and row["sut"] == "llava-int8"

    @pytest.mark.parametrize("cfg, arm", [
        ({}, None),
        ({"optimizer": {"algorithm": "mmba"}}, "a0_directed"),
        ({"optimizer": {"algorithm": "mmba_unguided"}}, "x1_mmba_unguided"),
        ({"optimizer": {"algorithm": "agemoea2", "boundary_objective": "softmax_balance"}}, "a5_mimicry_objective"),
        ({"optimizer": {"algorithm": "ga", "boundary_objective": "softmax_balance"}}, "a5_mimicry_objective"),
        ({"modality": "text_only", "optimizer": {"algorithm": "mmba"}}, "a6_text_only"),
        ({"modality": "image_only"}, "a7_image_only"),
        ({"optimizer": {"algorithm": "deepjanus"}}, "a9_deepjanus"),
        ({"optimizer": {"algorithm": "nsga2"}}, "nsga2"),                 # removed arms pass through
    ])
    def test_derive_arm(self, cfg, arm):
        assert col.derive_arm(cfg) == arm

    def test_config_found_in_parent_directory(self, tmp_path):
        (tmp_path / "config.yaml").write_text("optimizer:\n  algorithm: mmba_unguided\nseed: 3\n")
        run = write_run(tmp_path / "sub", "r", [1, -1], pop=2)
        row = col.load_run(run, tmp_path)
        assert row["arm"] == "x1_mmba_unguided" and row["rng_seed"] == 3

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
        rm = {"algorithm": "deepjanus", "sampling_mode": "stratified_magnitude", "modality": "joint", "rng_seed": 5,
              "boundary_objective": "logprob_gap", "proximity_only": False, "eval_budget": 8,
              "sut_model_id": "x/internvl", "exp_id": "E3", "task": "pope"}
        snap = {"optimizer": {"algorithm": "mmba"}, "seed": 99}           # must lose against run_meta
        run = write_run(tmp_path, "r", [1.0, 2.0, 3.0, 4.0], pop=2,
                        stats_extra={"run_meta": rm, "config_snapshot": snap, "anchor_g": -0.5,
                                     "anchor_argmax": "sparrow"})
        row = col.load_run(run, tmp_path)
        assert row["arm"] == "a9_deepjanus" and row["rng_seed"] == 5 and row["algorithm"] == "deepjanus"
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


STATS_REAL = {"image_slice": [0, 2], "text_slice": [2, 4], "w_img": 2.0, "w_txt": 0.5,
              "prompt_template": "Is there a dog in the image?"}


class TestMinCrossAndPair:
    def test_min_cross_and_pair_columns(self, tmp_path):
        mc = {"eval_idx": 3, "within_budget": True, "confirmed": True, "magnitude": 0.5, "mag_image": 0.3,
              "mag_text": 0.4, "text": "Does the image show a dog?", "genotype": [0.1, 0.2, 0.8, 0.1]}
        bp = {"pair_source": "own_phase2", "gap": 0.001, "lam_on": 0.5, "lam_off": 0.499, "kind": "text_step",
              "text_differs": True, "eps": 0.001, "nb_extra_calls": 3,
              "on": {"g": -0.1, "magnitude": 0.5, "mag_image": 0.3, "mag_text": 0.4, "text": "A?"},
              "off": {"g": 0.2, "magnitude": 0.49, "mag_image": 0.29, "mag_text": 0.39, "text": "B?"}}
        pool = tmp_path / "pool.json"
        pool.write_text(json.dumps({"meta": {}, "questions": {"Is there a dog in the image?": {
            "pool": [{"text": "Does the image show a dog?", "dist": 0.135}]}}}))
        run = write_run(tmp_path, "r", [1.0, 0.5, -0.2, 0.3], pop=2, stats_extra=STATS_REAL, min_cross=mc,
                        boundary_pair=bp)
        row = col.load_run(run, tmp_path, pool_path=pool)
        assert row["min_found"] and row["min_within_budget"] and row["min_confirmed"] and row["min_eval_idx"] == 3
        assert (row["magnitude"], row["mag_image"], row["mag_text"]) == (0.5, 0.3, 0.4)
        assert row["min_text_changed"] and row["min_text_d"] == pytest.approx(0.135) and row["min_modality"] == "joint"
        assert row["bp_source"] == "own_phase2" and row["bp_kind"] == "text_step" and row["bp_text_differs"]
        assert row["nb_extra_calls"] == 3 and row["bp_gap"] == 0.001 and row["bp_on_g"] == -0.1
        assert row["bp_off_magnitude"] == 0.49 and row["bp_off_text"] == "B?"
        assert row["w_img"] == 2.0 and row["w_txt"] == 0.5

    def test_min_cross_outside_budget_is_nan(self):
        r = col.min_cross_columns({"eval_idx": None, "within_budget": False, "magnitude": 0.2, "text": "q"}, "q")
        assert r["min_found"] and r["min_within_budget"] is False and np.isnan(r["magnitude"]) and r["min_modality"] is None

    def test_missing_block_magnitudes_recomputed_from_genotype(self):
        # image c = (0.3, 0.4): RMS = 0.3536, x w_img 2 = 0.7071; text t = 0.8 x w_txt 0.5 = 0.4
        mc = {"eval_idx": 1, "within_budget": True, "magnitude": 0.81, "text": "Is there a dog in the image?",
              "genotype": [0.3, 0.4, 0.8, 0.9]}
        r = col.min_cross_columns(mc, "Is there a dog in the image?", stats=STATS_REAL)
        assert r["mag_image"] == pytest.approx(2 * 0.5 / np.sqrt(2)) and r["mag_text"] == pytest.approx(0.4)
        assert r["min_text_changed"] is False and r["min_text_d"] == 0.0 and r["min_modality"] == "image"

    @pytest.mark.parametrize("mi, changed, want", [(0.1, False, "image"), (0.0, True, "text"), (0.1, True, "joint"),
                                                   (0.0, False, "none"), (float("nan"), True, None)])
    def test_modality(self, mi, changed, want):
        assert col.modality_of(mi, changed) == want

    def test_pair_none_and_missing(self):
        assert col.pair_columns(None)["bp_source"] is None
        r = col.pair_columns({"pair_source": "none", "nb_extra_calls": 11, "eps": 0.001})
        assert r["bp_source"] == "none" and r["nb_extra_calls"] == 11 and r["bp_on_g"] is None


class TestCollectCli:
    def test_cli_writes_expected_columns(self, tmp_path):
        root = tmp_path / "runs"
        write_run(root / "x", "r1", [1, 2, -1, 3], pop=2, config={"optimizer": {"algorithm": "deepjanus"}, "seed": 1})
        write_run(root / "y", "r2", [1, 2, 3, 4], pop=2, config={"optimizer": {"algorithm": "mmba"}, "seed": 1})
        out = tmp_path / "res" / "e.csv"
        assert col.main([str(root), "--out", str(out), "--exp-id", "E1"]) == 0
        df = pd.read_csv(out)
        assert list(df.columns) == col.COLUMNS
        for c in ("min_edits_cross", "crossed_robust", "mag_source", "min_cross_eval"):
            assert c not in df                                       # integer-genotype columns are gone
        assert sorted(df["arm"]) == ["a0_directed", "a9_deepjanus"]
        assert set(df["exp_id"]) == {"E1"}
        assert df.set_index("arm").loc["a9_deepjanus", "first_cross_eval"] == 3
        assert not df["min_found"].any() and df["bp_source"].isna().all()


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
        from mmb.optimizer.real_pymoo_optimizer import RealPymooOptimizer
        from conftest import make_vlm_manipulator, write_question_pool

        # calls 0 and 1 are the anchor and content-free prior reference evaluations (outside the budget);
        # then pop 5 x 3 generations = 15 evaluations, of which the 8th (schedule index 9) is the only
        # negative contrast
        schedule = [2.0, 0.5] + [1.0 + 0.1 * i for i in range(15)]
        schedule[9] = -0.4
        schedule += [1.0] * 64          # canonical re-scoring after the search (outside the budget)
        cfg = ExperimentConfig(
            categories=("cat_a", "cat_b", "cat_c"), prompt_template="Classify this",
            answer_format=" from: {categories}.", generations=3, pop_size=5,
            save_dir=tmp_path / "runs", name="e1_test",
            tags={"exp_id": "E1", "arm": "a9_deepjanus"},
        )
        tester = VLMBoundaryTester(
            sut=ScriptedSUT(schedule),
            manipulator=make_vlm_manipulator(write_question_pool(tmp_path / "pool.json", ["Classify this"])),
            optimizer=RealPymooOptimizer(space=None, num_objectives=3, pop_size=5, seed=0),
            objectives=CriterionCollection(MatrixDistance(), TextEmbeddingDistance(), TargetedBalance()),
            config=cfg,
        )
        tester.test([SeedTriple(image=Image.new("RGB", (64, 64)), class_a="cat_a", class_b="cat_b")])

        df = col.collect(tmp_path / "runs", defaults={"arm": "ignored"})
        assert len(df) == 1
        row = df.iloc[0]
        tr = pd.read_parquet(next((tmp_path / "runs").iterdir()) / "trace.parquet")
        assert row["anchor_source"] == "reference_eval"
        assert row["arm"] == "a9_deepjanus" and row["exp_id"] == "E1"      # run_meta tags beat defaults
        assert bool(row["crossed"]) and row["first_cross_eval"] == 8
        assert row["first_cross_gen"] == 1 and not bool(row["crossed_at_gen0"])
        assert row["floor_nats"] == pytest.approx(0.4)
        assert row["anchor_g"] == pytest.approx(2.0) and row["prior_g"] == pytest.approx(0.5)
        g0 = np.abs(tr.loc[tr["generation"] == 0, "logprobs"].map(lambda v: v[0] - v[1]).to_numpy())
        assert row["probe_gen0"] == pytest.approx(np.median(g0)) and row["min_gen0"] == pytest.approx(g0.min())
        # the optimizer may drop duplicates in this 1-gene space, so a generation can hold < 5 rows
        assert row["evals_total"] == len(tr) <= 15 and row["budget"] == 15
        assert bool(row["complete"]) is True
