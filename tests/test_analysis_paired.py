"""Cost of crossing, cluster bootstrap, primary table, mixed model, per-cell figure."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_analysis_fixtures import write_run  # noqa: E402

from mmb.analysis import collect as col  # noqa: E402
from mmb.analysis import cost, paired  # noqa: E402


class TestHypervolume:
    def test_single_point(self):
        assert cost.hypervolume_2d(np.array([[1.0, 1.0]]), (4.0, 4.0)) == pytest.approx(9.0)

    def test_two_nondominated_points_hand_computed(self):
        # (1,3) and (3,1) in box [0,4]^2: 3*1 (x in [1,3), y>=3 -> height 1) + 1*... compute: strips
        # x in [1,3): best_y 3 -> 2*1 = 2 ; x in [3,4): best_y 1 -> 1*3 = 3  => 5
        assert cost.hypervolume_2d(np.array([[1.0, 3.0], [3.0, 1.0]]), (4.0, 4.0)) == pytest.approx(5.0)

    def test_dominated_point_adds_nothing_and_outside_ignored(self):
        base = cost.hypervolume_2d(np.array([[1.0, 1.0]]), (4.0, 4.0))
        more = cost.hypervolume_2d(np.array([[1.0, 1.0], [2.0, 2.0], [5.0, 0.1], [np.nan, 1.0]]), (4.0, 4.0))
        assert more == pytest.approx(base)

    def test_empty_is_zero(self):
        assert cost.hypervolume_2d(np.zeros((0, 2)), (1.0, 1.0)) == 0.0

    def test_order_invariant(self):
        rng = np.random.default_rng(0)
        p = rng.random((20, 2))
        assert cost.hypervolume_2d(p, (1, 1)) == pytest.approx(cost.hypervolume_2d(p[::-1], (1, 1)))


class TestCostOfCrossing:
    def geno(self, img_active, txt_active, image_dim=6, text_dim=4):
        g = np.zeros(image_dim + text_dim, int)
        g[:img_active] = 1
        g[image_dim:image_dim + txt_active] = 1
        return g

    def test_cheapest_crossing_edits(self):
        gens = [self.geno(5, 3), self.geno(2, 1), self.geno(1, 3), self.geno(0, 0)]
        crossing = np.array([True, True, True, False])
        r = cost.cost_of_crossing(crossing, gens, 6, 4)
        assert r["min_edits_cross"] == 3 and r["edits_img_at_min"] == 2 and r["edits_txt_at_min"] == 1
        # hv over points (5,3), (2,1), (1,3) in box (7, 5): dominated area of (2,1) and (1,3)
        exp = cost.hypervolume_2d(np.array([[5, 3], [2, 1], [1, 3]], float), (7.0, 5.0)) / 35
        assert r["hv_edits_cross"] == pytest.approx(exp) and 0 < r["hv_edits_cross"] < 1

    def test_never_crossed_is_censored(self):
        r = cost.cost_of_crossing(np.zeros(3, bool), [self.geno(1, 1)] * 3, 6, 4)
        assert r["min_edits_cross"] == 10 and r["hv_edits_cross"] == 0.0 and np.isnan(r["edits_img_at_min"])

    def test_zero_edit_crossing_fills_the_box(self):
        r = cost.cost_of_crossing(np.array([True]), [self.geno(0, 0)], 6, 4)
        assert r["hv_edits_cross"] == pytest.approx(1.0)

    def test_magnitudes_nan_without_columns_and_present_with(self):
        gens = [self.geno(1, 1)] * 2
        r = cost.cost_of_crossing(np.array([True, True]), gens, 6, 4)
        assert np.isnan(r["best_img_mag_cross"]) and np.isnan(r["hv_mag_cross"])
        r = cost.cost_of_crossing(np.array([True, True]), gens, 6, 4, img_mag=np.array([0.05, 0.02]),
                                  txt_mag=np.array([0.5, 0.9]))
        assert r["best_img_mag_cross"] == pytest.approx(0.02) and r["best_txt_mag_cross"] == pytest.approx(0.5)
        exp = cost.hypervolume_2d(np.array([[0.05, 0.5], [0.02, 0.9]]), (0.1, 1.0)) / 0.1
        assert r["hv_mag_cross"] == pytest.approx(exp)

    def test_char_distance(self):
        r = cost.cost_of_crossing(np.array([True, True]), [self.geno(1, 1)] * 2, 6, 4,
                                  texts=["Is ther e a dog?", "Is there a dog?"], seed_text="Is there a dog?")
        assert r["min_txt_chardist_cross"] == 0
        assert cost.levenshtein("kitten", "sitting") == 3


class TestCollectColumns:
    def test_columns_from_trace(self, tmp_path):
        image_dim, text_dim = 5, 3
        genos = [[1, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0] * 8]
        run = write_run(tmp_path, "r", [1.0, -0.5, -0.2, 0.7], pop=2, genotypes=genos,
                        stats_extra={"anchor_g": 2.0, "image_dim": image_dim, "text_dim": text_dim,
                                     "seed_metadata": {"prompt_template": "q"}},
                        img_mag=[0.1, 0.03, 0.04, 0.0], txt_mag=[0.2, 0.5, 0.4, 0.0])
        row = col.load_run(run, tmp_path)
        assert row["crossed"] and row["min_edits_cross"] == 1.0        # third evaluation: one active gene
        assert row["edits_img_at_min"] == 1.0 and row["edits_txt_at_min"] == 0.0
        assert row["best_img_mag_cross"] == pytest.approx(0.03) and row["best_txt_mag_cross"] == pytest.approx(0.4)
        assert 0 < row["hv_edits_cross"] < 1 and 0 < row["hv_mag_cross"] < 1
        assert row["min_txt_chardist_cross"] == 0.0                    # decoded text 'q' equals the seed question


class TestDiagColumns:
    def _run(self, tmp_path, with_diag, with_fitness=True):
        import pandas as pd
        genos = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
        run = write_run(tmp_path, "r", [1.0, -0.5, 0.7], pop=1, genotypes=genos,
                        stats_extra={"anchor_g": 2.0, "image_dim": 2, "text_dim": 1}, img_mag=[0.1, 0.03, 0.0],
                        txt_mag=[0.2, 0.5, 0.0])
        t = pd.read_parquet(run / "trace.parquet")
        if with_diag:
            t["diag_img_rms"] = [0.2, 0.011, 0.0]
            t["diag_txt_cosdist"] = [0.3, 0.6, 0.0]
        if not with_fitness:
            t = t.drop(columns=[c for c in t.columns if c.startswith("fitness_Matrix") or c.startswith("fitness_Text")])
        t.to_parquet(run / "trace.parquet")
        return col.load_run(run, tmp_path)

    def test_diag_is_primary(self, tmp_path):
        r = self._run(tmp_path, True)
        assert r["best_img_mag_cross"] == pytest.approx(0.011) and r["img_mag_at_cross"] == pytest.approx(0.011)
        assert r["best_txt_mag_cross"] == pytest.approx(0.6) and r["mag_source"] == "diag"

    def test_fitness_fallback(self, tmp_path):
        r = self._run(tmp_path, False)
        assert r["best_img_mag_cross"] == pytest.approx(0.03) and r["mag_source"] == "fitness"

    def test_none_when_nothing_logged_edit_counts_remain(self, tmp_path):
        r = self._run(tmp_path, False, with_fitness=False)
        assert np.isnan(r["best_img_mag_cross"]) and r["mag_source"] == "none"
        assert r["min_edits_cross"] == 1.0


def synth(seed=0, n_cells=10, seeds=(0, 1), arms=("a0_directed", "a1_random_tiered"), shift=True):
    rng = np.random.default_rng(seed)
    rows = []
    for task in ("imagenet", "pope"):
        for sut in ("qwen", "internvl"):
            for c in range(n_cells):
                base = rng.random()
                for s in seeds:
                    for a in arms:
                        p = min(0.95, base * (1.3 if a.startswith("a0") and shift else 0.8))
                        crossed = bool(rng.random() < p)
                        rows.append(dict(task=task, sut=sut, cell_id=f"c{c}", rng_seed=s, arm=a, crossed=crossed,
                                         first_cross_eval=int(rng.integers(1, 100)) if crossed else 1500,
                                         floor_nats=float(rng.random()), min_edits_cross=float(rng.integers(1, 50)),
                                         hv_edits_cross=float(rng.random()) * crossed))
    return pd.DataFrame(rows)


class TestPaired:
    def test_cluster_ci_brackets_point_and_shrinks_with_cells(self):
        d = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], float)
        p, lo, hi = paired.cluster_ci(d, "binary", 2000)
        assert lo <= p <= hi and p == pytest.approx(0.7)
        d2 = np.tile(d, 10)
        _, lo2, hi2 = paired.cluster_ci(d2, "binary", 2000)
        assert (hi2 - lo2) < (hi - lo)

    def test_paired_table_columns_and_holm(self):
        df = synth()
        s = df[(df.task == "imagenet") & (df.sut == "qwen")]
        t = paired.paired_table(s, "a0_directed", ("crossed", "first_cross_eval", "min_edits"), n_boot=500)
        assert {"diff", "diff_lo", "diff_hi", "p", "p_holm", "n_cells"} <= set(t.columns)
        assert (t.n_cells == 10).all() and (t.n_pairs == 20).all()
        assert (t.p_holm >= t.p - 1e-12).all()

    def test_binary_diff_is_share_difference_with_equal_cells(self):
        df = synth(shift=False)
        s = df[(df.task == "pope") & (df.sut == "qwen")]
        t = paired.paired_table(s, "a0_directed", ("crossed",), n_boot=200).iloc[0]
        a0 = s[s.arm == "a0_directed"].crossed.mean()
        a1 = s[s.arm == "a1_random_tiered"].crossed.mean()
        assert t["diff"] == pytest.approx(a1 - a0)       # arm minus reference; equal runs per cell

    def test_nan_pairs_dropped(self):
        df = synth()
        m = (df.task == "pope") & (df.sut == "qwen")
        df.loc[df.index[m][:6], "hv_edits_cross"] = np.nan
        t = paired.paired_table(df[m], "a0_directed", ("hv_edits",), n_boot=100)
        assert t.iloc[0].n_pairs < 20

    def test_primary_table_holm_across_strata(self):
        t = paired.primary_table(synth(), n_boot=200)
        assert len(t) == 4 and (t.p_holm_strata >= t.p - 1e-12).all()
        assert set(t.task) == {"imagenet", "pope"}

    def test_pooled_uses_cluster_column(self):
        df = synth()
        df["cluster_id"] = df.task + "/" + df.cell_id
        df["cell_id"] = df.task + "/" + df.sut + "/" + df.cell_id
        t = paired.paired_table(df, "a0_directed", ("crossed",), n_boot=100).iloc[0]
        assert t.n_pairs == 80 and t.n_cells == 20      # 2 tasks x 10 cells, SUTs share clusters

    def test_mixed_model_runs_and_has_right_sign(self):
        pytest.importorskip("statsmodels")
        df = synth(n_cells=12)
        r = paired.mixed_model(df, "a0_directed", "a1_random_tiered")
        assert np.isfinite(r["log_or"]) and r["log_or"] < 0     # a1 is the lower-crossing arm in synth()
        assert r["lo"] < r["log_or"] < r["hi"]

    def test_mixed_model_degenerate_returns_nan(self):
        pytest.importorskip("statsmodels")
        df = synth()
        df["crossed"] = True
        assert np.isnan(paired.mixed_model(df, "a0_directed", "a1_random_tiered")["log_or"])

    def test_per_cell_figure(self, tmp_path):
        out = paired.plot_per_cell_diff(synth(), tmp_path / "pc.pdf")
        assert out.read_bytes()[:4] == b"%PDF"
