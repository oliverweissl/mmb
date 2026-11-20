"""Tests for mmb.analysis.predictors on hand-computed cases and synthetic data."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from mmb.analysis import predictors as pr


class TestVectorisedStats:
    def test_spearman_hand_computed(self):
        # ranks of y: 2,1,4,3,5 against 1..5: sum d^2 = 4 -> rho = 1 - 6*4/(5*24) = 0.8
        rho = pr.spearman_rows(np.array([[1, 2, 3, 4, 5.0]]), np.array([[2, 1, 4, 3, 5.0]]))[0]
        assert rho == pytest.approx(0.8)

    def test_spearman_rows_matches_scipy_with_ties(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 6, (7, 30)).astype(float)
        y = rng.integers(0, 6, (7, 30)).astype(float)
        want = [sps.spearmanr(a, b).statistic for a, b in zip(x, y)]
        assert pr.spearman_rows(x, y) == pytest.approx(want)

    def test_constant_row_gives_nan(self):
        assert np.isnan(pr.spearman_rows(np.ones((1, 5)), np.arange(5.0)[None])[0])

    def test_auc_hand_computed(self):
        # positives {2, 4}, negatives {1, 3}: 3 of 4 pairs ordered -> 0.75; ties count half
        assert pr.auc_rows(np.array([[2, 4, 1, 3.0]]), np.array([[1, 1, 0, 0.0]]))[0] == pytest.approx(0.75)
        assert pr.auc_rows(np.array([[1, 1.0]]), np.array([[1, 0.0]]))[0] == 0.5
        assert np.isnan(pr.auc_rows(np.array([[1, 2.0]]), np.array([[1, 1.0]]))[0])

    def test_partial_spearman_removes_common_cause(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=300)
        x = z + 0.05 * rng.normal(size=300)
        y = z + 0.05 * rng.normal(size=300)
        assert sps.spearmanr(x, y).statistic > 0.95
        assert abs(pr.partial_spearman(x, y, z[:, None])) < 0.2
        # an independent signal is kept
        w = rng.normal(size=300)
        assert pr.partial_spearman(w + z, w, z[:, None]) > 0.5

    def test_bootstrap_is_seeded_and_brackets(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=40); y = x + rng.normal(size=40)
        a = pr.spearman_ci(x, y, n_boot=500)
        assert a == pr.spearman_ci(x, y, n_boot=500) and a[1] <= a[0] <= a[2]
        lab = (x + rng.normal(size=40)) > 0
        r = pr.auc_ci(x, lab, n_boot=500)
        assert r[1] <= r[0] <= r[2] and r[0] > 0.5


def synthetic(n_cells=30, n_seeds=3, seed=0, sut="llava", task="imagenet_pair"):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        base = rng.normal(0, 1)                      # cell hardness
        prior = rng.normal(0, 1)
        for s in range(n_seeds):
            probe = np.exp(base + 0.3 * rng.normal())
            floor = np.exp(base + 0.3 * rng.normal()) * 0.1
            crossed = (base + 0.4 * rng.normal()) < 0
            rows.append(dict(sut=sut, task=task, arm="a0_directed", cell_id=f"c{c}", rng_seed=s,
                             anchor_g=rng.uniform(1, 3), prior_g=prior + 0.1 * rng.normal(),
                             probe_gen0=probe, min_gen0=probe * 0.5, floor_nats=floor,
                             first_cross_eval=int(5 + 400 * (not crossed) + rng.integers(0, 50)),
                             crossed=bool(crossed), answer_words_a=1 + c % 2, answer_words_b=1))
    return pd.DataFrame(rows)


class TestFrames:
    def test_cell_frame_averages_over_seeds(self):
        d = pr.prepare(pd.DataFrame({
            "sut": "a", "task": "t", "cell_id": ["x", "x", "x", "y", "y", "y"], "rng_seed": [0, 1, 2] * 2,
            "probe_gen0": [1, 2, 3, 10, 10, 10.0], "floor_nats": [1, 1, 4, 2, 2, 2.0],
            "first_cross_eval": [1, 2, 3, 4, 5, 6], "crossed": [True, True, False, False, False, True],
            "prior_g": [-1, -1, -1, 2, 2, 2.0]}))
        c = pr.cell_frame(d, ["sut", "task"]).set_index("cell_id")
        assert c.loc["x", "probe_gen0"] == 2.0 and c.loc["x", "floor_nats"] == 2.0
        assert c.loc["x", "crossed_share"] == pytest.approx(2 / 3) and bool(c.loc["x", "crossed"]) is True
        assert bool(c.loc["y", "crossed"]) is False and c.loc["y", "n_seeds"] == 3
        assert c.loc["x", "abs_prior_g"] == 1.0

    def test_prepare_derives_abs_prior_and_word_diff(self):
        d = pr.prepare(synthetic(3, 1))
        assert (d["abs_prior_g"] == d["prior_g"].abs()).all()
        assert set(d["answer_words_diff"]) <= {0, 1}


class TestAnalysis:
    def test_probe_tracks_floor_and_predicts_crossing(self):
        d = pr.prepare(synthetic())
        c = pr.correlations(d, "run", ["sut", "task"], n_boot=300).set_index(["predictor", "outcome", "stat"])
        rho = c.loc[("probe_gen0", "floor_nats", "spearman")]
        assert rho["value"] > 0.6 and rho["lo"] > 0.3 and rho["n"] == 90
        auc = c.loc[("probe_gen0", "crossed", "auc")]
        assert auc["value"] < 0.35 and auc["hi"] < 0.5          # larger probe margin -> fewer crossings
        assert c.loc[("anchor_g", "floor_nats", "spearman")]["lo"] < 0 < c.loc[("anchor_g", "floor_nats", "spearman")]["hi"]

    def test_partial_given_prior_and_min(self):
        d = pr.prepare(synthetic())
        p = pr.partials(d, "run", ["sut", "task"], n_boot=100)
        row = p[(p["predictor"] == "probe_gen0") & (p["given"] == "prior_g+min_gen0") & (p["outcome"] == "floor_nats")].iloc[0]
        # min_gen0 is half the probe, so nothing is left of the probe once min_gen0 is controlled
        assert abs(row["rho_partial"]) < abs(row["rho_raw"]) and abs(row["rho_partial"]) < 0.3

    def test_levels_are_separate_tables_and_groups_split(self, tmp_path):
        df = pd.concat([synthetic(25, 3, 0, "llava", "imagenet_pair"), synthetic(25, 3, 1, "qwen", "pope")])
        res = pr.run_all(df, tmp_path, n_boot=100)
        assert {"run_level", "cell_level", "partial_run", "partial_cell"} == set(res)
        run, cell = res["run_level"], res["cell_level"]
        assert set(zip(run["sut"], run["task"])) == {("llava", "imagenet_pair"), ("qwen", "pope")}
        assert run["n"].max() == 75 and cell["n"].max() == 25 and set(run["level"]) == {"run"} and set(cell["level"]) == {"cell"}
        for f in ("run_level", "cell_level", "partial_run", "partial_cell"):
            assert (tmp_path / f"{f}.csv").is_file() and (tmp_path / f"{f}.tex").is_file()
        tex = (tmp_path / "run_level.tex").read_text()
        assert r"\toprule" in tex and r"probe\_gen0" in tex and "imagenet\\_pair" in tex

    def test_small_groups_are_skipped(self):
        d = pr.prepare(synthetic(2, 1))
        assert pr.correlations(d, "run", ["sut", "task"], n_boot=50).empty

    def test_arm_filter_and_cli(self, tmp_path):
        a = synthetic(20, 2, 0); b = synthetic(20, 2, 1); b["arm"] = "a1_random_tiered"
        p = tmp_path / "r.csv"; pd.concat([a, b]).to_csv(p, index=False)
        assert pr.main([str(p), "--out", str(tmp_path / "o"), "--arm", "a0_directed", "--n-boot", "50"]) == 0
        assert pd.read_csv(tmp_path / "o" / "run_level.csv")["n"].max() == 40
        assert pr.main([str(p), "--out", str(tmp_path / "o2"), "--by-arm", "--n-boot", "50"]) == 0
        assert "arm" in pd.read_csv(tmp_path / "o2" / "run_level.csv").columns

    def test_missing_columns_exit(self, tmp_path):
        p = tmp_path / "r.csv"; synthetic(5, 1).drop(columns=["prior_g"]).to_csv(p, index=False)
        with pytest.raises(SystemExit, match="prior_g"):
            pr.main([str(p), "--out", str(tmp_path / "o")])
