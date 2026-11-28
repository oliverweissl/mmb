"""Cluster bootstrap, paired tables on the continuous metrics, primary table (a0 vs a5 / a9), mixed model."""
import numpy as np
import pandas as pd
import pytest

from mmb.analysis import paired  # noqa: E402


def synth(seed=0, n_cells=10, seeds=(0, 1), arms=("a0_directed", "a9_deepjanus"), shift=True):
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
                                         floor_nats=float(rng.random()), min_within_budget=crossed,
                                         magnitude=float(rng.uniform(0.01, 0.5)) if crossed else np.nan,
                                         mag_image=float(rng.uniform(0, 0.4)) if crossed else np.nan,
                                         mag_text=float(rng.uniform(0, 0.3)) if crossed else np.nan))
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
        t = paired.paired_table(s, "a0_directed", ("crossed", "first_cross_eval"), n_boot=500)
        assert {"diff", "diff_lo", "diff_hi", "p", "p_holm", "n_cells"} <= set(t.columns)
        assert (t.n_cells == 10).all() and (t.n_pairs == 20).all()
        tm = paired.paired_table(s, "a0_directed", ("magnitude",), n_boot=200)
        both = (s[s.arm == "a0_directed"].set_index(["cell_id", "rng_seed"]).crossed
                & s[s.arm == "a9_deepjanus"].set_index(["cell_id", "rng_seed"]).crossed)
        assert len(tm) == 0 or tm.iloc[0].n_pairs == int(both.sum())          # magnitude: pairs where both crossed
        assert (t.p_holm >= t.p - 1e-12).all()

    def test_binary_diff_is_share_difference_with_equal_cells(self):
        df = synth(shift=False)
        s = df[(df.task == "pope") & (df.sut == "qwen")]
        t = paired.paired_table(s, "a0_directed", ("crossed",), n_boot=200).iloc[0]
        a0 = s[s.arm == "a0_directed"].crossed.mean()
        a9 = s[s.arm == "a9_deepjanus"].crossed.mean()
        assert t["diff"] == pytest.approx(a9 - a0)       # arm minus reference; equal runs per cell

    def test_out_of_budget_smallest_crossing_is_not_paired(self):
        df = synth()
        m = (df.task == "pope") & (df.sut == "qwen")
        sub = df[m].copy()
        sub["crossed"], sub["min_within_budget"] = True, True
        sub["magnitude"] = 0.1
        full = paired.paired_table(sub, "a0_directed", ("magnitude",), n_boot=100).iloc[0].n_pairs
        sub.loc[sub.index[:6], "min_within_budget"] = False
        t = paired.paired_table(sub, "a0_directed", ("magnitude",), n_boot=100)
        assert full == 20 and t.iloc[0].n_pairs < 20

    def test_primary_table_holm_across_strata(self):
        df = synth(arms=("a0_directed", "a5_mimicry_objective", "a9_deepjanus"))
        for arm in ("a5_mimicry_objective", "a9_deepjanus"):
            t = paired.primary_table(df, arm=arm, n_boot=200)
            assert len(t) == 4 and (t.p_holm_strata >= t.p - 1e-12).all()
            assert set(t.task) == {"imagenet", "pope"} and set(t.arm) == {arm}

    def test_pooled_uses_cluster_column(self):
        df = synth()
        df["cluster_id"] = df.task + "/" + df.cell_id
        df["cell_id"] = df.task + "/" + df.sut + "/" + df.cell_id
        t = paired.paired_table(df, "a0_directed", ("crossed",), n_boot=100).iloc[0]
        assert t.n_pairs == 80 and t.n_cells == 20      # 2 tasks x 10 cells, SUTs share clusters

    def test_mixed_model_runs_and_has_right_sign(self):
        pytest.importorskip("statsmodels")
        df = synth(n_cells=12)
        r = paired.mixed_model(df, "a0_directed", "a9_deepjanus")
        assert np.isfinite(r["log_or"]) and r["log_or"] < 0     # a9 is the lower-crossing arm in synth()
        assert r["lo"] < r["log_or"] < r["hi"]

    def test_mixed_model_degenerate_returns_nan(self):
        pytest.importorskip("statsmodels")
        df = synth()
        df["crossed"] = True
        assert np.isnan(paired.mixed_model(df, "a0_directed", "a9_deepjanus")["log_or"])
