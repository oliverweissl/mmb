"""Tests for mmb.analysis.stats / plots on hand-computed cases and synthetic result tables."""
import numpy as np
import pandas as pd
import pytest

from mmb.analysis import stats as st


def frame(arm, firsts, floors=None, crossed=None, seeds=None, cells=None, budget=100, mags=None):
    n = len(firsts)
    crossed = [f is not None for f in firsts] if crossed is None else crossed
    mags = mags if mags is not None else [np.nan if f is None else 0.01 * f for f in firsts]
    return pd.DataFrame({
        "magnitude": mags,
        "arm": arm,
        "cell_id": cells if cells is not None else [f"c{i}" for i in range(n)],
        "rng_seed": seeds if seeds is not None else [0] * n,
        "crossed": crossed,
        "crossed_at_gen0": [bool(c) and f is not None and f <= 3 for c, f in zip(crossed, firsts)],
        "first_cross_eval": [budget if f is None else f for f in firsts],
        "floor_nats": floors if floors is not None else [0.5 if f is None else 0.01 for f in firsts],
        "reached_band": [False] * n,
        "label_flip": crossed,
        "budget": budget,
    })


class TestIntervals:
    def test_wilson_matches_thesis_interval(self):
        lo, hi = st.wilson(99, 119)           # Exp-100: 99/119, reported 75.5--88.8
        assert lo == pytest.approx(0.755, abs=2e-3) and hi == pytest.approx(0.888, abs=2e-3)

    def test_wilson_edges(self):
        assert st.wilson(0, 10)[0] == 0.0 and st.wilson(10, 10)[1] == pytest.approx(1.0)
        assert np.isnan(st.wilson(0, 0)[0])

    def test_bootstrap_is_seeded_and_brackets_point(self):
        v = np.arange(1, 41, dtype=float)
        a = st.bootstrap_stat(v, n_boot=2000)
        b = st.bootstrap_stat(v, n_boot=2000)
        assert a == b
        assert a[1] <= a[0] <= a[2]

    def test_default_resamples_and_seed(self):
        assert st.N_BOOT == 10_000 and st.BOOT_SEED == 42


class TestCurve:
    def test_share_at_budgets_hand_computed(self):
        # runs: cross at 1, 5, 5, 100, never  -> share(b) = #{first<=b} / 5
        first = np.array([1, 5, 5, 100, 100.0])
        crossed = np.array([True, True, True, True, False])
        s = st.share_curve(first, crossed, [1, 4, 5, 99, 100])
        assert s.tolist() == [0.2, 0.2, 0.6, 0.6, 0.8]

    def test_curve_table_shares_wilson_and_bands(self):
        df = frame("a", [1, 5, 5, 100, None])
        c = st.curve_with_bands(df, [1, 5, 100], n_boot=500)
        assert c["share"].tolist() == [0.2, 0.6, 0.8]
        assert c["k"].tolist() == [1, 3, 4] and (c["n"] == 5).all()
        assert (c["boot_lo"] <= c["share"]).all() and (c["share"] <= c["boot_hi"]).all()
        assert (c["wilson_lo"] <= c["share"]).all() and (c["share"] <= c["wilson_hi"]).all()
        assert c["share"].is_monotonic_increasing


class TestPairedTests:
    def test_holm_hand_computed(self):
        assert st.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])

    def test_mcnemar_exact_hand_computed(self):
        # b = 3 ref-only, c = 0 arm-only -> p = 2 * 0.5^3 = 0.25
        r = st.mcnemar_exact([1, 1, 1, 0, 0], [0, 0, 0, 0, 0])
        assert (r["b_ref_only"], r["c_arm_only"]) == (3, 0) and r["p"] == pytest.approx(0.25)
        # b = 5, c = 1 -> p = 2 * (C(6,0)+C(6,1)) / 64 = 0.21875
        r = st.mcnemar_exact([1] * 5 + [0], [0] * 5 + [1])
        assert r["p"] == pytest.approx(0.21875)
        assert st.mcnemar_exact([1, 0], [1, 0])["p"] == 1.0          # no discordant pairs

    def test_rank_biserial_hand_computed(self):
        # d = [1, 2, -3, 4]: ranks 1..4, W+ = 7, W- = 3 -> r = 0.4; zeros dropped
        assert st.rank_biserial_paired([1, 2, -3, 4, 0]) == pytest.approx(0.4)
        assert st.rank_biserial_paired([-1, -2, -3]) == -1.0
        assert st.rank_biserial_paired([0, 0]) == 0.0

    def test_wilcoxon_all_equal_gives_p_one(self):
        r = st.wilcoxon_paired([5, 5, 5], [5, 5, 5])
        assert r["p"] == 1.0 and r["r_rb"] == 0.0

    def test_paired_vs_reference_matches_on_cell_and_seed(self):
        cells = [f"c{i}" for i in range(12)]
        ref = frame("ref", [10] * 12, cells=cells)                      # always crosses at 10
        slow = frame("slow", [10 + 5 * (i + 1) for i in range(12)], cells=cells)
        noc = frame("none", [None] * 12, cells=cells, crossed=[False] * 12)
        extra = frame("slow", [7], cells=["unmatched"])                 # only in one arm: dropped
        df = pd.concat([ref, slow, extra, noc], ignore_index=True)
        out = st.paired_vs_reference(df, "ref")
        assert set(out["arm"]) == {"slow", "none"} and (out["n_pairs"] == 12).all()
        m = out.set_index(["arm", "metric"])
        # slow never differs in crossing; none loses all 12 crossings: b=12,c=0 -> p = 2 * 0.5^12
        assert m.loc[("slow", "crossed"), "p"] == 1.0
        assert m.loc[("none", "crossed"), "p"] == pytest.approx(2 * 0.5**12)
        assert m.loc[("none", "crossed"), "effect"] == pytest.approx(-1.0)
        # slow needs more evaluations in every pair -> r = +1 (arm worse), none censored at 100 -> also +1
        assert m.loc[("slow", "first_cross_eval"), "effect"] == pytest.approx(1.0)
        assert m.loc[("none", "first_cross_eval"), "effect"] == pytest.approx(1.0)
        # Holm across the two arms, per metric
        for metric in ("crossed", "first_cross_eval", "floor"):
            sub = out[out["metric"] == metric]
            assert (sub["p_holm"] >= sub["p"] - 1e-12).all() and (sub["p_holm"] <= 1).all()
        p2 = m.loc[("none", "first_cross_eval"), "p"]
        p1 = m.loc[("slow", "first_cross_eval"), "p"]
        expected = st.holm([p1, p2])
        assert m.loc[("slow", "first_cross_eval"), "p_holm"] == pytest.approx(expected[0])
        assert m.loc[("none", "first_cross_eval"), "p_holm"] == pytest.approx(expected[1])

    def test_duplicate_pairs_are_rejected(self):
        df = pd.concat([frame("ref", [1, 2], cells=["c", "c"]), frame("x", [1, 2], cells=["c", "d"])])
        with pytest.raises(ValueError, match="several runs"):
            st.paired_vs_reference(df, "ref")

    def test_missing_reference_raises(self):
        with pytest.raises(ValueError, match="reference arm"):
            st.paired_vs_reference(frame("a", [1, 2]), "nope")


class TestMagnitude:
    def test_magnitude_paired_only_where_both_crossed(self):
        cells = [f"c{i}" for i in range(6)]
        ref = frame("ref", [10, 10, 10, 10, None, 10], cells=cells, mags=[0.1, 0.2, 0.3, 0.4, np.nan, 0.5])
        arm = frame("arm", [10, 10, 10, 10, 10, None], cells=cells, mags=[0.2, 0.4, 0.6, 0.8, 0.1, np.nan])
        out = st.paired_vs_reference(pd.concat([ref, arm]), "ref").set_index("metric")
        r = out.loc["magnitude"]
        assert r["n_pairs"] == 4                       # c4 and c5: one side did not cross within the budget
        assert r["effect"] == pytest.approx(1.0)       # arm larger in every pair
        assert r["ref_value"] == pytest.approx(0.25) and r["arm_value"] == pytest.approx(0.5)

    def test_summary_pair_and_modality_columns(self):
        d = frame("a0_directed", [5, 6, 7, None])
        d["min_modality"] = ["image", "text", "image", None]
        d["bp_source"] = ["own_phase2", "posthoc_bisection", "none", None]
        d["bp_gap"] = [0.001, 0.003, np.nan, np.nan]
        d["bp_kind"] = ["continuous", "text_step", None, None]
        d["nb_extra_calls"] = [3, 9, 11, np.nan]
        s = st.arm_summary(d, n_boot=200).iloc[0]
        assert s.share_min_image == pytest.approx(2 / 3) and s.share_min_text == pytest.approx(1 / 3)
        assert s.n_pair == 2 and s.median_bp_gap == pytest.approx(0.002) and s.share_bp_text_step == 0.5
        assert s.median_nb_extra_calls == 9


class TestTablesAndCli:
    def make(self):
        cells = [f"c{i}" for i in range(10)]
        a = frame("a0_directed", [3, 8, 12, 20, 40, 90, None, None, 5, 60], cells=cells)
        b = frame("a9_deepjanus", [30, 80, None, None, None, None, None, None, 50, None], cells=cells)
        return pd.concat([a, b], ignore_index=True)

    def test_run_all_writes_csv_and_latex(self, tmp_path):
        df = self.make()
        res = st.run_all(df, "a0_directed", tmp_path, budgets=[30, 100], n_boot=300)
        for f in ("summary.csv", "summary.tex", "curve.csv", "curve_budgets.tex", "paired.csv", "paired.tex"):
            assert (tmp_path / f).is_file(), f
        tex = (tmp_path / "paired.tex").read_text()
        assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
        assert "DeepJanus-style" in tex and "a9_deepjanus" not in tex.replace(r"\_", "")
        assert "Magnitude of the smallest crossing" in tex and "MMBA" in tex
        s = res["summary"].set_index("arm")
        assert s.loc["a0_directed", "n_crossed"] == 8 and s.loc["a9_deepjanus", "n_crossed"] == 3
        assert s.loc["a0_directed", "n_min"] == 8 and s.loc["a9_deepjanus", "n_min"] == 3
        assert s.loc["a9_deepjanus", "median_magnitude"] == pytest.approx(0.5)            # 0.3, 0.5, 0.8
        assert s.loc["a0_directed", "median_first_cross_eval"] == pytest.approx(30.0)   # sorted: 3,5,8,12,20,40,60,90,100,100 -> (20+40)/2

    def test_load_results_normalises_booleans(self, tmp_path):
        p = tmp_path / "r.csv"
        self.make().to_csv(p, index=False)
        df = st.load_results([p])
        assert df["crossed"].dtype == bool and df["crossed"].sum() == 11

    def test_cli(self, tmp_path):
        p = tmp_path / "r.csv"
        self.make().to_csv(p, index=False)
        assert st.main([str(p), "--ref", "a0_directed", "--out", str(tmp_path / "t"), "--n-boot", "200",
                        "--budgets", "30,100"]) == 0
        assert (tmp_path / "t" / "paired.csv").is_file()


class TestPlot:
    def test_writes_pdf(self, tmp_path):
        from mmb.analysis import plots
        df = TestTablesAndCli().make()
        out = plots.plot_crossing_curve(df, tmp_path / "f" / "curve.pdf", ref_arm="a0_directed",
                                        max_budget=100, vlines=[30], n_boot=200)
        data = out.read_bytes()
        assert data.startswith(b"%PDF") and len(data) > 1500

    def test_palette_is_colour_blind_safe_size(self):
        from mmb.analysis import plots
        assert len(set(plots.PALETTE)) == len(plots.PALETTE) == 8
        assert plots.COLUMN_WIDTH_IN < 3.5
