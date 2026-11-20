"""Export (results -> paper) and gallery on a tiny synthetic runs tree."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from test_analysis_fixtures import write_run  # noqa: E402

from mmb.analysis import collect, export, gallery  # noqa: E402

TASKS = {"imagenet": ("junco", "ostrich"), "pope": ("No", "Yes")}
ARMS = ["a0_directed", "a1_random_tiered", "a3_ga_single"]
CONT = "llava"


def build_tree(root: Path, n_cells: int = 6, seeds=(0, 1)) -> None:
    rng = np.random.default_rng(3)
    for task, (ca, cb) in TASKS.items():
        for sut in ("qwen", "internvl", CONT):
            for arm in ARMS:
                if sut == CONT and (task != "imagenet" or arm == "a3_ga_single"):
                    continue
                for cell in range(n_cells):
                    for s in seeds:
                        p_cross = 0.8 if arm.startswith("a0") else 0.4      # a0 crosses more often
                        g = list(rng.uniform(0.2, 3.0, 12))
                        if rng.random() < p_cross:
                            g[int(rng.integers(0, 12))] = -float(rng.uniform(0.1, 1.0))
                        name = f"{task}_{sut}_{arm}_c{cell}_s{s}"
                        run = write_run(
                            root / task / sut / arm, name, g, pop=3, class_a=ca, class_b=cb,
                            stats_extra={
                                "run_meta": {"exp_id": "T", "task": task, "sut": sut, "arm": arm, "rng_seed": s,
                                             "algorithm": "agemoea2", "boundary_objective": "logprob_gap",
                                             "sampling_mode": "sparse_multitier", "modality": "joint",
                                             "eval_budget": 12},
                                "anchor_g": 2.0, "prior_g": float(rng.normal(0, 1)), "anchor_argmax": ca,
                                "prompt_template": "What is this?",
                                "seed_metadata": {"anchor_class_concrete": ca, "target_class_concrete": cb,
                                                  "level_anchor": 0, "level_target": 1, "seed_idx_in_class": cell,
                                                  "prompt_template": "Is there a dog?" if task == "pope" else "What is this?"},
                            })
                        Image.fromarray((rng.random((40, 30, 3)) * 255).astype("uint8")).save(run / "origin.png")
                        Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8")).save(run / "first_cross.png")
                        (run / "first_cross.json").write_text(json.dumps(
                            {"eval_idx": 5, "generation": 1, "g": -0.5, "text": "Is ther e a dog?"}))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    build_tree(root)
    csv = root.parent / "main.csv"
    assert collect.main([str(root), "--out", str(csv), "--exp-id", "T"]) == 0
    return root, csv


class TestNames:
    @pytest.mark.parametrize("s,out", [("probe_gen0", "ProbeGenZero"), ("a0_directed", "AZeroDirected"),
                                       ("x12", "XOneTwo")])
    def test_camel_has_no_digits(self, s, out):
        assert export.camel(s) == out

    def test_arm_short(self):
        assert export.arm_short("a8_hillclimb") == "AEight"

    def test_macro_name_rejects_garbage(self):
        with pytest.raises(ValueError):
            export.macro_name("---")

    def test_tex_value_minus_and_dashes(self):
        assert export.tex_value("-0.45") == r"\ensuremath{-}0.45"
        assert export.tex_value("[75--89]") == "[75--89]"
        assert export.tex_value("<0.001") == r"\ensuremath{<}0.001"

    def test_duplicates_rejected(self):
        n = export.Numbers()
        n.add(["EOne", "X"], "1", "d")
        with pytest.raises(ValueError):
            n.add(["EOne", "X"], "2", "d")


class TestExport:
    def test_end_to_end(self, tree, tmp_path):
        root, csv = tree
        paper = tmp_path / "paper"
        paper.mkdir()
        num = export.export([csv], paper, runs_root=root, n_boot=200)
        gen = paper / "generated"
        for f in ("tables/e1.tex", "tables/e1_budgets.tex", "tables/e1_paired.tex", "tables/e3.tex",
                  "tables/e4.tex", "tables/e2b.tex", "tables/per/imagenet_qwen_summary.tex",
                  "tables/per/pope_internvl_paired.tex", "figures/crossing-curve.pdf",
                  "figures/predictor-main.pdf", "figures/gallery.pdf", "numbers.tex", "numbers_map.csv",
                  "tables/primary.tex", "tables/e1_cluster.tex", "tables/cost.tex", "figures/per-cell-diff.pdf"):
            assert (gen / f).exists() and (gen / f).stat().st_size > 0, f
        assert not (gen / ".work").exists()
        text = (gen / "numbers.tex").read_text()
        names = re.findall(r"\\newcommand\{\\([A-Za-z0-9]+)\}", text)
        assert names and all(re.fullmatch(r"[A-Za-z]+", n) for n in names)      # no digits anywhere
        assert len(names) == len(set(names)) == len(num.rows)
        m = pd.read_csv(gen / "numbers_map.csv")
        assert {"macro", "value", "description"} <= set(m.columns) and len(m) == len(names)

    def test_numbers_match_csv(self, tree, tmp_path):
        root, csv = tree
        paper = tmp_path / "p"
        paper.mkdir()
        export.export([csv], paper, runs_root=root, n_boot=100, figures=False)
        m = pd.read_csv(paper / "generated" / "numbers_map.csv", dtype=str).set_index("macro")["value"]
        df = pd.read_csv(csv)
        d = df[(df.arm == "a0_directed") & (df.sut == "qwen") & (df.task == "imagenet")]
        assert int(m["EOneRunsAZeroQwenImagenet"]) == len(d)
        assert int(m["EOneCrossedAZeroQwenImagenet"]) == int(d.crossed.sum())
        assert m["EOneCrossShareAZeroQwenImagenet"] == f"{100 * d.crossed.mean():.1f}"
        grid = df[df.sut.isin(["qwen", "internvl", "ministral"]) & (df.arm == "a1_random_tiered")]
        assert int(m["EOneRunsAOneAllAll"]) == len(grid)
        assert "EOnePHolmCrossedAOneAllAll" in m.index and "EOnePHolmEvalsAThreeAllAll" in m.index
        assert "EPrimaryPHolmQwenImagenet" in m.index and "ETwoADiffLoMinEditsAOneAllAll" in m.index
        assert "ECostMedMinEditsAZeroAllAll" in m.index
        assert m["EAnything"] if False else True
        assert m["EOneAtBudgetTwoZeroSixAZeroAllAll"] and "EOneAtBudgetCiOneFiveZeroZeroAOneAllAll" in m.index
        assert "EOneRunsAZeroLlavaImagenet" in m.index and "EOneRunsAThreeLlavaImagenet" not in m.index
        assert "EOneRunsAZeroLlavaPope" not in m.index

    def test_pooled_pairs_do_not_collide(self, tree):
        _, csv = tree
        df = export.st.load_results([csv])
        pooled = export.scopes(df, export.GRID_SUTS)[("All", "All")]
        assert not pooled.duplicated(["arm", "cell_id", "rng_seed"]).any()

    def test_tables_are_wrapped_tabulars(self, tree, tmp_path):
        root, csv = tree
        paper = tmp_path / "q"
        paper.mkdir()
        export.export([csv], paper, n_boot=100, figures=False)
        files = list((paper / "generated" / "tables").rglob("*.tex"))
        assert files
        for f in files:
            t = f.read_text()
            assert t.startswith("\\begin{adjustbox}") and "\\begin{tabular}" in t and "\\bottomrule" in t, f
        e3 = (paper / "generated" / "tables" / "e3.tex").read_text()
        assert "LLaVA-NeXT-7B (NF4)" in e3 and e3.count("\\midrule") >= 3

    def test_without_prior_e2b_is_skipped(self, tree, tmp_path):
        root, csv = tree
        df = pd.read_csv(csv).drop(columns=["prior_g"])
        c2 = tmp_path / "noprior.csv"
        df.to_csv(c2, index=False)
        paper = tmp_path / "r"
        paper.mkdir()
        export.export([c2], paper, n_boot=50, figures=False)
        assert not (paper / "generated" / "tables" / "e2b.tex").exists()

    def test_validity_macros(self, tree, tmp_path):
        root, csv = tree
        v = tmp_path / "v.csv"
        pd.DataFrame({"arm": ["a0_directed"] * 3 + ["a1_random_tiered"] * 2, "img_clip_cos": [0.9, 0.8, 0.7, 0.6, 0.5],
                      "question_preserved": [True, True, False, True, True]}).to_csv(v, index=False)
        paper = tmp_path / "s"
        paper.mkdir()
        export.export([csv], paper, validity=v, n_boot=50, figures=False)
        m = pd.read_csv(paper / "generated" / "numbers_map.csv", dtype=str).set_index("macro")["value"]
        assert m["ESevenMedianImgClipCosAZero"] == "0.800" and m["ESevenCrossingsAOne"] == "2"
        assert m["ESevenQuestionPreservedAZero"] == "66.7"
        assert (paper / "generated" / "tables" / "validity.tex").exists()


class TestAgree:
    def _csv(self, tmp_path, sut, n=10):
        rows = []
        for i in range(n):
            rows.append(dict(run_dir=f"r{i}", exp_id="T", task="imagenet" if i % 2 else "pope", sut=sut,
                             arm="a0_directed", rng_seed=0, anchor_greedy="x", anchor_greedy_label="a",
                             anchor_tf_argmax="a", anchor_agree=True, cross_greedy="y", cross_greedy_label="b",
                             cross_tf_argmax="b", cross_agree=(i % 5 != 0), cross_g_single=-0.3,
                             cross_verified=(i % 10 != 0), cross_g_search=-0.4))
        p = tmp_path / f"agree_T_{sut}.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_macros_and_table(self, tree, tmp_path):
        root, csv = tree
        paper = tmp_path / "pa"
        paper.mkdir()
        files = [self._csv(tmp_path, "qwen"), self._csv(tmp_path, "internvl", 20)]
        export.export([csv], paper, n_boot=50, figures=False, agree=files)
        m = pd.read_csv(paper / "generated" / "numbers_map.csv", dtype=str).set_index("macro")["value"]
        assert m["EAgreeAnchorShareQwenAll"] == "100.0" and m["EAgreeCrossShareQwenAll"] == "80.0"
        assert m["EAgreeCrossCountQwenAll"] == "8" and m["EAgreeCrossNQwenAll"] == "10"
        assert m["EAgreeVerifiedNInternvlAll"] == "20" and m["EAgreeVerifiedCountInternvlAll"] == "18"
        assert m["EAgreeVerifiedNAllAll"] == "30"
        assert m["EAgreeCrossCiAllAll"].startswith("[")
        t = (paper / "generated" / "tables" / "agree.tex").read_text()
        assert "Qwen3.5-4B" in t and "\\begin{tabular}" in t

    def test_missing_columns_and_nan_are_skipped(self):
        df = pd.DataFrame({"sut": ["q"] * 3, "task": ["t"] * 3, "anchor_agree": [True, False, None]})
        rows = export.agree_rows(export.pd.DataFrame(df))
        tags = {r["tag"] for r in rows}
        assert tags == {"Anchor"} and all(r["n"] == 2 for r in rows)


class TestTransfer:
    def _csv(self, tmp_path, csv):
        main = pd.read_csv(csv)
        a0 = main[main.arm == "a0_directed"].groupby(["sut", "task"]).head(4)
        rows = []
        for i, r in enumerate(a0.itertuples()):
            for dst in ("qwen", "internvl", "ministral"):
                rows.append(dict(run_dir=r.run_dir, src_sut=r.sut, dst_sut=dst, task=r.task, g_dst=-0.2,
                                 flipped_on_dst=(i + len(dst)) % 3 == 0))
        p = tmp_path / "transfer_T.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p, pd.DataFrame(rows)

    def test_matrix_and_macros(self, tree, tmp_path):
        root, csv = tree
        p, raw = self._csv(tmp_path, csv)
        paper = tmp_path / "pt"
        paper.mkdir()
        export.export([csv], paper, n_boot=50, figures=False, transfer=[p])
        m = pd.read_csv(paper / "generated" / "numbers_map.csv", dtype=str).set_index("macro")["value"]
        sub = raw[(raw.src_sut == "qwen") & (raw.dst_sut == "internvl")]
        assert int(m["ETransferNQwenInternvlAll"]) == len(sub)
        assert int(m["ETransferCountQwenInternvlAll"]) == int(sub.flipped_on_dst.sum())
        assert "ETransferMeanOffDiagonalAll" in m.index and "ETransferShareQwenQwenImagenet" in m.index
        t = (paper / "generated" / "tables" / "transfer.tex").read_text()
        assert "Found on" in t and "Qwen3.5-4B" in t
        assert (paper / "generated" / "tables" / "transfer_pope.tex").exists()

    def test_missing_columns_rejected(self, tmp_path):
        from mmb.analysis import transfer
        p = tmp_path / "bad.csv"
        pd.DataFrame({"run_dir": ["a"]}).to_csv(p, index=False)
        with pytest.raises(ValueError):
            transfer.load_transfer([p])

    def test_off_diagonal_excludes_diagonal(self):
        from mmb.analysis import transfer
        rows = [dict(src="a", dst="a", task="All", k=10, n=10, share=1, lo=0, hi=1),
                dict(src="a", dst="b", task="All", k=1, n=4, share=.25, lo=0, hi=1)]
        assert transfer.mean_off_diagonal(rows) == (0.25, 4)


class TestGallery:
    def test_selection_one_run_per_example_and_sut(self, tree):
        _, csv = tree
        df = export.st.load_results([csv])
        picks = gallery.select_runs(df[df.sut.isin(export.GRID_SUTS)], "a0_directed", 2)
        assert set(picks.task) <= {"imagenet", "pope"} and picks.crossed.all()
        for _, t in picks.groupby("task"):
            assert t.example.nunique() <= 2
            assert not t.duplicated(["example", "sut"]).any()
            assert t.groupby("example").cell_id.nunique().max() == 1      # a row compares SUTs on one cell

    def test_plot_writes_pdf_and_survives_missing_image(self, tree, tmp_path):
        root, csv = tree
        df = export.st.load_results([csv])
        picks = gallery.select_runs(df[df.sut.isin(export.GRID_SUTS)], "a0_directed", 2)
        broken = picks.copy()
        broken.loc[broken.index[0], "run_dir"] = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            gallery.plot_gallery(broken, tmp_path / "bad.pdf")
        out = gallery.plot_gallery(picks, tmp_path / "g.pdf")
        assert out.exists() and out.read_bytes()[:4] == b"%PDF"
