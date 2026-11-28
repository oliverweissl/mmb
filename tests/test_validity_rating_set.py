"""Human-study item sampler, static rating page and rating analysis on a synthetic runs tree."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_export as tae  # noqa: E402

from mmb.analysis import collect  # noqa: E402
from mmb.validity import analyze_ratings, rating_set  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    tae.build_tree(root, n_cells=8)
    # give pope seeds an object and ground truth like the real stats.json
    for sj in root.rglob("stats.json"):
        s = json.loads(sj.read_text())
        if "pope" in str(sj):
            s["seed_metadata"].update(object="dog", gt="yes")
        sj.write_text(json.dumps(s))
        run = sj.parent
        if (run / "first_cross.png").exists():      # minimal crossing + codec reference like the real runs
            shutil.copy(run / "first_cross.png", run / "min_cross.png")
            shutil.copy(run / "first_cross.json", run / "min_cross.json")
        if (run / "origin.png").exists():
            shutil.copy(run / "origin.png", run / "anchor_recon.png")
    csv = root.parent / "main.csv"
    assert collect.main([str(root), "--out", str(csv), "--exp-id", "T"]) == 0
    out = root.parent / "study"
    info = rating_set.build(csv, out, n=24, controls=6, arms=("a0_directed", "a9_deepjanus"), seed=1)
    return csv, out, info


class TestSampler:
    def test_counts_and_files(self, built):
        csv, out, info = built
        m = json.loads((out / "rater_view.json").read_text())
        assert info["edited"] == 24 and info["controls"] == 6 and m["n_items"] == 30
        for it in m["items"]:
            assert (out / it["image"]).exists()
        assert len(list((out / "items").glob("*.png"))) == 30

    def test_manifest_is_blind_and_key_is_private(self, built):
        csv, out, _ = built
        for f in ("rater_view.json", "rate.html"):
            text = (out / f).read_text()
            assert not re.search(r"a0_directed|a9_deepjanus|qwen|internvl|ministral|llava|run_dir|/tmp/", text)
        k = pd.read_csv(out / "manifest.csv", keep_default_na=False)
        assert {"item_id", "stratum", "run_id", "task", "sut", "arm", "run_dir", "cell_id", "rng_seed", "candidates",
                "anchor_label", "gt", "img_lpips", "txt_word_edit"} <= set(k.columns)
        assert set(k.stratum) == {"edited", "control"} and (k[k.stratum == "control"].arm == "").all()

    def test_even_over_task_and_arm_and_spread_over_suts(self, built):
        _, out, _ = built
        k = pd.read_csv(out / "manifest.csv", keep_default_na=False)
        e = k[k.stratum == "edited"]
        sz = e.groupby(["task", "arm", "sut"]).size()
        assert sz.max() - sz.min() <= 1                               # even over method x task x model
        assert e.groupby(["task", "arm"]).sut.nunique().min() >= 2

    def test_controls_use_origin_and_original_question(self, built):
        _, out, _ = built
        k = pd.read_csv(out / "manifest.csv", keep_default_na=False)
        c = k[k.stratum == "control"]
        assert (c.shown_question == c.original_question).all()
        e = k[k.stratum == "edited"]
        assert (e.shown_question != e.original_question).all()       # synthetic crossings carry an edited text

    def test_agree_filter_keeps_only_verified(self, built, tmp_path):
        csv, _, _ = built
        df = pd.read_csv(csv)
        a0 = df[(df.arm == "a0_directed") & df.crossed]
        keep = list(a0.run_dir[:5])
        ag = pd.DataFrame({"run_dir": list(a0.run_dir[:10]), "cross_verified": [True] * 5 + [False] * 5})
        p = tmp_path / "agree.csv"
        ag.to_csv(p, index=False)
        info = rating_set.build(csv, tmp_path / "o", n=10, controls=2, arms=("a0_directed",), seed=3, agree=[p])
        k = pd.read_csv(tmp_path / "o" / "manifest.csv", keep_default_na=False)
        assert set(k[k.stratum == "edited"].run_dir) <= set(keep)

    def test_minimal_crossing_is_the_stimulus(self, built):
        _, out, _ = built
        k = pd.read_csv(out / "manifest.csv", keep_default_na=False)
        e = k[k.stratum == "edited"]
        for r in e.itertuples():
            mc = json.loads((Path(r.run_dir) / "min_cross.json").read_text())
            assert r.shown_question == (mc.get("text") or r.original_question)

    def test_missing_method_is_skipped_with_warning(self, built, tmp_path):
        csv, _, _ = built
        with pytest.warns(UserWarning, match="a8_hillclimb"):
            info = rating_set.build(csv, tmp_path / "w", n=8, controls=2,
                                    arms=("a0_directed", "a8_hillclimb"), seed=2)
        assert info["edited"] > 0 and all(s.startswith("a0_directed") for s in info["strata"])

    def test_deterministic(self, built, tmp_path):
        csv, out, _ = built
        rating_set.build(csv, tmp_path / "again", n=24, controls=6, arms=("a0_directed", "a9_deepjanus"), seed=1)
        assert (tmp_path / "again" / "manifest.csv").read_text() == (out / "manifest.csv").read_text()


class TestPage:
    def test_single_file_with_inlined_manifest_and_consent_gate(self, built):
        _, out, _ = built
        html = (out / "rate.html").read_text()
        assert "var M = {" in html and "__MANIFEST__" not in html and "agree" in html
        assert "data:image/png;base64," in html     # images embedded: the page is one file
        assert "http://" not in html and "https://" not in html and "fetch(" not in html and "XMLHttpRequest" not in html
        assert "localStorage" in html            # draft only; nothing leaves the browser

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_script_parses(self, built, tmp_path):
        _, out, _ = built
        html = (out / "rate.html").read_text()
        js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
        f = tmp_path / "page.js"
        f.write_text(js)
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_page_flow_end_to_end_with_dom_stub(self, built):
        _, out, _ = built
        r = subprocess.run(["node", str(Path(__file__).parent / "rating_page_harness.js"), str(out / "rate.html")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        res = json.loads(r.stdout)
        m = json.loads((out / "rater_view.json").read_text())
        assert res["steps"] == m["n_items"]
        rows = [l for l in res["csv"].split("\n")[1:] if l]
        assert len(rows) == m["n_items"] and res["csv"].startswith("rater_id,item_id,order_index,answer_content,answer_same_question")
        ids = {l.split(",")[1] for l in rows}
        assert ids == {i["id"] for i in m["items"]}            # every item rated exactly once


class TestAnalysis:
    def _ratings(self, out, tmp_path):
        k = pd.read_csv(out / "manifest.csv", keep_default_na=False)
        rows = []
        for rater in ("r1", "r2", "r3"):
            for i, r in k.iterrows():
                if r.task == "pope":
                    ans = r["gt"] if i % 4 else ("no" if r["gt"] == "yes" else "yes")
                else:
                    ans = "yes" if i % 4 else "__cant"
                rows.append(dict(rater_id=rater, item_id=r.item_id, order_index=i, answer_content=ans,
                                 answer_same_question="yes", rt_ms=1000, client_time="x"))
        p = tmp_path / "ratings_r.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p, k

    def test_valid_rules_and_summary(self, built, tmp_path):
        _, out, _ = built
        p, k = self._ratings(out, tmp_path)
        d = analyze_ratings.load(out / "manifest.csv", [p])
        assert len(d) == 3 * len(k)
        pope = d[d.task == "pope"]
        assert pope[pope.answer_content == pope["gt"]].valid.all()
        assert not pope[pope.answer_content != pope["gt"]].valid.any()
        inet = d[d.task != "pope"]
        assert inet[inet.answer_content == "yes"].valid.all() and not inet[inet.answer_content == "__cant"].valid.any()
        s = analyze_ratings.summarize(d, n_boot=200)
        assert {"stratum", "task", "arm", "items", "valid_share", "lo", "hi", "raters"} <= set(s.columns)
        assert (s.lo <= s.valid_share + 1e-9).all() and (s.valid_share <= s.hi + 1e-9).all()
        assert s.raters.max() == 3

    def test_json_ratings_agreement_and_calibration(self, built, tmp_path):
        _, out, _ = built
        p, k = self._ratings(out, tmp_path)
        r = pd.read_csv(p, dtype=str)
        jp = []
        for rid, g in r.groupby("rater_id"):
            f = tmp_path / f"ratings_{rid}.json"
            f.write_text(json.dumps({"rater_id": rid, "version": 2, "rows": g.to_dict("records")}))
            jp.append(f)
        d = analyze_ratings.load(out / "manifest.csv", jp)
        assert len(d) == 3 * len(k)
        a = analyze_ratings.agreement(d)
        allv = a[(a.group == "all") & (a.variable == "valid")].iloc[0]
        assert allv.krippendorff_alpha == pytest.approx(1.0) and allv.gwet_ac1 == pytest.approx(1.0)
        ids = sorted(d.item_id.unique())
        d["img_lpips"] = d.item_id.map({i: str(0.01 * n) for n, i in enumerate(ids)})
        d["txt_word_edit"] = d.item_id.map({i: str(0.02 * (n % 7)) for n, i in enumerate(ids)})
        c = analyze_ratings.proxy_calibration(d, n_boot=50)
        assert len(c) > 0

    def test_agreement_known_values(self):
        units = [["a", "a"], ["a", "b"], ["b", "b"], ["a", "a"]]
        alpha = analyze_ratings.coincidence_alpha(units, levels=("a", "b"))
        assert alpha == pytest.approx(0.5333, abs=1e-3)
        assert analyze_ratings.gwet_ac1([["a", "a"], ["b", "b"]], ("a", "b")) == pytest.approx(1.0)

    def test_duplicate_submissions_collapse(self, built, tmp_path):
        _, out, _ = built
        p, k = self._ratings(out, tmp_path)
        d = analyze_ratings.load(out / "manifest.csv", [p, p])
        assert len(d) == 3 * len(k)


class TestHumanExport:
    def test_macros_and_table(self, built, tmp_path):
        from mmb.analysis import export
        csv, out, _ = built
        k = pd.read_csv(out / "manifest.csv", keep_default_na=False)
        rows = [dict(rater_id=r, item_id=i.item_id, order_index=0,
                     answer_content=(i.gt if i.task == "pope" else "yes"), answer_same_question="yes",
                     rt_ms=1, client_time="x")
                for r in ("a", "b") for i in k.itertuples()]
        p = tmp_path / "r.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        s = analyze_ratings.summarize(analyze_ratings.load(out / "manifest.csv", [p]), n_boot=100)
        sp = tmp_path / "validity_human.csv"
        s.to_csv(sp, index=False)
        paper = tmp_path / "paper"
        paper.mkdir()
        export.export([csv], paper, n_boot=50, figures=False, human=sp)
        m = pd.read_csv(paper / "generated" / "numbers_map.csv", dtype=str).set_index("macro")["value"]
        assert m["EHumanShareEditedImagenetAZero"] == "100.0" and "EHumanShareControlImagenetControl" in m.index
        assert (paper / "generated" / "tables" / "human_validity.tex").exists()
