"""Direction arrows and bold-best conventions of the generated tables (mmb.analysis.tablefmt)."""
import numpy as np
import pandas as pd

from mmb.analysis import stats as st
from mmb.analysis import tablefmt as tf
from mmb.analysis import validity_cost as vc


def test_head_arrows_follow_the_direction_map():
    assert tf.head("Crossed", "crossed").endswith(r"\uparrow$")
    assert tf.head("LPIPS", "lpips").endswith(r"\downarrow$")
    assert tf.head("Runs", None) == "Runs" and tf.head("p", "p_value") == "p"
    for key in ("crossed", "band", "vis_identical", "budget_share"):
        assert tf.DIRECTION[key] == tf.UP
    for key in ("first_cross_eval", "floor", "min_edits", "lpips", "txt_dist", "wall"):
        assert tf.DIRECTION[key] == tf.DOWN


def test_generation_zero_and_budget_30_are_neutral():
    assert "gen0" not in tf.DIRECTION and tf.head("Gen.0", "gen0") == "Gen.0"
    curve = pd.DataFrame([dict(arm=a, budget=b, share=s, boot_lo=s, boot_hi=s)
                          for a, s in (("a0_directed", 0.9), ("a1_random_tiered", 0.4)) for b in (30, 100)])
    tex = st.budgets_tex(curve, [30, 100])
    header = [l for l in tex.splitlines() if l.startswith("Arm")][0]
    assert header.count("uparrow") == 1 and header.split("&")[1].count("uparrow") == 0   # arrow only on the larger budget
    row = [l for l in tex.splitlines() if "Directed" in l][0]
    assert row.split("&")[1].count("textbf") == 0 and row.split("&")[2].count("textbf") == 1


def test_best_indices_ties_and_degenerate_groups():
    assert tf.best_indices([0.5, 0.7, 0.7], "crossed") == {1, 2}              # ties: all tied values
    assert tf.best_indices([10, 4, 4], "floor") == {1, 2}                      # lower is better
    assert tf.best_indices([0.5, 0.5], "crossed") == set()                     # nothing to distinguish
    assert tf.best_indices([np.nan, 0.3], "crossed") == set()                  # a single value has no best
    assert tf.best_indices([0.501, 0.502], "crossed", ["50.1", "50.2"]) == {1}
    assert tf.best_indices([0.5004, 0.5001, 0.3], "crossed", ["50.0", "50.0", "30.0"]) == {0, 1}   # ties by display


def test_summary_table_has_arrows_and_one_best_per_column():
    s = pd.DataFrame(dict(arm=["a0_directed", "a1_random_tiered"], n_runs=[36, 36], n_crossed=[30, 20],
                          share_crossed=[30 / 36, 20 / 36], wilson_lo=[.7, .4], wilson_hi=[.9, .7],
                          share_gen0=[.1, .1], gen0_lo=[0, 0], gen0_hi=[.2, .2], share_band=[.5, .2],
                          share_label_flip=[.9, .6], median_first_cross_eval=[100., 400.], mfc_lo=[80., 300.],
                          mfc_hi=[120., 500.], fc_q1=[1, 1], fc_q3=[1, 1], median_floor_nats=[0.01, 0.2],
                          mfl_lo=[0, 0], mfl_hi=[1, 1]))
    tex = st.summary_tex(s)
    assert r"\uparrow" in tex and r"\downarrow" in tex
    rows = [l for l in tex.splitlines() if l.startswith(("Directed", "Random"))]
    assert r"\textbf{83.3}" in rows[0] and r"\textbf" not in rows[1].split("&")[2]
    assert r"\textbf{100}" in rows[0] and r"\textbf{0.01}" in rows[0]
    assert "[" not in tex                                                         # no intervals in the tables
    assert r"\textbf{36}" not in tex                                            # counts are never bold
    gen0 = [r.split("&")[3] for r in rows]
    assert all(r"\textbf" not in g for g in gen0)                               # all tied: no best


def test_validity_table_bold_within_task_block():
    t = pd.DataFrame(dict(task=["imagenet", "imagenet", "pope", "pope"], arm=["a0_directed", "a1_random_tiered"] * 2,
                          n_cross=[5, 5, 5, 5], n_runs=[10] * 4, med_lpips=[0.05, 0.10, 0.20, 0.10], med_txt=[0.2] * 4,
                          img_share=[1, .8, .6, .9], txt_share=[0] * 4, both_share=[0] * 4, cons_share=[0, 0, 0, 0],
                          cons_lo=[0] * 4, cons_hi=[0] * 4, rho_eval_lpips=[0] * 4, rho_eval_txt=[0] * 4,
                          med_vis_cos=[.9] * 4, visid_share=[0] * 4, visone_share=[0] * 4))
    tex = vc.table_tex(t, {"imagenet": "ImageNet", "pope": "POPE"}, objective=False)
    lines = [l for l in tex.splitlines() if "Directed" in l or "Random" in l]
    assert r"\textbf{0.05}" in lines[0] and r"\textbf{0.10}" in lines[3]       # LPIPS: lower is better, per task block
    assert r"\textbf{0.10}" not in lines[1] and r"\textbf{0.20}" not in lines[2]
    assert r"\downarrow" in tex


def test_effect_cell_magnitude_bullet_and_dagger():
    assert tf.magnitude(1.2) == "L" and tf.magnitude(-1.2) == "L" and tf.magnitude(0.8) == "M" and tf.magnitude(0.5) == "S"
    assert tf.magnitude(0.2) == "S" and tf.magnitude(float("nan")) is None
    assert tf.effect_cell("+0.80", 0.8, 0.01) == r"+0.80 \effM{}$^{\dagger}$"
    assert tf.effect_cell("-12", -1.5, 0.2) == r"-12 \effL{}"                    # not significant after Holm: no dagger
    assert tf.effect_cell("+3", 0.1, 0.04) == r"+3 \effS{}$^{\dagger}$"
    assert tf.effect_cell("--", float("nan"), float("nan")) == "--"


def test_cohens_dz():
    assert tf.cohens_dz([1, 1, 1]) == 9.99 and tf.cohens_dz([-1, -1]) == -9.99 and tf.cohens_dz([0, 0, 0]) == 0.0
    assert abs(tf.cohens_dz([1.0, 3.0]) - (2.0 / (2 ** 0.5))) < 1e-9 and np.isnan(tf.cohens_dz([1.0]))


def test_rq_tables_blocks_bold_and_partner_rule():
    from mmb.analysis import rq_tables as rq
    rows = []
    for t in ("imagenet", "pope"):
        for a, cr, lp, np_ in (("a0_directed", 90.0, 0.05, 20), ("a1_random_tiered", 50.0, 0.08, 2)):
            rows.append(dict(task=t, sut="qwen", arm=a, n_runs=10, CR=cr, S=0.9, SCR=cr * 0.9, lpips=lp, lpips_bp=lp if np_ >= 3 else float("nan"), n_partner=np_,
                             d_edit=0.0 if a.startswith("a0") else 10.0, d_sem=0.01, first_cross_eval=50.0 if a.startswith("a0") else 90.0,
                             min_cross_eval=400.0, wall=170.0))
    m = pd.DataFrame(rows)
    t1 = rq.rq1_tex(m)
    assert "Qwen3.5-4B" in t1 and "ImageNet" in t1 and "POPE" in t1 and r"\uparrow" in t1 and r"\downarrow" in t1 and "SCR" in t1
    d_row = [l for l in t1.splitlines() if "Directed" in l][0]
    r_row = [l for l in t1.splitlines() if "Random" in l][0]
    assert r"\textbf{90}" in d_row and r"\textbf{0.05}" in d_row and "LPIPS" in t1 and "bp" not in t1
    t2 = rq.rq2_tex(m)
    assert "Runtime (s)" in t2 and "Calls to first" in t2 and r"\textbf{50}" in t2
    ab = rq.ablation(pd.DataFrame(dict(arm=["a0_directed"] * 4, rng_seed=0, crossed=[True, True, True, False], first_cross_eval=[10, 20, 30, 40],
                                      wall_s=100.0, min_cross_eval=[50, 60, 70, float("nan")], run_dir=list("abcd"), edits_img_at_min=2, edits_txt_at_min=0)),
                     pd.DataFrame(dict(view=["min"] * 3 + ["first"] * 3, run_dir=list("abcabc"), vis_char_dist=0.0, txt_minilm_cos=1.0,
                                       has_partner=True, pair_lpips=0.02, txt_edited=False, img_lpips=0.1)))
    tab = rq.ablation_tex(ab)
    assert "Full" in tab and "w/o Phase 2" in tab and "Variant" in tab
    assert ab.loc[ab.label == "w/o Phase 2", "lpips_bp"].isna().all()


def test_validity_and_generalization_main_tables():
    from mmb.analysis import export as ex
    vct = pd.DataFrame(dict(task=["imagenet", "imagenet"], arm=["a0_directed", "a1_random_tiered"], n_cross=[8, 5],
                            med_lpips=[0.012, 0.030], med_vis_char=[0.42, 0.40]))
    tex = ex.validity_main_tex(vct, {"imagenet": "ImageNet"})
    assert "Image change" in tex and "Text change" in tex and r"\textbf{0.012}" in tex and r"\textbf{40}" in tex
    prim = pd.DataFrame(dict(sut=["qwen", "qwen"], task=["imagenet", "pope"], ref_value=[0.9, 0.8], arm_value=[0.5, 0.8],
                             diff=[-0.4, 0.0], dz=[-1.3, 0.0], p_holm_strata=[0.004, 0.9]))
    g = ex.generalization_main_tex(prim, [dict(task="All", src="qwen", dst="internvl", k=3, n=10, share=0.3, lo=0, hi=1)])
    assert "Transfer rate" in g and r"-40 \effL{}$^{\dagger}$" in g and "30" in g and r"\textbf{90}" in g


def test_data_status_partial_until_every_planned_config_has_all_cells(tmp_path):
    from mmb.analysis import export as ex
    cfg = tmp_path / "configs" / "generated" / "MAIN"
    cfg.mkdir(parents=True)
    for n in ("a", "b"):
        (cfg / f"{n}.yaml").write_text(f"save_dir: runs/MAIN/{n}\n")
    rows = [dict(run_dir=str(tmp_path / "runs" / "MAIN" / "a" / f"r{i}"), cell_id=f"c{i}", complete=True) for i in range(3)]
    df = pd.DataFrame(rows)
    assert ex.data_status(df, cfg, tmp_path, n_cells=3) == ("partial", 1, 2)          # b has no runs
    rows += [dict(run_dir=str(tmp_path / "runs" / "MAIN" / "b" / f"r{i}"), cell_id=f"c{i}", complete=i < 2) for i in range(3)]
    assert ex.data_status(pd.DataFrame(rows), cfg, tmp_path, n_cells=3)[0] == "partial"  # one run of b incomplete
    rows[-1]["complete"] = True
    assert ex.data_status(pd.DataFrame(rows), cfg, tmp_path, n_cells=3) == ("complete", 2, 2)


def test_crossing_only_edit_metrics_use_pairs_where_both_arms_crossed():
    from mmb.analysis import paired as pr
    n = 8
    base = dict(cell_id=[f"c{i}" for i in range(n)], rng_seed=[0] * n, task=["imagenet"] * n, sut=["x"] * n)
    ref = pd.DataFrame(dict(base, arm="a0_directed", crossed=[True] * 4 + [False] * 4, min_edits_cross=[10, 12, 14, 16, 300, 300, 300, 300],
                            edits_img_at_min=[10, 12, 14, 16] + [np.nan] * 4, edits_txt_at_min=[0] * 4 + [np.nan] * 4))
    arm = pd.DataFrame(dict(base, arm="a1_random_tiered", crossed=[True, True, True, False] * 2, min_edits_cross=[2, 3, 2, 300] * 2,
                            edits_img_at_min=[0, 0, 0, np.nan] * 2, edits_txt_at_min=[2, 3, 2, np.nan] * 2))
    df = pd.concat([ref, arm], ignore_index=True)
    pf = pr.pair_frame(df, "a0_directed", "a1_random_tiered", "min_edits_x")
    assert len(pf) == 3 and set(pf["ref"]) == {10.0, 12.0, 14.0}                # censored 300s never enter
    t = pr.paired_table(df, "a0_directed", ("min_edits_x", "min_edits"), n_boot=50)
    assert int(t[t.metric == "min_edits_x"].iloc[0].n_pairs) == 3 and int(t[t.metric == "min_edits"].iloc[0].n_pairs) == 8


def test_plain_sign_flip_is_primary_and_robust_is_a_diagnostic():
    from mmb.analysis import collect as co
    lp = np.array([[-1.0, -2.0], [-1.55, -1.5], [-1.9, -1.0], [-1.0, -1.2]])      # g = +1, -0.05, -0.9, +0.2
    out = co.evaluate_trace(lp, np.array([0, 0, 1, 1]), anchor_g=1.0)
    assert out["crossed"] and out["first_cross_eval"] == 2                         # sign flip at row 1 (g = -0.05)
    assert out["crossed_robust"] and out["first_cross_eval_robust"] == 3           # diagnostic: |g| >= 0.15 at row 2
    assert int(np.argmax(out["_crossing"])) == 1


def _synthetic_minimal(tmp_path):
    rows = []
    def add(arm, sut, task, ver, n0, img, txt, **ops):
        rows.append(dict(run_dir=f"r{len(rows)}", task=task, sut=sut, arm=arm, rng_seed=0, verified=ver, n_genes_first=n0,
                         n_genes_min=img + txt, img_genes_min=img, txt_genes_min=txt, txt_zero_width=ops.get("zw", 0),
                         txt_homoglyph=ops.get("hg", 0), txt_typo_words=ops.get("ty", 0), txt_synonym_words=ops.get("sy", 0)))
    add("a0_directed", "internvl", "imagenet", True, 25, 0, 1, ty=1)        # one typo
    add("a0_directed", "internvl", "imagenet", True, 90, 6, 0)               # image only, 5+ regions
    add("a0_directed", "internvl", "imagenet", True, 30, 1, 0)               # image only, one region
    add("a0_directed", "internvl", "imagenet", False, 40, 0, 0)              # did not re-verify
    add("a1_random_tiered", "internvl", "imagenet", True, 6, 0, 2, sy=2)
    add("a1_random_tiered", "internvl", "imagenet", True, 8, 1, 1)
    add("a1_random_tiered", "qwen", "pope", True, 5, 0, 1, hg=1)
    add("a1_random_tiered", "qwen", "pope", True, 5, 0, 1, zw=1)
    p = tmp_path / "minimal.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_failure_types_classify_summarise_and_plot(tmp_path):
    from mmb.analysis import failure_types as ft
    df = ft.load_minimal([_synthetic_minimal(tmp_path)])
    assert list(df["ftype"].dropna()) == ["text_typo", "image_many", "image_few", "text_synonym", "joint", "text_homoglyph",
                                          "text_zero_width"]
    s = ft.summary(df)
    top = s[(s.sut == "All") & (s.task == "All") & (s.group == "directed")].iloc[0]
    assert top.n == 4 and top.n_verified == 3 and abs(top.verified_share - 0.75) < 1e-9
    assert top.med_genes_min == 1 and abs(top.share_text_typo - 1 / 3) < 1e-9 and abs(top.one_gene_share - 2 / 3) < 1e-9
    assert ft.figure(df, tmp_path / "f.pdf", {"internvl": "InternVL3.5-4B"}, {"imagenet": "ImageNet"}) and (tmp_path / "f.pdf").exists()


def test_failure_numbers_are_digit_free_macros(tmp_path):
    from mmb.analysis import export as ex
    from mmb.analysis import failure_types as ft
    num = ex.Numbers()
    ex.add_failure_numbers(num, ft.load_minimal([_synthetic_minimal(tmp_path)]))
    names = {r["macro"]: r["value"] for r in num.rows}
    assert names["EFailShareTextTypoDirectedInternvlImagenet"] == "33"
    assert names["EFailVerifiedDirectedAllAll"] == "75" and names["EFailMedGenesMinDirectedAllAll"] == "1"
