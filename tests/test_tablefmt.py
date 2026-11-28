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
    for key in ("crossed", "band", "obj_share", "budget_share"):
        assert tf.DIRECTION[key] == tf.UP
    for key in ("first_cross_eval", "floor", "magnitude", "mag_image", "mag_text", "bp_gap", "lpips", "txt_dist", "wall"):
        assert tf.DIRECTION[key] == tf.DOWN


def test_generation_zero_and_budget_50_are_neutral():
    assert "gen0" not in tf.DIRECTION and tf.head("Gen.0", "gen0") == "Gen.0"
    curve = pd.DataFrame([dict(arm=a, budget=b, share=s, boot_lo=s, boot_hi=s)
                          for a, s in (("a0_directed", 0.9), ("a9_deepjanus", 0.4)) for b in (50, 100)])
    tex = st.budgets_tex(curve, [50, 100])
    header = [l for l in tex.splitlines() if l.startswith("Arm")][0]
    assert header.count("uparrow") == 1 and header.split("&")[1].count("uparrow") == 0   # arrow only on the larger budget
    row = [l for l in tex.splitlines() if l.startswith("MMBA")][0]
    assert row.split("&")[1].count("textbf") == 0 and row.split("&")[2].count("textbf") == 1


def test_best_indices_ties_and_degenerate_groups():
    assert tf.best_indices([0.5, 0.7, 0.7], "crossed") == {1, 2}              # ties: all tied values
    assert tf.best_indices([10, 4, 4], "floor") == {1, 2}                      # lower is better
    assert tf.best_indices([0.5, 0.5], "crossed") == set()                     # nothing to distinguish
    assert tf.best_indices([np.nan, 0.3], "crossed") == set()                  # a single value has no best
    assert tf.best_indices([0.501, 0.502], "crossed", ["50.1", "50.2"]) == {1}
    assert tf.best_indices([0.5004, 0.5001, 0.3], "crossed", ["50.0", "50.0", "30.0"]) == {0, 1}   # ties by display


def test_summary_table_has_arrows_and_one_best_per_column():
    s = pd.DataFrame(dict(arm=["a0_directed", "a9_deepjanus"], n_runs=[36, 36], n_crossed=[30, 20],
                          median_magnitude=[0.05, 0.2],
                          share_crossed=[30 / 36, 20 / 36], wilson_lo=[.7, .4], wilson_hi=[.9, .7],
                          share_gen0=[.1, .1], gen0_lo=[0, 0], gen0_hi=[.2, .2], share_band=[.5, .2],
                          share_label_flip=[.9, .6], median_first_cross_eval=[100., 400.], mfc_lo=[80., 300.],
                          mfc_hi=[120., 500.], fc_q1=[1, 1], fc_q3=[1, 1], median_floor_nats=[0.01, 0.2],
                          mfl_lo=[0, 0], mfl_hi=[1, 1]))
    tex = st.summary_tex(s)
    assert r"\uparrow" in tex and r"\downarrow" in tex
    rows = [l for l in tex.splitlines() if l.startswith(("MMBA", "DeepJanus"))]
    assert r"\textbf{83.3}" in rows[0] and r"\textbf" not in rows[1].split("&")[2]
    assert r"\textbf{100}" in rows[0] and r"\textbf{0.01}" in rows[0] and r"\textbf{0.05}" in rows[0]
    assert "[" not in tex                                                         # no intervals in the tables
    assert r"\textbf{36}" not in tex                                            # counts are never bold
    gen0 = [r.split("&")[3] for r in rows]
    assert all(r"\textbf" not in g for g in gen0)                               # all tied: no best


def test_validity_table_bold_within_task_block():
    t = pd.DataFrame(dict(task=["imagenet", "imagenet", "pope", "pope"], arm=["a0_directed", "a9_deepjanus"] * 2,
                          n_cross=[5, 5, 5, 5], n_runs=[10] * 4, med_lpips=[0.05, 0.10, 0.20, 0.10], med_txt=[0.2] * 4,
                          img_share=[1, .8, .6, .9], txt_share=[0] * 4, both_share=[0] * 4, cons_share=[0, 0, 0, 0],
                          cons_lo=[0] * 4, cons_hi=[0] * 4, rho_eval_lpips=[0] * 4, rho_eval_txt=[0] * 4,
                          med_word_edit=[.2] * 4))
    tex = vc.table_tex(t, {"imagenet": "ImageNet", "pope": "POPE"}, objective=False)
    lines = [l for l in tex.splitlines() if "MMBA" in l or "DeepJanus" in l]
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
        for a, cr, lp, np_ in (("a0_directed", 90.0, 0.05, 20), ("a9_deepjanus", 50.0, 0.08, 2)):
            rows.append(dict(task=t, sut="qwen", arm=a, n_runs=10, CR=cr, S=0.9, SCR=cr * 0.9, lpips=lp, lpips_bp=lp if np_ >= 3 else float("nan"), n_partner=np_,
                             d_edit=0.0 if a.startswith("a0") else 10.0, d_sem=0.01, first_cross_eval=50.0 if a.startswith("a0") else 90.0,
                             min_eval_idx=400.0, wall=170.0, magnitude=0.04 if a.startswith("a0") else 0.09))
    m = pd.DataFrame(rows)
    t1 = rq.rq1_tex(m)
    assert "Qwen3.5-4B" in t1 and "ImageNet" in t1 and "POPE" in t1 and r"\uparrow" in t1 and r"\downarrow" in t1 and "SCR" in t1
    assert r"\rotatebox[origin=c]{90}" in t1 and r"\multirow{4}" in t1
    d_row = [l for l in t1.splitlines() if r"\sysname" in l][0]
    assert any("DeepJanus" in l for l in t1.splitlines())
    assert r"\textbf{90}" in d_row and r"\textbf{0.05}" in d_row and r"\textbf{0.040}" in d_row and "LPIPS" in t1 and "bp" not in t1
    t2 = rq.rq2_tex(m)
    assert "Runtime (s)" in t2 and "Calls to first" in t2 and "Calls to smallest" in t2 and r"\textbf{50}" in t2
    ab = rq.ablation(pd.DataFrame(dict(arm=["a0_directed"] * 4, rng_seed=0, crossed=[True, True, True, False], first_cross_eval=[10, 20, 30, 40],
                                      wall_s=100.0, min_eval_idx=[50, 60, 70, float("nan")], run_dir=list("abcd"),
                                      magnitude=[0.1, 0.2, 0.3, float("nan")], mag_image=[0.1, 0.2, 0.3, float("nan")],
                                      min_text_changed=False)),
                     pd.DataFrame(dict(view=["min"] * 3 + ["first"] * 3, run_dir=list("abcabc"), txt_word_edit=0.0, txt_minilm_cos=1.0,
                                       has_partner=True, pair_lpips=0.02, txt_edited=False, img_lpips=0.1)))
    tab = rq.ablation_tex(ab)
    assert "Full" in tab and "w/o Phase 2" not in tab and "Variant" in tab and "Magnitude" in tab
    full = ab.iloc[0]
    assert full["magnitude"] == 0.2 and abs(full["S"] - 0.9) < 1e-9              # stealth: image changed, LPIPS 0.1


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


def test_smallest_crossing_metrics_use_pairs_where_both_arms_have_one_in_budget():
    from mmb.analysis import paired as pr
    n = 8
    base = dict(cell_id=[f"c{i}" for i in range(n)], rng_seed=[0] * n, task=["imagenet"] * n, sut=["x"] * n)
    ref = pd.DataFrame(dict(base, arm="a0_directed", crossed=[True] * 4 + [False] * 4,
                            min_within_budget=[True] * 4 + [False] * 4,
                            magnitude=[0.1, 0.12, 0.14, 0.16] + [np.nan] * 4))
    arm = pd.DataFrame(dict(base, arm="a9_deepjanus", crossed=[True, True, True, False] * 2,
                            min_within_budget=[True, True, True, False] * 2, magnitude=[0.2, 0.3, 0.2, np.nan] * 2))
    df = pd.concat([ref, arm], ignore_index=True)
    pf = pr.pair_frame(df, "a0_directed", "a9_deepjanus", "magnitude")
    assert len(pf) == 3 and np.allclose(sorted(10 ** pf["ref"]), [0.1, 0.12, 0.14])     # log10 scale; never-crossed never enter
    t = pr.paired_table(df, "a0_directed", ("magnitude", "crossed"), n_boot=50)
    assert int(t[t.metric == "magnitude"].iloc[0].n_pairs) == 3 and int(t[t.metric == "crossed"].iloc[0].n_pairs) == 8


def test_plain_sign_flip_is_the_crossing():
    from mmb.analysis import collect as co
    lp = np.array([[-1.0, -2.0], [-1.55, -1.5], [-1.9, -1.0], [-1.0, -1.2]])      # g = +1, -0.05, -0.9, +0.2
    out = co.evaluate_trace(lp, np.array([0, 0, 1, 1]), anchor_g=1.0)
    assert out["crossed"] and out["first_cross_eval"] == 2                         # sign flip at row 1 (g = -0.05)
    assert "crossed_robust" not in out
    assert int(np.argmax(out["_crossing"])) == 1


def test_ablation_effect_markers():
    """Variant cells carry the magnitude bullet and the dagger of the paired comparison with the full method."""
    n = 30
    base = dict(task="imagenet", sut="qwen", rng_seed=0, wall_s=100.0)
    rows = []
    for i in range(n):
        rows.append(dict(base, arm="a0_directed", seed_idx=i, run_dir=f"a0/{i}", crossed=True, first_cross_eval=50 + i))
        rows.append(dict(base, arm="a6_text_only", seed_idx=i, run_dir=f"a6/{i}", crossed=i < 3, first_cross_eval=1500 if i >= 3 else 900))
    df = pd.DataFrame(rows)
    vdf = pd.DataFrame(dict(run_dir=[f"a0/{i}" for i in range(n)], view="min", img_lpips=0.2, txt_word_edit=0.0))
    from mmb.analysis import rq_tables as rq
    ab = rq.ablation(df, vdf)
    e = rq.ablation_effects(ab)
    tab = rq.ablation_tex(ab, e)
    row = next(l for l in tab.splitlines() if l.startswith("Text only"))
    assert r"\effL{}$^{\dagger}$" in row and r"\textbf{100}" in tab
