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
    row = [l for l in tex.splitlines() if l.startswith("Directed")][0]
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


def test_arm_main_table_effects_bold_and_headers():
    from mmb.analysis import export as ex
    summary = pd.DataFrame(dict(arm=["a0_directed", "a1_random_tiered"], n_runs=[36, 36], share_crossed=[0.9, 0.5],
                                median_first_cross_eval=[120.0, 600.0]))
    cost = pd.DataFrame(dict(arm=["a0_directed", "a1_random_tiered"], med_min_edits=[20.0, 80.0]))
    paired = pd.DataFrame(dict(arm=["a1_random_tiered"] * 3, metric=["crossed", "first_cross_eval", "min_edits"],
                               diff=[-0.40, 5.0, 60.0], dz=[-1.3, 0.7, 0.2], p_holm=[0.004, 0.2, 0.03]))
    tex = ex.arm_main_tex(summary, cost, paired)
    rows = [l for l in tex.splitlines() if l.startswith(("Directed", "Random"))]
    assert "Crossing rate" in tex and "Model calls to first crossing" in tex and "Edits at crossing" in tex
    assert r"\uparrow" in tex and r"\downarrow" in tex
    assert r"\textbf{90}" in rows[0] and r"\textbf{120}" in rows[0] and r"\textbf{20}" in rows[0]
    assert r"-40 \effL{}$^{\dagger}$" in rows[1]                                  # crossing rate: pp, large, significant
    assert r"+0.70 \effM{} &" in rows[1].replace("\\\\", "&") or r"+0.70 \effM{}" in rows[1]
    assert r"+0.20 \effS{}$^{\dagger}$" in rows[1]
    assert "star" not in tex and "$^{*" not in tex and "p_" not in tex


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
