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


def test_best_indices_ties_and_degenerate_groups():
    assert tf.best_indices([0.5, 0.7, 0.7], "crossed") == {1, 2}              # ties: all tied values
    assert tf.best_indices([10, 4, 4], "floor") == {1, 2}                      # lower is better
    assert tf.best_indices([0.5, 0.5], "crossed") == set()                     # nothing to distinguish
    assert tf.best_indices([np.nan, 0.3], "crossed") == set()                  # a single value has no best
    assert tf.best_indices([0.501, 0.502], "crossed", ["50.1", "50.2"]) == {1}
    assert tf.best_indices([0.5004, 0.5001, 0.3], "crossed", ["50.0", "50.0", "30.0"]) == {0, 1}   # ties by display


def test_paired_bold_only_when_significant_and_favourable():
    assert tf.bold_if_better("+0.10", 0.10, 0.01, "crossed") == r"\textbf{+0.10}"
    assert tf.bold_if_better("+0.10", 0.10, 0.20, "crossed") == "+0.10"        # not significant after Holm
    assert tf.bold_if_better("-0.30", -0.30, 0.01, "first_cross_eval") == r"\textbf{-0.30}"
    assert tf.bold_if_better("+0.30", 0.30, 0.01, "first_cross_eval") == "+0.30"   # significantly worse


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
    assert r"\textbf{30 (83.3)" in rows[0] and r"\textbf" not in rows[1].split("&")[2]
    assert r"\textbf{100 [80--120]}" in rows[0] and r"\textbf{0.01}" in rows[0]
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
