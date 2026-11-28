import numpy as np, pandas as pd
from mmb.analysis import validity_cost as vc


def _data():
    rng = np.random.default_rng(0)
    rows, v = [], []
    for arm in ("a0_directed", "a9_deepjanus"):
        for i in range(12):
            cr = i % 3 != 0
            rows.append(dict(arm=arm, task="imagenet", sut="qwen", crossed=cr, run_dir=f"{arm}{i}", min_within_budget=cr,
                             min_text_d=rng.uniform(0, 0.3) if cr else np.nan))
            if cr:
                v.append(dict(arm=arm, task="imagenet", sut="qwen", run_dir=f"{arm}{i}", first_cross_eval=10 + i,
                              img_lpips=rng.uniform(0.1, 0.5), txt_minilm_cos=rng.uniform(0.7, 1.0), txt_edited=True))
    return pd.DataFrame(v), pd.DataFrame(rows)


def test_table_and_macros():
    from mmb.analysis.export import Numbers, TASK_NAMES
    v, d = _data()
    t = vc.validity_cost(v, d, ["qwen"])
    assert set(t["task"]) == {"imagenet", "All"}
    r = t[(t.task == "All") & (t.arm == "a0_directed")].iloc[0]
    assert r.n_runs == 12 and r.n_cross == 8 and 0 <= r.cons_share <= r.both_share <= 1
    n = Numbers(); vc.add_numbers(n, t)
    assert any(x["macro"].startswith("EValidConsShare") for x in n.rows)
    assert "MMBA" in vc.table_tex(t, TASK_NAMES) and r.obj_n == 8 and 0 <= r.obj_txt_share <= 1
    assert len(vc.sensitivity(v, d, ["qwen"])) == 2 * 16


def test_empty_arm_is_nan_not_crash():
    v, d = _data()
    v = v[v.arm == "a0_directed"]
    t = vc.validity_cost(v, d, ["qwen"])
    assert np.isnan(t[(t.task == "All") & (t.arm == "a9_deepjanus")].iloc[0].med_lpips)
