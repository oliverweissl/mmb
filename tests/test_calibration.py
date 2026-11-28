"""mmb.analysis.calibration: step configs, trajectory (shares, exact McNemar) and the figure."""
import pandas as pd
import pytest
import yaml

from mmb.analysis import calibration as cal


def runs(exp, w_img, w_txt, text_cross, image_cross, task="pope"):
    rows = []
    for i, (tc, ic) in enumerate(zip(text_cross, image_cross)):
        for arm, c in ((cal.TEXT_ARM, tc), (cal.IMAGE_ARM, ic)):
            rows.append(dict(exp_id=exp, arm=arm, task=task, sut="qwen", cell_id=f"c{i}", rng_seed=0, crossed=bool(c),
                             magnitude=0.1 * (i + 1) if c else float("nan"), w_img=w_img, w_txt=w_txt))
    return pd.DataFrame(rows)


def test_step_rows_hand_computed():
    # 10 pairs: text crosses 2, image crosses 9; discordant: image only 7, text only 0 -> p = 2 * 0.5^7
    d = runs("CALIB_s0", 1.0, 1.0, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    r = cal.step_rows(d, 1, "s0")[0]
    assert r["scope"] == "pooled" and r["pairs"] == 10 and (r["text_k"], r["image_k"]) == (2, 9)
    assert (r["text_only"], r["image_only"]) == (0, 7) and r["p_mcnemar"] == pytest.approx(2 * 0.5 ** 7)
    assert r["balanced"] is False and r["text_cr"] == 20 and r["image_cr"] == 90
    assert r["text_lo"] < 20 < r["text_hi"] and (r["w_img"], r["w_txt"]) == (1.0, 1.0)
    assert r["text_mag_med"] == pytest.approx(0.15)


def test_trajectory_order_scopes_and_final(tmp_path):
    s0 = runs("CALIB_s0", 1.0, 1.0, [1, 0, 0, 0], [1, 1, 1, 1])
    s1 = pd.concat([runs("CALIB_s1", 1.0, 2.0, [1, 1, 1, 0], [1, 1, 1, 1]),
                    runs("CALIB_s1", 1.0, 2.0, [1, 0], [1, 1], task="imagenet")])
    t = cal.trajectory([("s0", s0), ("s1", s1)], final="CALIB_s1")
    assert list(t.columns) == cal.TRAJ_COLUMNS
    assert list(t["step"]) == [1, 1, 2, 2, 2] and list(t["scope"]) == ["pooled", "pope", "pooled", "imagenet", "pope"]
    p = t[t.scope == "pooled"].set_index("step")
    assert p.loc[2, "pairs"] == 6 and p.loc[2, "w_txt"] == 2.0 and bool(p.loc[2, "final"]) and not bool(p.loc[1, "final"])
    assert bool(p.loc[2, "balanced"])                         # 4 vs 6 crossed, 2 discordant: p = 0.5
    out = cal.plot_trajectory(t, tmp_path / "f" / "calib-trajectory.pdf")
    assert out.read_bytes().startswith(b"%PDF")


def test_unmatched_runs_are_not_pairs():
    d = runs("E", 1.0, 1.0, [1, 1], [1, 0])
    d = d[~((d.arm == cal.IMAGE_ARM) & (d.cell_id == "c1"))]
    assert cal.step_rows(d, 1, "E")[0]["pairs"] == 1


def test_step_config_sets_weights_cells_and_names():
    base = {"seed": 0, "image": {"latent": {"d": 16, "w_img": 1.0}}, "text": {"composite": {"profile": "question_pool_only"}},
            "seeds": {"slot_items": {"items": [{"prompt": "old"}]}},
            "tags": {"task": "pope", "sut": "qwen", "arm": "a6_text_only", "exp_id": "MAIN"}}
    c = cal.step_config(base, step="s2", w_img=0.5, w_txt=3.0, cells=[{"prompt": "Is there a dog in the image?"}])
    assert c["image"]["latent"] == {"d": 16, "w_img": 0.5}
    assert c["text"]["composite"] == {"profile": "question_pool_only", "overrides": {"question_pool": {"w_txt": 3.0}}}
    assert c["seeds"]["slot_items"]["items"][0]["prompt"] == "Is there a dog in the image?"
    assert c["tags"]["exp_id"] == "CALIB_s2" and c["save_dir"] == "runs/CALIB/s2/pope/qwen/a6_text_only/s0"
    assert base["image"]["latent"]["w_img"] == 1.0                    # base untouched


def test_cli_configs_and_analyze(tmp_path):
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "pope_calib.yaml").write_text(yaml.safe_dump({"seeds": {"slot_items": {"items": [{"prompt": "q"}]}}}))
    base = tmp_path / "a6.yaml"
    base.write_text(yaml.safe_dump({"seed": 0, "tags": {"task": "pope", "sut": "qwen", "arm": "a6_text_only"}}))
    w = cal.make_configs([base], step="s1", w_img=1.0, w_txt=2.0, out=tmp_path / "cfg", cells_dir=cells)
    c = yaml.safe_load(w[0].read_text())
    assert c["text"]["composite"]["overrides"]["question_pool"]["w_txt"] == 2.0 and c["seeds"]["slot_items"]["items"] == [{"prompt": "q"}]
    csv = tmp_path / "s1.csv"
    runs("CALIB_s1", 1.0, 2.0, [1, 0, 1], [1, 1, 1]).to_csv(csv, index=False)
    assert cal.main(["analyze", str(csv), "--out", str(tmp_path / "res")]) == 0
    t = pd.read_csv(tmp_path / "res" / "trajectory.csv")
    assert t.iloc[0]["label"] == "CALIB_s1" and (tmp_path / "res" / "calib-trajectory.pdf").is_file()
