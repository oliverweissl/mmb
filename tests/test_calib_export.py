"""ECalib*/ECalibTraj* macros and the calibration-trajectory figure (export.add_calib_numbers)."""
import pandas as pd

from mmb.analysis import export


def _calib_dir(tmp_path, with_baselines=True):
    d = tmp_path / "calib"
    d.mkdir()
    mods = ["text", "image", "both"] + (["mimicry", "deepjanus"] if with_baselines else [])
    cr = {"text": 18.8, "image": 33.3, "both": 50.0, "mimicry": 41.7, "deepjanus": 35.4}
    pd.DataFrame([dict(level="pooled", mod=m, n=48, CR=cr[m], calls_med=1500.0, edits_med=4.0, lpips_med=0.03,
                       vis_char_med=0.03, q_preserved=80.0, out_margin_med=0.1) for m in mods]).to_csv(d / "summary.csv", index=False)
    pd.DataFrame([dict(scope="pooled", a="both", b=b, pairs=48, share_a=50.0, share_b=cr[b], a_only=8, b_only=0,
                       p_mcnemar=0.008, distinguishable=True) for b in mods if b != "both"]).to_csv(d / "tests.csv", index=False)
    pd.DataFrame([
        dict(step="Zero", label="uncalibrated", final=False, pairs=24, text_cr=8.3, text_lo=2.3, text_hi=25.8, image_cr=87.5,
             image_lo=69.0, image_hi=95.7, text_only=0, image_only=19, p_mcnemar=3.8e-6, image_lpips_med=0.2, text_vis_char_med=0.05),
        dict(step="Final", label="C* final check", final=True, pairs=48, text_cr=18.8, text_lo=10.2, text_hi=31.9, image_cr=33.3,
             image_lo=21.7, image_hi=47.5, text_only=4, image_only=11, p_mcnemar=0.118, image_lpips_med=0.037, text_vis_char_med=0.032),
    ]).to_csv(d / "trajectory.csv", index=False)
    return d


def test_trajectory_macros_and_figure(tmp_path):
    num = export.Numbers()
    figs = tmp_path / "figs"
    export.add_calib_numbers(num, _calib_dir(tmp_path), figs=figs)
    tex = num.seen
    for name in ("ECalibTrajZeroTextCr", "ECalibTrajZeroImageCr", "ECalibTrajZeroImageLo", "ECalibTrajFinalP",
                 "ECalibTrajFinalPairs", "ECalibTrajZeroLpips", "ECalibCrMimicryAll", "ECalibCrDeepJanusAll",
                 "ECalibPBothMimicryAll", "ECalibPBothDeepJanusAll"):
        assert name in tex, name
    assert (figs / "calib-trajectory.pdf").stat().st_size > 0


def test_figure_without_baselines(tmp_path):
    figs = tmp_path / "figs"
    export.add_calib_numbers(export.Numbers(), _calib_dir(tmp_path, with_baselines=False), figs=figs)
    assert (figs / "calib-trajectory.pdf").exists()
