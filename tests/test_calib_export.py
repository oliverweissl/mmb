"""ECalibTraj* macros and the calibration-trajectory figure (export.add_calib_numbers on calibration.trajectory output)."""
import pandas as pd

from mmb.analysis import calibration as cal
from mmb.analysis import export


def _calib_dir(tmp_path, final=True):
    d = tmp_path / "calib"
    d.mkdir()
    rows = []
    for k, (wi, wt, tk, ik) in enumerate([(1.0, 1.0, 4, 21), (1.0, 2.0, 10, 18), (0.8, 2.0, 15, 16)], 1):
        rows += [dict(step=k, label=f"CALIB_s{k}", exp_id=f"CALIB_s{k}", w_img=wi, w_txt=wt, scope=sc, pairs=24,
                      text_k=tk, text_n=24, text_cr=100 * tk / 24, text_lo=5.0, text_hi=60.0, image_k=ik, image_n=24,
                      image_cr=100 * ik / 24, image_lo=50.0, image_hi=95.0, text_only=1, image_only=ik - tk + 1,
                      p_mcnemar=0.001 if k < 3 else 1.0, balanced=k == 3, text_mag_med=0.1, image_mag_med=0.05,
                      final=final and k == 3) for sc in ("pooled", "pope")]
    pd.DataFrame(rows, columns=cal.TRAJ_COLUMNS).to_csv(d / "trajectory.csv", index=False)
    return d


def test_trajectory_macros_and_figure(tmp_path):
    num = export.Numbers()
    figs = tmp_path / "figs"
    export.add_calib_numbers(num, _calib_dir(tmp_path), figs=figs)
    tex = num.seen
    for name in ("ECalibTrajOneTextCr", "ECalibTrajOneImageCr", "ECalibTrajOneImageLo", "ECalibTrajTwoP",
                 "ECalibTrajFinalP", "ECalibTrajFinalPairs", "ECalibTrajFinalWTxt", "ECalibTrajOneWImg",
                 "ECalibTrajOneTextMag", "ECalibTrajFinalDisc"):
        assert name in tex, name
    assert not any(ch.isdigit() for n in tex if n.startswith("ECalib") for ch in n)
    assert (figs / "calib-trajectory.pdf").stat().st_size > 0


def test_without_final_and_without_file(tmp_path):
    num = export.Numbers()
    export.add_calib_numbers(num, _calib_dir(tmp_path, final=False), figs=None)
    assert "ECalibTrajThreeTextCr" in num.seen and "ECalibTrajFinalP" not in num.seen
    export.add_calib_numbers(export.Numbers(), tmp_path / "missing")          # warns, no crash
