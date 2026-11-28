import pandas as pd

import pytest

from mmb.analysis.export import UnbalancedExport, check_balance, complete_blocks, method_marks, pending_confirmation
from mmb.analysis import rq_tables as rq


def _rows(arm, seed, cells, suts=("qwen", "internvl"), tasks=("imagenet", "pope")):
    return [dict(arm=arm, rng_seed=seed, task=t, sut=s, cell_id=c) for t in tasks for s in suts for c in cells]


def test_partial_block_is_held_back_and_complete_kept():
    df = pd.DataFrame(_rows("a0_directed", 0, "abc") + _rows("a0_directed", 1, "abc")
                      + _rows("a5_mimicry_objective", 0, "ab")              # partial seed-0 block
                      + _rows("x1_mmba_unguided", 0, "abc"))        # complete seed-0 block, no seed 1
    out = complete_blocks(df, "a0_directed", ("qwen", "internvl"))
    assert "a5_mimicry_objective" not in set(out.arm)
    assert len(out[out.arm == "x1_mmba_unguided"]) == 12 and len(out[out.arm == "a0_directed"]) == 24


def test_unconfirmed_runs_are_pending_and_marked():
    rows = [dict(arm="a0_directed", rng_seed=s, run_dir=f"a0/{s}/{c}", min_found=True, min_confirmed=True) for s in (0, 1) for c in "ab"]
    rows += [dict(arm="a5_mimicry_objective", rng_seed=0, run_dir=f"a5/0/{c}", min_found=True, min_confirmed=False) for c in "ab"]
    rows += [dict(arm="a9_deepjanus", rng_seed=0, run_dir=f"a9/0/{c}", min_found=True, min_confirmed=True) for c in "ab"]
    rows += [dict(arm="a9_deepjanus", rng_seed=1, run_dir=f"a9/1/{c}", min_found=False, min_confirmed=None) for c in "ab"]
    df = pd.DataFrame(rows)
    pend = pending_confirmation(df)
    assert pend == {"a5/0/a", "a5/0/b"}
    marks, suffix = method_marks(df, "a0_directed", pend)
    assert "a0_directed" not in marks
    assert marks == {"a5_mimicry_objective": r"$^{\ddagger}$"} and suffix == {}     # no seed marks
    notes = rq.note_rows(marks, 5)
    assert len(notes) == 1 and notes[0].startswith(r"\multicolumn{5}{l}")
    assert rq.note_rows({}, 5) == []


def test_balance_guard_fails_loudly_and_passes_balanced():
    grid = ("qwen", "internvl")
    ok = pd.DataFrame(_rows("a0_directed", 0, "abc", grid) + _rows("a5_mimicry_objective", 0, "abc", grid))
    assert check_balance(ok, grid) == {("a0_directed", 0): 3, ("a5_mimicry_objective", 0): 3}
    # tasks have their own seed inputs (different ids, same count): balanced
    two = pd.DataFrame(_rows("a0_directed", 0, "abc", grid, ("imagenet",)) + _rows("a0_directed", 0, "xyz", grid, ("pope",)))
    assert check_balance(two, grid) == {("a0_directed", 0): 3}
    # same count but different seed inputs on two models of one task: unbalanced
    odd = pd.DataFrame(_rows("a0_directed", 0, "abc", ("qwen",)) + _rows("a0_directed", 0, "abd", ("internvl",)))
    with pytest.raises(UnbalancedExport):
        check_balance(odd, grid)
    # a method block present for one model only
    bad = pd.DataFrame(_rows("a0_directed", 0, "abc", grid) + _rows("a5_mimicry_objective", 0, "abc", ("qwen",)))
    with pytest.raises(UnbalancedExport):
        check_balance(bad, grid)
    # the reference itself missing a seed on one model x task cell
    bad2 = pd.DataFrame(_rows("a0_directed", 0, "abc", grid) + _rows("a0_directed", 1, "abc", grid, ("imagenet",)))
    with pytest.raises(UnbalancedExport):
        check_balance(bad2, grid)


def test_complete_blocks_output_is_balanced():
    grid = ("qwen", "internvl")
    df = pd.DataFrame(_rows("a0_directed", 0, "abc", grid) + _rows("a5_mimicry_objective", 0, "abc", ("qwen",)))
    check_balance(complete_blocks(df, "a0_directed", grid), grid)


def test_ablation_table_text_only_lpips_dash_and_no_baseline_rows():
    import numpy as np
    def runs(lp):
        return pd.DataFrame({"lpips": lp, "crossed": [1.0] * len(lp)})
    a = pd.DataFrame([
        dict(label="Full \\sysname", pre="a0", n_runs=3, CR=97.0, SCR=81.0, lpips=0.20, magnitude=0.11, first_cross_eval=56.0, wall=170.0, _runs=runs([0.2, 0.1, 0.3])),
        dict(label="Text only", pre="a6", n_runs=3, CR=7.0, SCR=5.0, lpips=0.0, magnitude=0.3, first_cross_eval=1500.0, wall=220.0, _runs=runs([0.0, 0.0, np.nan])),
        dict(label="Image only", pre="a7", n_runs=3, CR=44.0, SCR=42.0, lpips=0.03, magnitude=0.05, first_cross_eval=1500.0, wall=170.0, _runs=runs([0.03, 0.02, 0.04])),
    ])
    t = rq.ablation_tex(a, pd.DataFrame())
    assert "Text only & 7 & 5 & -- &" in t and r"\textbf{0.03}" in t and r"\textbf{0.00}" not in t
    assert r"\textbf{0.050}" in t
    assert not [v for v in rq.VARIANTS if v[1] in ("a5", "a9", "x2")] and not [v for v in rq.VARIANTS if v[0] == "w/o Phase 2"]
