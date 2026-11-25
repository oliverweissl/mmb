import pandas as pd

import pytest

from mmb.analysis.export import UnbalancedExport, check_balance, complete_blocks, method_marks, pending_confirmation
from mmb.analysis import rq_tables as rq


def _rows(arm, seed, cells, suts=("qwen", "internvl"), tasks=("imagenet", "pope")):
    return [dict(arm=arm, rng_seed=seed, task=t, sut=s, cell_id=c) for t in tasks for s in suts for c in cells]


def test_partial_block_is_held_back_and_complete_kept():
    df = pd.DataFrame(_rows("a0_directed", 0, "abc") + _rows("a0_directed", 1, "abc")
                      + _rows("a5_mimicry_objective", 0, "ab")              # partial seed-0 block
                      + _rows("x1_unguided", 0, "abc"))             # complete seed-0 block, no seed 1
    out = complete_blocks(df, "a0_directed", ("qwen", "internvl"))
    assert "a5_mimicry_objective" not in set(out.arm)
    assert len(out[out.arm == "x1_unguided"]) == 12 and len(out[out.arm == "a0_directed"]) == 24


def test_unconfirmed_runs_are_pending_and_marked():
    rows = [dict(arm="a0_directed", rng_seed=s, run_dir=f"a0/{s}/{c}", min_confirmed=True) for s in (0, 1) for c in "ab"]
    rows += [dict(arm="a5_mimicry_objective", rng_seed=0, run_dir=f"a5/0/{c}", min_confirmed=False) for c in "ab"]
    rows += [dict(arm="a1_random_tiered", rng_seed=0, run_dir=f"a1/0/{c}", min_confirmed=True) for c in "ab"]
    df = pd.DataFrame(rows)
    pend = pending_confirmation(df)
    assert pend == {"a5/0/a", "a5/0/b"}
    marks, suffix = method_marks(df, "a0_directed", pend)
    assert "a0_directed" not in marks
    assert marks["a5_mimicry_objective"] == r"$^{\S\ddagger}$" and marks["a1_random_tiered"] == r"$^{\S}$"
    assert suffix == {"a5_mimicry_objective": " (seed 0)", "a1_random_tiered": " (seed 0)"}
    notes = rq.note_rows(marks, 5)
    assert len(notes) == 2 and all(n.startswith(r"\multicolumn{5}{l}") for n in notes)
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

