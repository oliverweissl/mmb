import pandas as pd

from mmb.analysis.export import complete_blocks


def _rows(arm, seed, cells, suts=("qwen", "internvl"), tasks=("imagenet", "pope")):
    return [dict(arm=arm, rng_seed=seed, task=t, sut=s, cell_id=c) for t in tasks for s in suts for c in cells]


def test_partial_block_is_held_back_and_complete_kept():
    df = pd.DataFrame(_rows("a0_directed", 0, "abc") + _rows("a0_directed", 1, "abc")
                      + _rows("a3_ga_single", 0, "ab")              # partial seed-0 block
                      + _rows("x1_unguided", 0, "abc"))             # complete seed-0 block, no seed 1
    out = complete_blocks(df, "a0_directed", ("qwen", "internvl"))
    assert "a3_ga_single" not in set(out.arm)
    assert len(out[out.arm == "x1_unguided"]) == 12 and len(out[out.arm == "a0_directed"]) == 24
