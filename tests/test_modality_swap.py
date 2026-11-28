"""Modality-swap analysis (mmb.analysis.modality_swap) on synthetic replay rows."""
import pandas as pd

from mmb.analysis import export, modality_swap as msw
from mmb.analysis import replay as script


def test_classify():
    assert script.classify(False, True) == "image" and script.classify(True, False) == "text"
    assert script.classify(True, True) == "either" and script.classify(False, False) == "joint"


def _rows():
    r = []
    for arm, cls in [("a0_directed", "image")] * 2 + [("a0_directed", "joint"), ("a0_directed", "text"),
                     ("a6_text_only", "text"), ("a7_image_only", "image")]:
        r.append(dict(run_dir=f"r{len(r)}", view="min", arm=arm, task="pope" if len(r) % 2 else "imagenet", sut="qwen",
                      verified=True, cls=cls, rel_text_only=0.5, rel_image_only=-0.2, g_text_only=1.0, g_image_only=-0.3))
    r.append(dict(run_dir="u", view="min", arm="a0_directed", task="pope", sut="qwen", verified=False, cls="unverified",
                  rel_text_only=1.0, rel_image_only=1.0, g_text_only=2.0, g_image_only=2.0))
    return pd.DataFrame(r)


def test_shares_exclude_unverified():
    s = msw.shares(_rows()).set_index("arm")
    assert s.at["a0_directed", "n"] == 5 and s.at["a0_directed", "n_verified"] == 4
    assert s.at["a0_directed", "image"] == 50.0 and s.at["a0_directed", "joint"] == 25.0
    assert abs(s.loc["a0_directed", list(msw.CLASSES)].sum() - 100) < 1e-9


def test_macros_and_plot(tmp_path):
    num = export.Numbers()
    msw.add_numbers(num, _rows(), export.camel, "min")
    for m in ("EModSwapAZeroImageAll", "EModSwapAZeroJointAll", "EModSwapAZeroVerifiedAll", "EModSwapASixTextAll",
              "EModSwapAZeroRelTextAll"):
        assert m in num.seen, m
    out = msw.plot(_rows(), tmp_path / "ms.pdf", lambda a: a)
    assert out.stat().st_size > 0
