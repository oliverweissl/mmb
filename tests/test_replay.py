"""mmb.analysis.replay: stored crossings replayed on the canonical path (fake scorer, no GPU)."""
import json

import numpy as np
import pytest
from PIL import Image

from mmb.analysis import replay


class FakeScorer:
    """g = (mean pixel / 255 - 0.5) * 10 + (+2 if 'dog' in the prompt else -2): image and text both move the decision."""

    def __init__(self):
        self.calls = []

    def score_categories_tensor(self, image, prompt, cats):
        self.calls.append((prompt, tuple(cats)))
        g = (np.asarray(image, float).mean() / 255 - 0.5) * 10 + (2.0 if "dog" in prompt else -2.0)
        return np.array([g / 2, -g / 2])

    def generate(self, image, prompt):
        lp = self.score_categories_tensor(image, prompt, ("Yes", "No"))
        return ("Yes, there is." if lp[0] > lp[1] else "No."), None, None


def make_run(root, name, *, img0=140, img1=140, q0="Is there a dog in the image?", q1="Is there a dog in the image?",
             sut="qwen", arm="a0_directed"):
    d = root / name
    d.mkdir(parents=True)
    suffix = " Answer with exactly one of these options: Yes, No."
    Image.new("RGB", (8, 8), (img0,) * 3).save(d / "anchor_recon.png")
    Image.new("RGB", (8, 8), (img1,) * 3).save(d / "min_cross.png")
    (d / "stats.json").write_text(json.dumps({
        "categories": ["Yes", "No"], "target_classes": [0, 1], "prompt_template": q0, "anchor_g": 2.0, "seed_idx": 0,
        "answer_format": " Answer with exactly one of these options: {categories}.",
        "run_meta": {"exp_id": "E", "task": "pope", "sut": sut, "arm": arm, "rng_seed": 0}}))
    (d / "min_cross.json").write_text(json.dumps({"text": q1, "full_prompt": q1 + suffix, "g": -0.5}))
    return d


def test_item_prompts_are_exact(tmp_path):
    make_run(tmp_path, "r", q1="Does the image show a cat?")
    it = replay.items([tmp_path])[0]
    assert it["anchor_prompt"] == "Is there a dog in the image? Answer with exactly one of these options: Yes, No."
    assert it["cross_prompt"] == "Does the image show a cat? Answer with exactly one of these options: Yes, No."
    assert it["arm"] == "a0_directed" and it["sut"] == "qwen" and it["target"] == (0, 1)
    assert replay.items([tmp_path], arms=["a9_deepjanus"]) == [] and replay.items([tmp_path], sut="internvl") == []


@pytest.mark.parametrize("img1, q1, cls", [
    (20, "Is there a dog in the image?", "image"),          # image edit alone flips (text unchanged)
    (140, "Is there a cat in the image?", "text"),          # text edit alone flips
    (60, "Is there a cat in the image?", "either"),         # each alone flips
    (140, "Is there a dog here?", "unverified"),            # the stored crossing does not cross when scored alone
])
def test_swap_classes(tmp_path, img1, q1, cls):
    make_run(tmp_path, "r", img1=img1, q1=q1)
    row = replay.swap_row(replay.items([tmp_path])[0], FakeScorer())
    g0 = (140 / 255 - 0.5) * 10 + 2
    assert row["g_anchor"] == pytest.approx(g0)
    assert row["cls"] == cls and row["verified"] == (cls != "unverified")
    assert row["text_edited"] == (q1 != "Is there a dog in the image?")


def test_joint_class():
    assert replay.classify(False, False) == "joint" and replay.classify(True, False) == "text"


def test_transfer_skips_source_and_flags_flips(tmp_path):
    make_run(tmp_path, "q", sut="qwen", img1=20)
    make_run(tmp_path, "i", sut="internvl", img1=140)
    its = replay.items([tmp_path])
    sc = FakeScorer()
    rows = replay.transfer_rows(its, sc, "qwen")
    assert [r["src_sut"] for r in rows] == ["internvl"] and rows[0]["flipped_on_dst"] is False
    rows = replay.transfer_rows(its, sc, "ministral")
    assert {r["src_sut"]: r["flipped_on_dst"] for r in rows} == {"qwen": True, "internvl": False}
    assert all(len(c[1]) == 2 for c in sc.calls)          # one input per call, the run's candidates


def test_agreement_row(tmp_path):
    make_run(tmp_path, "r", img1=20)
    row = replay.agreement_row(replay.items([tmp_path])[0], FakeScorer())
    assert row["anchor_greedy_label"] == "Yes" and row["anchor_agree"] and row["cross_greedy_label"] == "No"
    assert row["cross_agree"] and row["cross_verified"] and row["cross_g_search"] == -0.5
    assert replay.match_answer("maybe yes, maybe no", ["Yes", "No"]) == "<ambiguous>"


def test_write_csv(tmp_path):
    replay.write_csv([dict(a=1, b=2)], tmp_path / "x" / "o.csv")
    assert (tmp_path / "x" / "o.csv").read_text().splitlines() == ["a,b", "1,2"]
