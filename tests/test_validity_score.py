"""Tests for mmb.validity.score and mmb.validity.calibrate (models faked, so no downloads)."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mmb.validity import calibrate as cal
from mmb.validity import score
from mmb.validity.proxies import ProxyModels


class FakeModels(ProxyModels):
    """Deterministic stand-ins: cosine of mean pixel values / of text lengths."""

    def clip_cos(self, a, b):
        return 1.0 - abs(float(np.asarray(a).mean()) - float(np.asarray(b).mean())) / 255

    def dino_cos(self, a, b):
        return self.clip_cos(a, b) ** 2

    def lpips(self, a, b):
        return abs(float(np.asarray(a).mean()) - float(np.asarray(b).mean())) / 255

    def minilm_cos(self, a, b):
        return 1.0 if a == b else 0.5


def make_run(root, name, *, seed_level=100, cross_level=110, text="What is the main subject?", crossing=True,
             prompt="What is the main subject?", extra_stats=None, meta=None):
    d = root / name
    d.mkdir(parents=True)
    Image.new("RGB", (32, 32), (seed_level,) * 3).save(d / "origin.png")
    Image.new("RGB", (32, 32), (seed_level,) * 3).save(d / "anchor_recon.png")   # the reference (codec reconstruction)
    stats = {"prompt_template": prompt, "class_a": "junco", "class_b": "sparrow", "seed_idx": 0,
             "model_id": "x/llava", "seed_selection_mode": "roster",
             "run_meta": {"arm": "a0_directed", "exp_id": "E1", "task": "imagenet_pair", "sut": "llava", "rng_seed": 3},
             **(extra_stats or {})}
    (d / "stats.json").write_text(json.dumps(stats))
    if crossing:
        Image.new("RGB", (32, 32), (cross_level,) * 3).save(d / "first_cross.png")
        (d / "first_cross.json").write_text(json.dumps({
            "text": text, "full_prompt": text + " Answer: junco or sparrow.", "g": -0.2, "eval_idx": 17,
            "generation": 0, "genotype": [1, 0], **({"meta": meta} if meta else {})}))
    return d


class TestScore:
    def test_reference_is_saved_recon_not_raw_origin(self, tmp_path):
        from PIL import Image
        from mmb.validity.score import anchor_reference
        Image.new("RGB", (40, 20), (10, 10, 10)).save(tmp_path / "origin.png")       # raw, non-square
        Image.new("RGB", (32, 32), (200, 200, 200)).save(tmp_path / "anchor_recon.png")
        im, kind = anchor_reference(tmp_path)
        assert kind == "saved" and im.size == (32, 32) and im.getpixel((0, 0)) == (200, 200, 200)

    def test_one_row_per_crossing_run_with_hand_checked_values(self, tmp_path):
        make_run(tmp_path, "r_cross", seed_level=100, cross_level=110, text="What is the primary subject?")
        make_run(tmp_path, "r_none", crossing=False)                 # never crossed: no row
        df = score.score_runs(tmp_path, FakeModels())
        assert list(df.columns) == score.COLUMNS and len(df) == 1
        r = df.iloc[0]
        assert r["item_id"] == "r_cross" and r["arm"] == "a0_directed" and r["task"] == "imagenet_pair"
        assert r["first_cross_eval"] == 17 and r["first_cross_gen"] == 0 and r["g_at_cross"] == pytest.approx(-0.2)
        assert r["img_lpips"] == pytest.approx(10 / 255)             # fake: mean pixel difference
        assert r["img_ssim"] < 1.0                                    # real SSIM on the two flat images
        assert r["txt_word_edit"] == pytest.approx(1 / 5)            # main -> primary, 5 words
        assert bool(r["txt_edited"]) is True
        assert r["txt_minilm_cos"] == 0.5

    def test_identical_text_and_image_give_neutral_row(self, tmp_path):
        make_run(tmp_path, "same", seed_level=100, cross_level=100)
        r = score.score_runs(tmp_path, FakeModels()).iloc[0]
        assert r["img_ssim"] == pytest.approx(1.0) and r["img_lpips"] == 0.0
        assert r["txt_word_edit"] == 0.0 and bool(r["txt_edited"]) is False and bool(r["question_preserved"]) is True

    def test_pope_object_word_from_meta(self, tmp_path):
        make_run(tmp_path, "pope", prompt="Is there a dog in the image?", text="Is there a cat in the image?",
                 meta={"object": "dog"})
        r = score.score_runs(tmp_path, FakeModels()).iloc[0]
        assert bool(r["question_preserved"]) is False and r["question_keyword_sim"] < 0.8
        make_run(tmp_path, "pope2", prompt="Is there a dog in the image?", text="Is there a d" + chr(0x043E) + "g in the image?",
                 meta={"object": "dog"})
        df = score.score_runs(tmp_path, FakeModels()).set_index("item_id")
        assert bool(df.loc["pope2", "question_preserved"]) is True

    def test_cli_writes_csv(self, tmp_path, monkeypatch):
        make_run(tmp_path / "runs", "a", cross_level=120)
        monkeypatch.setattr(score, "ProxyModels", lambda local_files_only=False: FakeModels())
        out = tmp_path / "o" / "v.csv"
        assert score.main([str(tmp_path / "runs"), "--out", str(out)]) == 0
        assert list(pd.read_csv(out).columns) == score.COLUMNS


class TestCalibrate:
    def test_auc_hand_computed(self):
        assert cal.auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
        assert cal.auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
        assert cal.auc([1, 1], [True, False]) == 0.5
        # positives {2, 4}, negatives {1, 3}: 3 of 4 pairs ordered correctly
        assert cal.auc([2, 4, 1, 3], [True, True, False, False]) == pytest.approx(0.75)
        assert np.isnan(cal.auc([1, 2], [True, True]))

    def test_youden_threshold(self):
        assert cal.youden_threshold([1, 2, 3, 4], [False, False, True, True]) == 3.0

    def test_calibrate_signs_and_ranking(self):
        rng = np.random.default_rng(0)
        validity = np.linspace(0.05, 0.95, 20)
        joined = pd.DataFrame({
            "validity": validity,
            "img_clip_cos": validity + rng.normal(0, 0.02, 20),          # increases with validity
            "img_lpips": 1 - validity + rng.normal(0, 0.02, 20),         # distance: decreases with validity
            "img_ssim": rng.random(20),                                   # unrelated
        })
        out = cal.calibrate(joined, n_boot=300).set_index("proxy")
        assert out.loc["img_clip_cos", "auc"] > 0.95 and out.loc["img_lpips", "auc"] > 0.95
        assert out.loc["img_lpips", "direction"] == "negated distance"
        assert out.loc["img_clip_cos", "spearman"] > 0.9 and out.loc["img_lpips", "spearman"] > 0.9
        assert out.loc["img_ssim", "auc"] < 0.95
        assert out.loc["img_clip_cos", "n_items"] == 20 and out.loc["img_clip_cos", "n_valid"] == 10
        lo, hi = out.loc["img_clip_cos", ["auc_lo", "auc_hi"]]
        assert lo <= out.loc["img_clip_cos", "auc"] <= hi

    def test_cli_requires_common_item_ids(self, tmp_path):
        pd.DataFrame({"item_id": ["x"], "img_clip_cos": [0.9]}).to_csv(tmp_path / "p.csv", index=False)
        pd.DataFrame({"item_id": ["y"], "validity": [0.5]}).to_csv(tmp_path / "h.csv", index=False)
        with pytest.raises(SystemExit, match="item_id"):
            cal.main(["--proxies", str(tmp_path / "p.csv"), "--human", str(tmp_path / "h.csv"), "--out", str(tmp_path / "o.csv")])
