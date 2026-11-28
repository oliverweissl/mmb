"""Tests for mmb.validity.proxies on hand-computed text cases and synthetic images."""
import numpy as np
import pytest
from PIL import Image, ImageFilter

from mmb.validity import proxies as px

Q = "What is the main subject in this image?"


class TestTextProxies:
    def test_word_edit_distance_hand_computed(self):
        assert px.word_edit_distance(Q, Q) == 0.0
        # one substitution out of 8 words
        assert px.word_edit_distance(Q, "What is the primary subject in this image?") == pytest.approx(1 / 8)
        # one deletion and one insertion on a 3-word text: distance 2 / 3
        assert px.word_edit_distance("a b c", "a c d") == pytest.approx(2 / 3)
        assert px.word_edit_distance("a b c", "") == 1.0
        assert px.word_edit_distance("", "") == 0.0

    def test_case_and_nfkc_are_normalised(self):
        assert px.word_edit_distance("What is the main subject", "what IS the main subject") == 0.0
        assert px.normalise_visible("Is  there a dog?") == "Is there a dog?"

    def test_keyword_similarity(self):
        assert px.keyword_similarity("iguana", "is there a green iguana here") == 1.0
        assert px.keyword_similarity("green iguana", "is there a green iguana here") == 1.0
        assert px.keyword_similarity("iguana", "is there a iguanas here") >= 0.8     # one extra letter
        assert px.keyword_similarity("iguana", "is there a cat here") < 0.5
        assert px.keyword_similarity("", "text") == 0.0

    def test_question_preserved_with_object_word(self):
        orig = "Is there a dog in the image?"
        flag, ks, _ = px.question_preserved(orig, "Does the image show a dog?", "dog")              # pool paraphrase
        assert flag is True and ks == 1.0
        flag, ks, _ = px.question_preserved(orig, "Is there a cat in the image?", "dog")
        assert flag is False and ks < 0.8

    def test_question_preserved_without_object_word_uses_content_words(self):
        flag, _, ck = px.question_preserved(Q, Q)
        assert flag is True and ck == 1.0
        # content words of Q: main, subject, image. Replace two of three -> 1/3 kept
        flag, _, ck = px.question_preserved(Q, "What is the foo bar in this image?")
        assert ck == pytest.approx(1 / 3) and flag is False
        # a question with no content words gives no flag
        flag, _, ck = px.question_preserved("Is it?", "Is it?")
        assert flag is None and np.isnan(ck)

    def test_cosine(self):
        assert px.cosine([1, 0], [1, 0]) == 1.0
        assert px.cosine([1, 0], [0, 1]) == 0.0
        assert px.cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)
        assert np.isnan(px.cosine([0, 0], [1, 0]))


def gradient(size=64):
    x = np.linspace(0, 255, size, dtype=np.uint8)
    arr = np.stack([np.tile(x, (size, 1)), np.tile(x[:, None], (1, size)), np.full((size, size), 128, np.uint8)], -1)
    return Image.fromarray(arr)


def noise(size=64, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8))


class TestSSIM:
    def test_identical_is_one_and_noise_is_low(self):
        g = gradient()
        assert px.ssim(g, g) == pytest.approx(1.0)
        assert px.ssim(g, noise()) < 0.3

    def test_small_perturbation_stays_high_and_blur_is_between(self):
        g = gradient()
        arr = np.asarray(g).astype(int)
        slight = Image.fromarray(np.clip(arr + np.random.default_rng(1).integers(-3, 4, arr.shape), 0, 255).astype(np.uint8))
        assert px.ssim(g, slight) > 0.9
        assert px.ssim(g, noise()) < px.ssim(g, slight)

    def test_different_size_is_resized_and_tiny_images_give_nan(self):
        g = gradient(64)
        assert px.ssim(g, g.resize((32, 32))) > 0.9
        assert np.isnan(px.ssim(Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))))

    def test_match_size(self):
        assert px.match_size(Image.new("L", (10, 20)), Image.new("RGB", (5, 5))).size == (5, 5)


needs_models = pytest.mark.skipif(not px.models_available(), reason="proxy models not in the local cache")


@needs_models
class TestModelProxies:
    @pytest.fixture(scope="class")
    def models(self):
        return px.ProxyModels(local_files_only=True)

    def test_identical_images(self, models):
        g = gradient(96)
        assert models.clip_cos(g, g) == pytest.approx(1.0, abs=1e-4)
        assert models.dino_cos(g, g) == pytest.approx(1.0, abs=1e-4)
        assert models.lpips(g, g) == pytest.approx(0.0, abs=1e-5)

    def test_ordering_slight_edit_beats_unrelated_image(self, models):
        g = gradient(96)
        arr = np.asarray(g).astype(int)
        slight = Image.fromarray(np.clip(arr + np.random.default_rng(2).integers(-4, 5, arr.shape), 0, 255).astype(np.uint8))
        other = noise(96)
        p_slight = px.image_proxies(models, g, slight)
        p_other = px.image_proxies(models, g, other)
        assert p_slight["img_clip_cos"] > p_other["img_clip_cos"]
        assert p_slight["img_dino_cos"] > p_other["img_dino_cos"]
        assert p_slight["img_lpips"] < p_other["img_lpips"]
        assert p_slight["img_ssim"] > p_other["img_ssim"]

    def test_blur_is_a_milder_edit_than_noise_for_lpips(self, models):
        g = gradient(96)
        assert models.lpips(g, g.filter(ImageFilter.GaussianBlur(1))) < models.lpips(g, noise(96))

    def test_sentence_embedding_cosine(self, models):
        assert models.minilm_cos(Q, Q) == pytest.approx(1.0, abs=1e-5)
        near = models.minilm_cos(Q, "What is the primary subject in this picture?")
        far = models.minilm_cos(Q, "Stock markets fell sharply on Tuesday.")
        assert near > far

    def test_text_proxies_bundle(self, models):
        t = px.text_proxies(models, Q, Q)
        assert t["txt_word_edit"] == 0.0 and t["txt_edited"] is False and t["question_preserved"] is True
