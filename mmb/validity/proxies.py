"""Image and text similarity proxies between a crossing input and its seed.

Image (all between the crossing image and the unmodified seed image):
  clip   -- CLIP ViT-B/32 image-image cosine
  dino   -- DINOv2-small CLS-embedding cosine
  lpips  -- LPIPS (AlexNet), a distance: 0 = identical
  ssim   -- structural similarity on RGB, 1 = identical
Text (edited prompt against the original question):
  minilm -- all-MiniLM-L6-v2 sentence-embedding cosine
  word_edit -- word-level Levenshtein distance divided by the longer word count, 0 = identical
  question_preserved -- heuristic flag that the object / class word of the question is still readable
All models run on CPU and are loaded lazily; pass your own :class:`ProxyModels` subclass in tests.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
from PIL import Image

CLIP_ID = "openai/clip-vit-base-patch32"
DINO_ID = "facebook/dinov2-small"
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"

#: A key word counts as still present when some token of the edited prompt matches it with a
#: difflib ratio >= this value (inflections such as "dogs" still match "dog"). Heuristic, not calibrated.
KEYWORD_MATCH_RATIO = 0.8
#: Without an object word, the question counts as preserved when at least this fraction of the
#: original question's content words is still present (same matching rule).
CONTENT_KEPT_MIN = 0.8

_STOP = frozenset("""a an the of in on at to is are was were be been this that these those it its what which who
whom whose how why where when do does did there here with without for from by as and or but not no yes
answer exactly one following options option listed list please""".split())



# ---------------------------------------------------------------------------
# Text proxies (pure, no models)
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", _normalise(text))


def word_edit_distance(a: str, b: str) -> float:
    """Word-level Levenshtein distance / max(len(words(a)), len(words(b))); 0 for two empty texts."""
    wa, wb = words(a), words(b)
    n = max(len(wa), len(wb))
    if n == 0:
        return 0.0
    prev = list(range(len(wb) + 1))
    for i, x in enumerate(wa, 1):
        cur = [i]
        for j, y in enumerate(wb, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / n


def keyword_similarity(keyword: str, text: str) -> float:
    """Best difflib ratio between the (possibly multi-word) keyword and the tokens / token n-grams of text."""
    kw = " ".join(words(keyword))
    toks = words(text)
    if not kw or not toks:
        return 0.0
    n_kw = len(kw.split())
    best = 0.0
    for n in {n_kw, 1}:
        for i in range(len(toks) - n + 1):
            cand = " ".join(toks[i:i + n])
            best = max(best, difflib.SequenceMatcher(None, kw, cand).ratio())
    return best


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in _STOP and len(w) > 1]


def content_kept(original: str, edited: str) -> float:
    """Fraction of the original question's content words still present in the edited prompt (NaN if none)."""
    cw = content_words(original)
    if not cw:
        return float("nan")
    return float(np.mean([keyword_similarity(w, edited) >= KEYWORD_MATCH_RATIO for w in cw]))


def question_preserved(original: str, edited: str, keyword: str | None = None) -> tuple[bool | None, float, float]:
    """(flag, keyword_similarity, content_kept).

    With a *keyword* (POPE: the object noun) the flag is whether the keyword is still present;
    otherwise it is whether enough content words of the original question survive. The flag is a
    heuristic about readability of the question's subject, not a judgment of meaning.
    """
    ck = content_kept(original, edited)
    if keyword:
        ks = keyword_similarity(keyword, edited)
        return bool(ks >= KEYWORD_MATCH_RATIO), ks, ck
    if np.isnan(ck):
        return None, float("nan"), ck
    return bool(ck >= CONTENT_KEPT_MIN), float("nan"), ck


def cosine(u, v) -> float:
    u, v = np.asarray(u, float).ravel(), np.asarray(v, float).ravel()
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d > 0 else float("nan")


# ---------------------------------------------------------------------------
# Image proxies
# ---------------------------------------------------------------------------


def match_size(img: Image.Image, ref: Image.Image) -> Image.Image:
    """RGB copy of img resized to ref's size (bicubic) if the sizes differ."""
    img = img.convert("RGB")
    return img if img.size == ref.size else img.resize(ref.size, Image.BICUBIC)


def ssim(a: Image.Image, b: Image.Image) -> float:
    """Structural similarity of two RGB images (b resized to a's size if needed)."""
    from skimage.metrics import structural_similarity

    a = a.convert("RGB")
    b = match_size(b, a)
    xa, xb = np.asarray(a), np.asarray(b)
    win = min(7, min(xa.shape[0], xa.shape[1]))
    win -= (win % 2 == 0)
    if win < 3:
        return float("nan")
    return float(structural_similarity(xa, xb, channel_axis=2, data_range=255, win_size=win))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ProxyModels:
    """Lazily loaded CPU models. Subclass and override the four methods to fake them in tests."""

    def __init__(self, device: str = "cpu", local_files_only: bool = False) -> None:
        self.device = device
        self.local = local_files_only
        self._clip = self._dino = self._lpips = self._minilm = None

    # -- loading -----------------------------------------------------------
    def _load_clip(self):
        if self._clip is None:
            from transformers import CLIPImageProcessor, CLIPModel

            m = CLIPModel.from_pretrained(CLIP_ID, local_files_only=self.local).to(self.device).eval()
            p = CLIPImageProcessor.from_pretrained(CLIP_ID, local_files_only=self.local)
            self._clip = (m, p)
        return self._clip

    def _load_dino(self):
        if self._dino is None:
            from transformers import AutoImageProcessor, AutoModel

            m = AutoModel.from_pretrained(DINO_ID, local_files_only=self.local).to(self.device).eval()
            p = AutoImageProcessor.from_pretrained(DINO_ID, local_files_only=self.local)
            self._dino = (m, p)
        return self._dino

    def _load_lpips(self):
        if self._lpips is None:
            import lpips

            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        return self._lpips

    def _load_minilm(self):
        if self._minilm is None:
            from sentence_transformers import SentenceTransformer

            self._minilm = SentenceTransformer(MINILM_ID, device=self.device, local_files_only=self.local)
        return self._minilm

    # -- proxies -----------------------------------------------------------
    def clip_cos(self, a: Image.Image, b: Image.Image) -> float:
        import torch

        m, p = self._load_clip()
        with torch.no_grad():
            px = p(images=[a.convert("RGB"), match_size(b, a)], return_tensors="pt")["pixel_values"].to(self.device)
            f = m.get_image_features(pixel_values=px)
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return cosine(f[0].cpu().numpy(), f[1].cpu().numpy())

    def dino_cos(self, a: Image.Image, b: Image.Image) -> float:
        import torch

        m, p = self._load_dino()
        with torch.no_grad():
            px = p(images=[a.convert("RGB"), match_size(b, a)], return_tensors="pt")["pixel_values"].to(self.device)
            h = m(pixel_values=px).last_hidden_state[:, 0]          # CLS token
        return cosine(h[0].cpu().numpy(), h[1].cpu().numpy())

    def lpips(self, a: Image.Image, b: Image.Image) -> float:
        import torch

        net = self._load_lpips()

        def t(img):
            x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0)
            return x.permute(2, 0, 1)[None].to(self.device)

        with torch.no_grad():
            return float(net(t(a), t(match_size(b, a))).item())

    def minilm_cos(self, a: str, b: str) -> float:
        m = self._load_minilm()
        e = m.encode([a, b], normalize_embeddings=True, convert_to_numpy=True)
        return cosine(e[0], e[1])


def image_proxies(models: ProxyModels, seed_img: Image.Image, cross_img: Image.Image) -> dict[str, float]:
    return {
        "img_clip_cos": models.clip_cos(seed_img, cross_img),
        "img_dino_cos": models.dino_cos(seed_img, cross_img),
        "img_lpips": models.lpips(seed_img, cross_img),
        "img_ssim": ssim(seed_img, cross_img),
    }


# ---------------------------------------------------------------------------
# Visible text view (pre-registered in METHODOLOGY.md, 2026-11-22 07:50 UTC)
# ---------------------------------------------------------------------------


def normalise_visible(text: str) -> str:
    """NFKC and collapsed spaces (question-pool questions are plain words; kept for callers that compare strings)."""
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def text_proxies(models: ProxyModels, original: str, edited: str, keyword: str | None = None) -> dict:
    flag, ks, ck = question_preserved(original, edited, keyword)
    return {
        "txt_minilm_cos": models.minilm_cos(original, edited),
        "txt_word_edit": word_edit_distance(original, edited),
        "question_preserved": flag,
        "question_keyword_sim": ks,
        "question_content_kept": ck,
        "txt_edited": _normalise(original) != _normalise(edited),
    }


def models_available() -> bool:
    """True if all four models load from the local cache (used to skip model-backed tests offline)."""
    try:
        m = ProxyModels(local_files_only=True)
        m._load_clip(); m._load_dino(); m._load_lpips(); m._load_minilm()
        return True
    except Exception:  # noqa: BLE001
        return False
