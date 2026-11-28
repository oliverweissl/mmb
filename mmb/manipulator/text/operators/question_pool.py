"""Continuous text block (2026-11-28): ONE text manipulator for the whole question.

Per seed question, a precomputed pool of WHOLE edited questions (scripts/build_question_pool.py ->
configs/text/question_pool.json, hand-reviewable):

* semantic word substitutions of the original (SemanticSubstitutionOperator candidates) combined over 1..3 sites,
* the reviewed paraphrase bank (configs/text/paraphrase_bank.json),
* extra paraphrases from the T5 paraphraser (humarin/chatgpt_paraphraser_on_T5_base),

each validated AS A WHOLE question (NLI mutual entailment >= 0.8, CoLA rule, protected words kept, dictionary words,
meaning rules of the T-B review), de-duplicated and ordered by MiniLM sentence distance d to the original.

Genes (float): t in [0, 1] = magnitude, u in [0, 1) = choice within a distance band. The manipulator applies the
text weight w_txt (the optimizer never sees it):

    target = w_txt * t * d_max                       (d_max: global constant, pool meta / config)
    bands  = [k h, (k+1) h), h = d_max / n_bands     (fixed, global)
    band   = the non-empty band nearest the target   (the original question sits at d = 0, band 0; ties -> lower)
    entry  = band members (sorted by d) [floor(u * n_band)]
    t = 0  -> the original question.

For a fixed u the selected distance is non-decreasing in t (bands are ordered and disjoint); the rendered text is
piecewise constant in t. scale(genes, lam) multiplies t by lam and keeps u.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..types import TokenSequence
from .base import OperatorContext

logger = logging.getLogger(__name__)
POOL = "configs/text/question_pool.json"
DEFAULT_N_BANDS = 20


@dataclass(frozen=True)
class QuestionPoolContext(OperatorContext):
    """texts[0] is the original question (d = 0); texts[1:] the pool, sorted by distance."""

    texts: tuple[str, ...] = ()
    distances: tuple[float, ...] = ()


@lru_cache(maxsize=4)
def load_pool(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"question pool {path} missing (scripts/build_question_pool.py)")
    return json.loads(p.read_text())


def select_index(distances, t: float, u: float, *, w_txt: float, d_max: float, n_bands: int = DEFAULT_N_BANDS) -> int:
    """Pure mapping (t, u) -> index into `distances` (sorted ascending, distances[0] = 0 = the original).

    Index 0 for t <= 0. Otherwise the target w_txt * t * d_max picks the nearest non-empty distance band of width
    d_max / n_bands (ties: the lower band), and u in [0, 1) picks a member of that band (members in distance order).
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.size == 0:
        raise ValueError("empty pool (needs at least the original)")
    t = float(t)
    if not math.isfinite(t) or t <= 0.0 or d.size == 1:
        return 0
    if d_max <= 0 or n_bands < 1:
        raise ValueError(f"d_max must be > 0 and n_bands >= 1, got {d_max}, {n_bands}")
    h = float(d_max) / int(n_bands)
    target = float(w_txt) * t * float(d_max)
    band = np.floor(d / h + 1e-12).astype(np.int64)
    occupied = np.unique(band)                      # ascending
    # distance from the target to each occupied band interval [k h, (k+1) h)
    lo, hi = occupied * h, (occupied + 1) * h
    gap = np.where(target < lo, lo - target, np.where(target >= hi, target - hi, 0.0))
    k = int(occupied[int(np.argmin(gap))])          # argmin returns the first (lowest) band on ties
    members = np.flatnonzero(band == k)             # already in distance order
    u = min(max(float(u) if math.isfinite(float(u)) else 0.0, 0.0), np.nextafter(1.0, 0.0))
    return int(members[min(int(u * members.size), members.size - 1)])


def scale(genes: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    """Text block of scale(g, lam): t -> lam * t (clipped to [0, 1]), u unchanged."""
    g = np.array(genes, dtype=np.float64, copy=True)
    if g.size:
        g[0] = min(max(lam * g[0], 0.0), 1.0)
    return g


@lru_cache(maxsize=1)
def _blank():
    import spacy
    return spacy.blank("en")


def _tokens(text: str) -> TokenSequence:
    doc = _blank()(text)
    return TokenSequence(tokens=tuple(t.text for t in doc), pos_tags=tuple("X" for _ in doc),
                         whitespace=tuple(t.whitespace_ for t in doc))


class QuestionPoolOperator:
    """Two float genes (t, u) per question; see the module docstring. ``continuous = True``: genes are not cast to int."""

    name = "question_pool"
    continuous = True

    def __init__(self, severity: float = 1.0, pool: str = POOL, w_txt: float = 1.0, d_max: float | None = None,
                 n_bands: int | None = None) -> None:
        self._severity = float(severity)
        self._pool = str(pool)
        self.w_txt = float(w_txt)
        self._d_max = None if d_max is None else float(d_max)
        self._n_bands = None if n_bands is None else int(n_bands)

    @property
    def severity(self) -> float:
        return self._severity

    @staticmethod
    def scale(genes, lam: float) -> NDArray[np.float64]:
        return scale(genes, lam)

    @property
    def d_max(self) -> float:
        return self._d_max if self._d_max is not None else float(load_pool(self._pool)["meta"]["d_max"])

    @property
    def n_bands(self) -> int:
        if self._n_bands is not None:
            return self._n_bands
        return int(load_pool(self._pool)["meta"].get("n_bands", DEFAULT_N_BANDS))

    def prepare(self, tokens: TokenSequence, exclude_words: frozenset[str] | None = None) -> QuestionPoolContext:
        q = tokens.text.strip()
        entry = load_pool(self._pool)["questions"].get(q)
        if entry is None or not entry["pool"]:
            logger.warning("no pool in %s for %r: text block off", self._pool, q)
            return QuestionPoolContext(positions=np.empty(0, dtype=np.intp), texts=(q,), distances=(0.0,))
        return QuestionPoolContext(positions=np.zeros(1, dtype=np.intp),
                                   texts=(q,) + tuple(x["text"] for x in entry["pool"]),
                                   distances=(0.0,) + tuple(float(x["dist"]) for x in entry["pool"]))

    def gene_dim(self, ctx: QuestionPoolContext) -> int:
        return 2 if len(ctx.texts) > 1 else 0

    def gene_bounds(self, ctx: QuestionPoolContext) -> NDArray[np.float64]:
        """Upper bounds of (t, u); lower bounds are 0 (u is half-open: u = 1 is treated as the last member)."""
        return np.array([1.0, 1.0]) if len(ctx.texts) > 1 else np.empty(0)

    def select(self, ctx: QuestionPoolContext, genes) -> int:
        if len(genes) != 2:
            raise ValueError(f"QuestionPool: gene length {len(genes)} != 2")
        return select_index(ctx.distances, float(genes[0]), float(genes[1]), w_txt=self.w_txt, d_max=self.d_max,
                            n_bands=self.n_bands)

    def render(self, ctx: QuestionPoolContext, genes) -> tuple[str, float]:
        """(question, its distance d) for genes (t, u)."""
        if len(ctx.texts) <= 1:
            return ctx.texts[0], 0.0
        i = self.select(ctx, genes)
        return ctx.texts[i], ctx.distances[i]

    def apply(self, ctx: QuestionPoolContext, genes, current: TokenSequence) -> TokenSequence:
        if len(ctx.texts) <= 1:
            return current
        i = self.select(ctx, genes)
        return current if i == 0 else _tokens(ctx.texts[i])
