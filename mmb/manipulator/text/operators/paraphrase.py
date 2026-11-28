"""T-B paraphrase gene (2026-11-28): one gene per question; gene k >= 1 replaces the whole question by the
ladder[k-1]-th nearest accepted paraphrase of the precomputed, hand-reviewable bank (scripts/build_paraphrase_bank.py
-> configs/text/paraphrase_bank.json; filters: protected words kept, dictionary words, NLI mutual entailment >= 0.8,
CoLA; ranked by MiniLM sentence distance). gene 0 keeps the question.

Applied last in the composite: an active paraphrase replaces the question including any word-level substitution, so
word genes are inert while it is active (phase-2 minimisation then drops them).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..types import TokenSequence
from .base import OperatorContext

logger = logging.getLogger(__name__)
BANK = "configs/text/paraphrase_bank.json"


@dataclass(frozen=True)
class ParaphraseContext(OperatorContext):
    candidates: tuple[str, ...] = ()
    distances: tuple[float, ...] = ()


@lru_cache(maxsize=4)
def _load_bank(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"paraphrase bank {path} missing (scripts/build_paraphrase_bank.py)")
    return json.loads(p.read_text())


@lru_cache(maxsize=1)
def _blank():
    import spacy
    return spacy.blank("en")


def _tokens(text: str) -> TokenSequence:
    doc = _blank()(text)
    return TokenSequence(tokens=tuple(t.text for t in doc), pos_tags=tuple("X" for _ in doc),
                         whitespace=tuple(t.whitespace_ for t in doc))


class ParaphraseOperator:
    name = "paraphrase"

    def __init__(self, severity: float = 1.0, ladder=(1, 2, 4, 8, 16), bank: str = BANK) -> None:
        self._severity = float(severity)
        self._ladder = tuple(int(r) for r in ladder)
        if not self._ladder or any(r < 1 for r in self._ladder) or list(self._ladder) != sorted(set(self._ladder)):
            raise ValueError(f"ladder must be strictly increasing ranks >= 1, got {ladder}")
        self._bank = str(bank)

    @property
    def severity(self) -> float:
        return self._severity

    @property
    def k_max(self) -> int:
        return len(self._ladder)

    @property
    def ladder(self) -> tuple[int, ...]:
        return self._ladder

    def prepare(self, tokens: TokenSequence, exclude_words: frozenset[str] | None = None) -> ParaphraseContext:
        q = "".join(t + w for t, w in zip(tokens.tokens, tokens.whitespace)).strip()
        entry = _load_bank(self._bank).get(q)
        if entry is None:
            logger.warning("no paraphrases in %s for %r: paraphrase gene off", self._bank, q)
            return ParaphraseContext(positions=np.empty(0, dtype=np.intp))
        lst = entry["paraphrases"]
        lv = [lst[r - 1] for r in self._ladder if r <= len(lst)]
        if not lv:
            return ParaphraseContext(positions=np.empty(0, dtype=np.intp))
        return ParaphraseContext(positions=np.zeros(1, dtype=np.intp), candidates=tuple(x["text"] for x in lv),
                                 distances=tuple(float(x["dist"]) for x in lv))

    def gene_dim(self, ctx: ParaphraseContext) -> int:
        return 1 if ctx.candidates else 0

    def gene_bounds(self, ctx: ParaphraseContext) -> NDArray[np.int64]:
        return np.array([len(ctx.candidates) + 1], dtype=np.int64) if ctx.candidates else np.empty(0, dtype=np.int64)

    def apply(self, ctx: ParaphraseContext, genes: NDArray[np.int64], current: TokenSequence) -> TokenSequence:
        if not ctx.candidates:
            return current
        if len(genes) != 1:
            raise ValueError(f"Paraphrase: gene length {len(genes)} != 1")
        k = int(genes[0])
        return current if k == 0 else _tokens(ctx.candidates[k - 1])
