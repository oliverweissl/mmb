"""Shared pytest configuration and reusable test fixtures.

Adds the project root to sys.path so that ``mmb.*`` is importable.
Provides synthetic data builders + composite-shaped fakes used by
multiple test modules.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmb.manipulator.text.composite import CompositeManipulationContext
from mmb.manipulator.text.operators.base import OperatorContext
from mmb.manipulator.text.types import TokenSequence


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def make_tokens(*items: tuple[str, str, str]) -> TokenSequence:
    """Build a TokenSequence from (word, pos, whitespace) triples."""
    return TokenSequence(
        tokens=tuple(w for w, _, _ in items),
        pos_tags=tuple(p for _, p, _ in items),
        whitespace=tuple(s for _, _, s in items),
    )


# ---------------------------------------------------------------------------
# Fake sub-manipulators (composite-shaped)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeSynonymContext(OperatorContext):
    candidates: tuple[tuple[str, ...], ...]


class _FakeSynonymOperator:
    """Stub Synonym operator with hard-coded positions/candidates.

    Mirrors the real :class:`SynonymOperator` interface (``name``,
    ``prepare``, ``gene_dim``, ``gene_bounds``, ``apply``) without
    needing ModernBERT or spaCy.
    """

    name = "synonym"

    def __init__(
        self,
        positions: tuple[int, ...] = (1, 3),
        candidates: tuple[tuple[str, ...], ...] = (("fast", "rapid"), ("wolf", "dog", "cat")),
    ) -> None:
        self._positions = np.array(positions, dtype=np.intp)
        self._candidates = candidates

    def prepare(self, tokens, exclude_words=None):
        return _FakeSynonymContext(
            positions=self._positions,
            candidates=self._candidates,
        )

    def gene_dim(self, ctx):
        return ctx.n_positions

    def gene_bounds(self, ctx):
        return np.array([len(c) + 1 for c in ctx.candidates], dtype=np.int64)

    def apply(self, ctx, genes, current):
        active = np.nonzero(genes)[0]
        if len(active) == 0:
            return current
        positions = ctx.positions[active]
        words = tuple(ctx.candidates[int(i)][int(genes[i]) - 1] for i in active)
        return current.replace(positions, words)


class FakeCompositeTextManipulator:
    """Minimal composite-shaped text manipulator for VLMManipulator tests.

    Stub Synonym operator over the canonical "The quick brown fox" sentence.
    Bypasses spaCy / ModernBERT — enough surface for VLMManipulator
    integration tests.
    """

    def __init__(self) -> None:
        self._op = _FakeSynonymOperator()

    def prepare(
        self,
        text: str,
        exclude_words: frozenset[str] | None = None,
    ) -> CompositeManipulationContext:
        tokens = make_tokens(
            ("The", "DET", " "),
            ("quick", "ADJ", " "),
            ("brown", "ADJ", " "),
            ("fox", "NOUN", ""),
        )
        op_ctx = self._op.prepare(tokens)
        return CompositeManipulationContext(
            original_tokens=tokens,
            op_contexts={"synonym": op_ctx},
            op_gene_dims=(self._op.gene_dim(op_ctx),),
            op_gene_bounds=(self._op.gene_bounds(op_ctx),),
            op_order=("synonym",),
        )

    def apply(self, ctx: CompositeManipulationContext, genotype) -> str:
        op_ctx = ctx.op_contexts["synonym"]
        out = self._op.apply(op_ctx, genotype, ctx.original_tokens)
        return out.text


# ---------------------------------------------------------------------------
# Continuous manipulators for tests (2026-11-28): fake VQ model + latent backend,
# question-pool text over a temporary pool file
# ---------------------------------------------------------------------------


class FakeVQ(torch.nn.Module):
    """VQModel stand-in: encode = 8x average pool -> (n, 4, 32, 32); decode = upsample + tanh."""

    def encode(self, x):
        p = F.avg_pool2d(x, 8)
        return torch.cat([p, p.mean(1, keepdim=True)], 1), None, None

    def decode(self, z):
        return torch.tanh(F.interpolate(z[:, :3], scale_factor=8, mode="nearest"))


def make_latent_manipulator(d: int = 8, m_max: float = 0.2):
    from mmb.manipulator.image.latent import LatentImageConfig, LatentImageManipulator
    return LatentImageManipulator(FakeVQ(), LatentImageConfig(d=d, m_max=m_max), device="cpu",
                                  decode_batch_size=4)


POOL_DISTANCES = (0.0, 0.012, 0.03, 0.045, 0.051, 0.08, 0.12, 0.2)


def write_question_pool(path: Path, questions) -> str:
    """A small question pool (configs/text/question_pool.json format) for the given questions."""
    import json
    from mmb.manipulator.text.operators.question_pool import load_pool
    path.write_text(json.dumps({"meta": {"d_max": 0.25, "n_bands": 10}, "questions": {
        q: {"pool": [{"text": f"{q.rstrip('?.')} variant {i}?", "dist": float(x)}
                     for i, x in enumerate(POOL_DISTANCES[1:], 1)]} for q in questions}}))
    load_pool.cache_clear()
    return str(path)


def make_question_pool_text(pool: str):
    import spacy
    from mmb.manipulator.text.composite import CompositeTextManipulator
    from mmb.manipulator.text.operators.question_pool import QuestionPoolOperator
    return CompositeTextManipulator(spacy.blank("en"), [QuestionPoolOperator(pool=pool)])


def make_vlm_manipulator(pool: str, d: int = 8):
    """VLMManipulator over the latent backend (fake VQ) and the question-pool text block."""
    from mmb.manipulator.vlm_manipulator import VLMManipulator
    return VLMManipulator(make_latent_manipulator(d=d), make_question_pool_text(pool))
