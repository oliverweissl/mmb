"""Shared pytest configuration and reusable test fixtures.

Adds the project root to sys.path so that ``mmb.*`` is importable.
Provides synthetic data builders and small fakes (latent image backend, question pool) used by
multiple test modules.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn.functional as F

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
