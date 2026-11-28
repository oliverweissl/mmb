"""Text manipulator (run time: the continuous question_pool operator).

The text block of the genotype is two floats (t, u): t in [0, 1] = magnitude, u in [0, 1) = choice within a distance
band of a precomputed, validated pool of whole edited questions (configs/text/question_pool.json, built offline by
scripts/build_question_pool.py from semantic word substitutions, the paraphrase bank and T5 paraphrases).

Usage::

    from mmb.manipulator.text import CompositeTextManipulator, TextConfig
    cm = CompositeTextManipulator.from_config(TextConfig())          # profile question_pool_only
    ctx = cm.prepare("Is there a dog in the image?")
    out = cm.apply(ctx, ctx.zero_genotype())                         # the original question
"""

from .composite import (
    CANONICAL_ORDER,
    CompositeManipulationContext,
    CompositeTextManipulator,
)
from .config import TextCompositeConfig, TextConfig
from .profiles import (
    OperatorSpec,
    TextProfile,
    build_operators_from_specs,
    load_profile_library,
    resolve_profile,
)
from .types import CONTENT_POS_TAGS, TokenSequence, tokenize

__all__ = [
    "CANONICAL_ORDER",
    "CONTENT_POS_TAGS",
    "CompositeManipulationContext",
    "CompositeTextManipulator",
    "OperatorSpec",
    "TextCompositeConfig",
    "TextConfig",
    "TextProfile",
    "TokenSequence",
    "build_operators_from_specs",
    "load_profile_library",
    "resolve_profile",
    "tokenize",
]
