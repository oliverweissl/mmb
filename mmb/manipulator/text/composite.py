"""Composite text manipulator: one genotype block per operator, applied in CANONICAL_ORDER.

Run time (2026-11-28 trim): the only text operator is ``question_pool`` (two float genes (t, u) choosing a whole
validated question from configs/text/question_pool.json); profile ``question_pool_only``. The machinery stays
generic: an operator declares ``continuous = True`` for real-valued genes in [0, 1] (bounds xl/xu, uniform random
genes in [0, 1), scale(genes, lam)); any other operator has integer genes 0 .. bound-1 (0 = no edit), as the test
stubs use. Operators not in CANONICAL_ORDER run after the listed ones, in the given order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import spacy
from numpy.typing import NDArray

from .config import TextConfig
from .operators.base import OperatorContext, TextOperator
from .profiles import (
    build_operators_from_specs,
    load_profile_library,
    resolve_profile,
)
from .types import TokenSequence, tokenize


CANONICAL_ORDER: tuple[str, ...] = (
    "question_pool",       # continuous text block (t, u): replaces the whole question
)


# ---------------------------------------------------------------------------
# Composite context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeManipulationContext:
    """Per-seed prepared state for the composite text manipulator."""

    original_tokens: TokenSequence
    op_contexts: Mapping[str, OperatorContext]
    op_gene_dims: tuple[int, ...]
    op_gene_bounds: tuple[NDArray[np.int64], ...]
    op_order: tuple[str, ...]
    op_continuous: tuple[bool, ...] = ()     # per operator: float genes in [0, 1] (question_pool) vs integer genes

    def _cont(self, i: int) -> bool:
        return bool(self.op_continuous[i]) if i < len(self.op_continuous) else False

    @property
    def is_continuous(self) -> bool:
        """True when every gene is real-valued (all active operators continuous; False for an empty genotype)."""
        act = [self._cont(i) for i, d in enumerate(self.op_gene_dims) if d > 0]
        return bool(act) and all(act)

    @property
    def xl(self) -> NDArray[np.float64]:
        """Inclusive lower bounds per gene (0 everywhere)."""
        return np.zeros(self.genotype_dim)

    @property
    def xu(self) -> NDArray[np.float64]:
        """Inclusive upper bounds per gene: 1.0 for continuous genes (u is upper-open: u = 1 renders as the last band
        member), bound - 1 for integer genes."""
        parts = [np.ones(d) if self._cont(i) else np.asarray(b, dtype=np.float64) - 1.0
                 for i, (d, b) in enumerate(zip(self.op_gene_dims, self.op_gene_bounds)) if d > 0]
        return np.concatenate(parts) if parts else np.empty(0)

    @property
    def genotype_dim(self) -> int:
        return int(sum(self.op_gene_dims))

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        non_empty = [b for b in self.op_gene_bounds if b.size > 0]
        if not non_empty:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(non_empty)

    def zero_genotype(self) -> NDArray:
        """The unedited seed (all genes 0); float64 when any operator is continuous."""
        cont = any(self._cont(i) for i, d in enumerate(self.op_gene_dims) if d > 0)
        return np.zeros(self.genotype_dim, dtype=np.float64 if cont else np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray:
        """Uniform genes: continuous genes in [0, 1) (upper-open), integer genes in [0, bound)."""
        if self.genotype_dim == 0:
            return np.empty(0, dtype=np.int64)
        cont = any(self._cont(i) for i, d in enumerate(self.op_gene_dims) if d > 0)
        out = []
        for i, (d, b) in enumerate(zip(self.op_gene_dims, self.op_gene_bounds)):
            if d == 0:
                continue
            out.append(rng.random(d) if self._cont(i) else np.array([rng.integers(0, int(x)) for x in b]))
        return np.concatenate(out).astype(np.float64 if cont else np.int64)


# ---------------------------------------------------------------------------
# Composite manipulator
# ---------------------------------------------------------------------------


class CompositeTextManipulator:
    """Text manipulator over a list of operators (run time: [QuestionPoolOperator]).

    Construction: ``CompositeTextManipulator(nlp, operators)`` or ``CompositeTextManipulator.from_config(text_config)``
    (profile library -> operators).
    """

    def __init__(
        self,
        nlp: spacy.language.Language,
        operators: Sequence[TextOperator],
    ) -> None:
        self._nlp = nlp
        self._operators = sorted(
            operators,
            key=lambda op: (
                CANONICAL_ORDER.index(op.name)
                if op.name in CANONICAL_ORDER
                else len(CANONICAL_ORDER)
            ),
        )

    @classmethod
    def from_config(
        cls,
        text_config: TextConfig,
        device: str | None = None,
        redis_url: str | None = None,
    ) -> "CompositeTextManipulator":
        """Build a composite from a TextConfig.

        :param device: Default device for model-bearing operators (Synonym
            MLM). Per-operator ``extras["device"]`` overrides. Pass the
            top-level experiment device so the MLM matches the SUT.
        :param redis_url: Optional Redis URL for the Synonym candidate
            cache. Typically ``cfg.sut.redis_url`` so it shares the SUT's
            cache server.
        """
        nlp = spacy.load(
            text_config.spacy_model,
            disable=["ner", "parser", "lemmatizer"],
        )
        library = load_profile_library(text_config.composite.profile_library)
        specs = resolve_profile(
            library,
            profile_name=text_config.composite.profile,
            operators=(
                list(text_config.composite.operators)
                if text_config.composite.operators
                else None
            ),
            overrides=text_config.composite.overrides or None,
        )
        operators = build_operators_from_specs(
            specs,
            content_pos=text_config.content_pos_tags,
            device=device,
            redis_url=redis_url,
        )
        return cls(nlp=nlp, operators=operators)

    @property
    def operators(self) -> tuple[TextOperator, ...]:
        return tuple(self._operators)

    @property
    def operator_names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self._operators)

    @property
    def is_continuous(self) -> bool:
        """True when every operator takes real-valued genes (e.g. profile question_pool_only)."""
        return bool(self._operators) and all(getattr(op, "continuous", False) for op in self._operators)

    @property
    def w_txt(self) -> float:
        """Text weight applied inside the continuous operators (the optimizer never sees it)."""
        ws = [float(op.w_txt) for op in self._operators if hasattr(op, "w_txt")]
        if not ws:
            raise AttributeError("no operator with a text weight (w_txt)")
        return ws[0]

    @w_txt.setter
    def w_txt(self, value: float) -> None:
        ops = [op for op in self._operators if hasattr(op, "w_txt")]
        if not ops:
            raise AttributeError("no operator with a text weight (w_txt)")
        for op in ops:
            op.w_txt = float(value)

    def scale(self, genes: NDArray, lam: float, ctx: CompositeManipulationContext | None = None) -> NDArray[np.float64]:
        """Text block of scale(g, lam): each continuous operator scales its own block (question_pool: t -> lam t,
        u unchanged). Needs ctx unless there is a single operator. Integer operators cannot be scaled."""
        genes = np.asarray(genes, dtype=np.float64)
        if ctx is None:
            if len(self._operators) != 1:
                raise ValueError("scale() needs ctx when the composite has several operators")
            dims = [genes.size]
        else:
            dims = list(ctx.op_gene_dims)
            if genes.size != sum(dims):
                raise ValueError(f"genotype length {genes.size} != expected {sum(dims)}")
        out, off = [], 0
        for op, d in zip(self._operators, dims):
            blk = genes[off:off + d]; off += d
            if d == 0:
                continue
            if not getattr(op, "continuous", False):
                raise TypeError(f"operator {op.name!r} has integer genes: scale() is defined for continuous genes only")
            out.append(op.scale(blk, lam))
        return np.concatenate(out) if out else np.empty(0)

    # ------------------------------------------------------------------
    # Two-phase API
    # ------------------------------------------------------------------

    def prepare(
        self,
        text: str,
        exclude_words: frozenset[str] | None = None,
    ) -> CompositeManipulationContext:
        tokens = tokenize(self._nlp, text)
        if exclude_words:   # a protected phrase ("dining table") protects each of its words (fix 2026-11-27)
            exclude_words = frozenset(exclude_words) | frozenset(
                p for w in exclude_words for p in str(w).lower().split())
        op_contexts: dict[str, OperatorContext] = {}
        op_gene_dims: list[int] = []
        op_gene_bounds: list[NDArray[np.int64]] = []
        op_order: list[str] = []
        op_continuous: list[bool] = []
        for op in self._operators:
            ctx = op.prepare(tokens, exclude_words=exclude_words)
            op_contexts[op.name] = ctx
            op_gene_dims.append(int(op.gene_dim(ctx)))
            op_gene_bounds.append(op.gene_bounds(ctx))
            op_order.append(op.name)
            op_continuous.append(bool(getattr(op, "continuous", False)))

        return CompositeManipulationContext(
            original_tokens=tokens,
            op_contexts=op_contexts,
            op_gene_dims=tuple(op_gene_dims),
            op_gene_bounds=tuple(op_gene_bounds),
            op_order=tuple(op_order),
            op_continuous=tuple(op_continuous),
        )

    def apply(
        self,
        ctx: CompositeManipulationContext,
        genotype: NDArray[np.int64],
    ) -> str:
        if len(genotype) != ctx.genotype_dim:
            raise ValueError(
                f"Composite genotype length {len(genotype)} != expected {ctx.genotype_dim}"
            )
        current = ctx.original_tokens
        offset = 0
        for op in self._operators:
            d = int(op.gene_dim(ctx.op_contexts[op.name]))
            if d == 0:
                continue
            block = genotype[offset:offset + d]
            if not getattr(op, "continuous", False):    # question_pool takes float genes (t, u)
                block = block.astype(np.int64)
            current = op.apply(ctx.op_contexts[op.name], block, current)
            offset += d
        return current.text


__all__ = [
    "CANONICAL_ORDER",
    "CompositeManipulationContext",
    "CompositeTextManipulator",
]
