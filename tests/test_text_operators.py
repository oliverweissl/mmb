"""Tests for the text side after the trim (2026-11-28): the run-time text manipulator is the continuous
question_pool operator alone; the synonym MLM base stays for scripts/build_question_pool.py (via semantic.py).

The MLM-based SynonymOperator is exercised via a stub MLM injected through ``_set_resources``; the real
ModernBERT is not loaded in pytest. The question_pool mapping itself is tested in tests/test_question_pool.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_tokens

from mmb.manipulator.text.composite import (
    CANONICAL_ORDER,
    CompositeTextManipulator,
)
from mmb.manipulator.text.operators.base import OperatorContext, severity_to_k_max
from mmb.manipulator.text.operators.question_pool import QuestionPoolOperator, load_pool
from mmb.manipulator.text.operators.synonym import (
    NEGATION_PREFIX_RE,
    SynonymOperator,
)
from mmb.manipulator.text.profiles import (
    OperatorSpec,
    TextProfile,
    build_operators_from_specs,
    load_profile_library,
    resolve_profile,
)
from mmb.manipulator.text.types import TokenSequence


def _tokens_quick_brown_fox() -> TokenSequence:
    return make_tokens(
        ("The", "DET", " "),
        ("quick", "ADJ", " "),
        ("brown", "ADJ", " "),
        ("fox", "NOUN", ""),
    )


# ===========================================================================
# severity_to_k_max (used by the synonym base of the pool builder)
# ===========================================================================


class TestSeverityToKmax:
    def test_zero_returns_zero(self):
        assert severity_to_k_max(0.0) == 0

    def test_full_returns_five(self):
        assert severity_to_k_max(1.0) == 5

    def test_quarter(self):
        assert severity_to_k_max(0.25) == 2

    def test_half(self):
        assert severity_to_k_max(0.5) == 3

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            severity_to_k_max(-0.1)
        with pytest.raises(ValueError):
            severity_to_k_max(1.1)

    def test_override_bypasses_formula(self):
        assert severity_to_k_max(0.55, override=25) == 25
        assert severity_to_k_max(0.0, override=10) == 10

    def test_override_clipped_at_hard_cap(self):
        assert severity_to_k_max(0.5, override=10_000) == 1024
        assert severity_to_k_max(0.5, override=10_000, hard_cap=50) == 50

    def test_override_negative_raises(self):
        with pytest.raises(ValueError, match="k_max override must be"):
            severity_to_k_max(0.5, override=-1)



# ===========================================================================
# Profile loader + builder
# ===========================================================================


class TestProfileLoader:
    def test_load_actual_library(self):
        from pathlib import Path
        lib_path = Path(__file__).resolve().parent.parent / "configs" / "templates" / "text_profiles.yaml"
        library = load_profile_library(lib_path)
        assert set(library) == {"question_pool_only", "noop"}
        assert [s.name for s in library["question_pool_only"].operators] == ["question_pool"]
        assert library["noop"].operators == ()

    def _lib(self):
        return {"p": TextProfile(name="p", operators=(OperatorSpec(name="question_pool", severity=1.0,
                                                                     extras={"w_txt": 1.0}),))}

    def test_resolve_profile_returns_specs(self):
        assert len(resolve_profile(self._lib(), profile_name="p")) == 1

    def test_resolve_with_overrides(self):
        specs = resolve_profile(self._lib(), profile_name="p", overrides={"question_pool": {"w_txt": 2.5}})
        assert specs[0].extras["w_txt"] == 2.5 and specs[0].severity == 1.0

    def test_resolve_explicit_operators(self):
        specs = resolve_profile(library={}, profile_name=None, operators=[{"name": "question_pool", "severity": 1.0}])
        assert len(specs) == 1 and specs[0].name == "question_pool"

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="not in library"):
            resolve_profile({}, profile_name="nonexistent")

    def test_neither_profile_nor_operators_raises(self):
        with pytest.raises(ValueError):
            resolve_profile({}, profile_name=None, operators=None)


class TestBuildOperators:
    def test_severity_zero_skipped(self):
        assert build_operators_from_specs((OperatorSpec(name="question_pool", severity=0.0),)) == []

    def test_question_pool_extras(self):
        ops = build_operators_from_specs((OperatorSpec(name="question_pool", severity=1.0,
                                                       extras={"w_txt": 1.5, "d_max": 0.3, "n_bands": 10}),))
        assert ops[0].name == "question_pool" and ops[0].w_txt == 1.5 and ops[0].d_max == 0.3 and ops[0].n_bands == 10

    @pytest.mark.parametrize("name", ["bogus", "synonym", "character_noise", "fragmentation", "saliency", "paraphrase"])
    def test_removed_or_unknown_operator_raises(self, name):
        with pytest.raises(ValueError, match="unknown text operator"):
            build_operators_from_specs((OperatorSpec(name=name, severity=0.5),))


# ===========================================================================
# SynonymOperator (MLM, stubbed)
# ===========================================================================


class _StubTokenizer:
    """Minimal HF-tokenizer stub for MLM-mask scoring."""

    def __init__(self, mask_token="[MASK]", mask_token_id=99, vocab=None):
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self._vocab = vocab or []

    def __call__(self, text, return_tensors="pt"):
        import torch
        words = text.split()
        ids = [self.mask_token_id if w == self.mask_token else 0 for w in words]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        class _Inputs:
            def __init__(self, ids, mask):
                self.input_ids = ids
                self.attention_mask = mask

            def to(self, device):
                return self

            def __iter__(self):
                return iter(["input_ids", "attention_mask"])

            def __getitem__(self, k):
                return getattr(self, k)

            def keys(self):
                return ["input_ids", "attention_mask"]

        return _Inputs(input_ids, attention_mask)

    def convert_ids_to_tokens(self, ids):
        return [self._vocab[i] if i < len(self._vocab) else f"tok_{i}" for i in ids]


class _StubMLM:
    def __init__(self, ranked_token_ids):
        import torch
        self._ranked = list(ranked_token_ids)
        vocab_size = max(self._ranked) + 1 if self._ranked else 1
        scores = torch.full((vocab_size,), -1e9)
        for rank, tid in enumerate(self._ranked):
            scores[tid] = float(len(self._ranked) - rank)
        self._scores = scores

    def __call__(self, **kwargs):
        import torch
        input_ids = kwargs["input_ids"]
        seq_len = input_ids.shape[1]
        vocab = self._scores.shape[0]
        logits = self._scores.unsqueeze(0).unsqueeze(0).expand(1, seq_len, vocab).clone()

        class _Out:
            pass
        out = _Out()
        out.logits = logits
        return out

    def to(self, device):
        return self

    def eval(self):
        return self


class TestSynonymOperator:
    """MLM-Synonym tests via stubbed model. Real ModernBERT is exercised
    in a separate smoke script, not in pytest."""

    def setup_method(self):
        import spacy
        # vocab[i] = token-string at id i; ranking = priority of MLM logits
        vocab = ["[PAD]", "key", "running", "non-main", "quickly", "main", "principal"]
        ranked = [1, 2, 3, 4, 5, 6]
        self.tokenizer = _StubTokenizer(mask_token="[MASK]", mask_token_id=99, vocab=vocab)
        self.model = _StubMLM(ranked_token_ids=ranked)
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.op = SynonymOperator(
            severity=1.0, k_max_override=10, topk_pre_filter=len(ranked),
        )
        self.op._set_resources(self.model, self.tokenizer, self.nlp)

    def test_severity_zero_disables(self):
        op = SynonymOperator(severity=0.0)
        ts = make_tokens(("the", "DET", " "), ("main", "ADJ", ""))
        ctx = op.prepare(ts)
        assert ctx.n_positions == 0
        assert op.gene_dim(ctx) == 0

    def test_negation_filter_rejects_non_prefixed(self):
        ts = make_tokens(
            ("What", "PRON", " "),
            ("is", "AUX", " "),
            ("the", "DET", " "),
            ("main", "ADJ", " "),
            ("subject", "NOUN", ""),
        )
        ctx = self.op.prepare(ts)
        flat = [c for cands in ctx.candidates for c in cands]
        assert "non-main" not in flat

    def test_apply_zero_genes_no_op(self):
        ts = make_tokens(
            ("What", "PRON", " "),
            ("is", "AUX", " "),
            ("the", "DET", " "),
            ("main", "ADJ", " "),
            ("subject", "NOUN", ""),
        )
        ctx = self.op.prepare(ts)
        if ctx.n_positions == 0:
            pytest.skip("stub MLM produced no surviving candidates")
        genes = np.zeros(ctx.n_positions, dtype=np.int64)
        assert self.op.apply(ctx, genes, ts).text == ts.text

    def test_negation_regex_pattern(self):
        should_match = (
            "non-main", "non_main", "un-easy", "in-correct", "im-pressed",
            "de-activate", "dis-able", "anti-pattern", "counter-act",
        )
        should_not_match = (
            "main", "subject", "image", "key", "primary", "topic",
            "incident", "imply", "unease", "deduce", "antibody", "nonmain",
        )
        for s in should_match:
            assert NEGATION_PREFIX_RE.match(s) is not None, f"should match: {s}"
        for s in should_not_match:
            assert NEGATION_PREFIX_RE.match(s) is None, f"should NOT match: {s}"

    def test_k_max_override(self):
        op = SynonymOperator(severity=0.5, k_max_override=42)
        assert op.k_max == 42

    def test_default_k_max_from_severity(self):
        op = SynonymOperator(severity=0.5)
        assert op.k_max == 3


# ===========================================================================

# ===========================================================================
# CompositeTextManipulator orchestration (stubbed nlp, no model load)
# ===========================================================================


class _StubNLP:
    """Minimal spaCy-API stub matching tokenize() expectations."""

    def __call__(self, text):
        det_set = {"the", "a", "an"}
        adj_set = {"quick", "brown", "rapid", "fast", "lazy"}
        items = []
        for chunk in text.split(" "):
            if not chunk:
                continue
            stripped = chunk.rstrip(".!?,;:")
            trailing_punct = chunk[len(stripped):]
            if stripped.lower() in det_set:
                pos = "DET"
            elif stripped.lower() in adj_set:
                pos = "ADJ"
            else:
                pos = "NOUN"
            items.append((stripped, pos, " "))
            if trailing_punct:
                items.append((trailing_punct, "PUNCT", " "))
        if items:
            last_token, last_pos, _ = items[-1]
            items[-1] = (last_token, last_pos, "")

        class _Tok:
            def __init__(self, text, pos, ws):
                self.text = text
                self.pos_ = pos
                self.whitespace_ = ws

        class _Doc:
            def __init__(self, toks):
                self._toks = toks

            def __iter__(self):
                return iter(self._toks)

        return _Doc([_Tok(t, p, w) for t, p, w in items])


class _IntOp:
    """Minimal integer-gene operator (one gene per NOUN, gene k > 0 appends k '!' to the word)."""

    name = "int_stub"

    def prepare(self, tokens, exclude_words=None):
        return OperatorContext(positions=np.array([i for i, p in enumerate(tokens.pos_tags) if p == "NOUN"], dtype=np.intp))

    def gene_dim(self, ctx):
        return ctx.n_positions

    def gene_bounds(self, ctx):
        return np.full(ctx.n_positions, 3, dtype=np.int64)

    def apply(self, ctx, genes, current):
        toks = list(current.tokens)
        for p, k in zip(ctx.positions, genes):
            toks[p] = toks[p] + "!" * int(k)
        return TokenSequence(tokens=tuple(toks), pos_tags=current.pos_tags, whitespace=current.whitespace)


@pytest.fixture
def pool(tmp_path):
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"meta": {"d_max": 0.25, "n_bands": 10}, "questions": {"The quick brown fox": {"pool": [
        {"text": "A quick brown fox", "dist": 0.05}, {"text": "The fast brown fox", "dist": 0.2}]}}}))
    load_pool.cache_clear()
    yield str(p)
    load_pool.cache_clear()


class TestCompositeTextManipulator:
    def setup_method(self):
        self.nlp = _StubNLP()

    def test_int_operator_path(self):
        cm = CompositeTextManipulator(self.nlp, operators=[_IntOp()])
        ctx = cm.prepare("The quick brown fox")
        assert ctx.genotype_dim == 1 and not ctx.is_continuous
        assert cm.apply(ctx, ctx.zero_genotype()) == "The quick brown fox"
        assert cm.apply(ctx, np.array([2])) == "The quick brown fox!!"
        bounds = ctx.gene_bounds
        assert bounds.dtype == np.int64 and len(bounds) == ctx.genotype_dim
        rng = np.random.default_rng(42)
        for _ in range(50):
            g = ctx.random_genotype(rng)
            assert (g >= 0).all() and (g < bounds).all()

    def test_question_pool_only(self, pool):
        cm = CompositeTextManipulator(self.nlp, operators=[QuestionPoolOperator(pool=pool)])
        ctx = cm.prepare("The quick brown fox")
        assert cm.is_continuous and ctx.genotype_dim == 2
        assert cm.apply(ctx, ctx.zero_genotype()) == "The quick brown fox"
        assert cm.apply(ctx, np.array([0.2, 0.0])) == "A quick brown fox"
        assert cm.apply(ctx, np.array([0.8, 0.0])) == "The fast brown fox"

    def test_genotype_length_mismatch_raises(self, pool):
        cm = CompositeTextManipulator(self.nlp, operators=[QuestionPoolOperator(pool=pool)])
        ctx = cm.prepare("The quick brown fox")
        with pytest.raises(ValueError, match="genotype length"):
            cm.apply(ctx, np.zeros(ctx.genotype_dim + 1))

    def test_empty_composite_noop(self):
        cm = CompositeTextManipulator(self.nlp, operators=[])
        ctx = cm.prepare("The quick brown fox")
        assert ctx.genotype_dim == 0 and cm.apply(ctx, ctx.zero_genotype()) == "The quick brown fox"


# ===========================================================================
# Canonical order
# ===========================================================================


class TestCanonicalOrder:
    def test_canonical_order(self):
        assert CANONICAL_ORDER == ("question_pool",)
