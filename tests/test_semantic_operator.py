"""T-A text operator: embedding-ranked word substitution (mmb.manipulator.text.operators.semantic)."""
import numpy as np
import pytest

from mmb.manipulator.text.operators import semantic as S
from mmb.manipulator.text.profiles import OperatorSpec, build_operators_from_specs


def test_article_agreement():
    assert S._article_ok("an", "apple") and not S._article_ok("an", "broccoli")
    assert S._article_ok("a", "broccoli") and not S._article_ok("a", "image")
    assert S._article_ok("an", "hour") and S._article_ok("a", "unicorn")
    assert S._article_ok("the", "image") and S._article_ok("an", "")


def test_match_case():
    assert S._match_case("What", "which") == "Which"
    assert S._match_case("main", "Major") == "major"


@pytest.mark.parametrize("bad", [(), (0, 1), (2, 1), (1, 1, 2)])
def test_ladder_validated(bad):
    with pytest.raises(ValueError):
        S.SemanticSubstitutionOperator(ladder=bad)


def test_builder_and_no_aux_sites():
    ops = build_operators_from_specs((OperatorSpec(name="semantic_substitution", severity=1.0,
                                                   extras={"ladder": [1, 2, 4], "nli_min": 0.9}),))
    assert len(ops) == 1 and ops[0].name == "semantic_substitution" and ops[0].ladder == (1, 2, 4) and ops[0].k_max == 3
    assert "AUX" not in S.ALL_UPOS and "PROPN" not in S.ALL_UPOS


def _models_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(m, "config.json"), str)
                   for m in ("answerdotai/ModernBERT-large", S.NLI, S.COLA, S.ENCODER))
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _models_cached(), reason="MLM / NLI / CoLA / encoder not in the local HF cache")
@pytest.mark.parametrize("q,prot", [("What is the main subject in this image?", {"subject", "image"}),
                                    ("Is there a dining table in the image?", {"dining", "table", "image"})])
def test_prepare_real_models(q, prot):
    import spacy
    from mmb.manipulator.text.types import tokenize
    nlp = spacy.load("en_core_web_sm")
    toks = tokenize(nlp, q)
    op = S.SemanticSubstitutionOperator(ladder=(1, 2, 4, 8), device="cpu")
    ctx = op.prepare(toks, frozenset(prot))
    words = S._wordlist()
    assert ctx.n_positions >= 2
    for p, cands, d in zip(ctx.positions, ctx.candidates, ctx.distances):
        assert toks.tokens[p].lower() not in prot and toks.pos_tags[p] != "AUX"
        assert 1 <= len(cands) <= 4 and list(d) == sorted(d)                     # ranked by sentence distance
        assert all(c.lower() in words for c in cands)
    b = op.gene_bounds(ctx)
    assert list(b) == [len(c) + 1 for c in ctx.candidates]
    g = np.zeros(ctx.n_positions, dtype=np.int64); g[0] = 1
    out = op.apply(ctx, g, toks)
    assert out.tokens[ctx.positions[0]] == ctx.candidates[0][0]
    assert all(t.isalpha() or not t.isalnum() for t in out.tokens)          # real words only, no invisible characters
