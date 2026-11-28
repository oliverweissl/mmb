"""Continuous text block: the (t, u) -> question mapping (mmb.manipulator.text.operators.question_pool)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from mmb.manipulator.text.operators.question_pool import (QuestionPoolOperator, _tokens, load_pool, scale,
                                                         select_index)

D = np.array([0.0, 0.01, 0.012, 0.05, 0.051, 0.052, 0.12, 0.2, 0.21, 0.33])   # [0] = the original
KW = dict(w_txt=1.0, d_max=0.25, n_bands=10)


def test_t_zero_is_identity():
    for u in (0.0, 0.3, 0.999, 1.0):
        assert select_index(D, 0.0, u, **KW) == 0
    assert select_index(D, -0.5, 0.5, **KW) == 0 and select_index(D, float("nan"), 0.5, **KW) == 0
    assert select_index([0.0], 0.7, 0.2, **KW) == 0        # empty pool -> original


@pytest.mark.parametrize("u", [0.0, 0.1, 0.33, 0.5, 0.77, 0.999])
def test_monotone_in_t(u):
    ts = np.linspace(0, 1, 2001)
    d = [D[select_index(D, t, u, **KW)] for t in ts]
    assert all(b >= a for a, b in zip(d, d[1:]))
    assert d[-1] > 0


def test_monotone_random_pools():
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = np.concatenate([[0.0], np.sort(rng.uniform(0, 0.5, rng.integers(1, 60)))])
        w = float(rng.uniform(0.5, 2))
        for u in rng.uniform(0, 1, 5):
            sel = [d[select_index(d, t, u, w_txt=w, d_max=0.3, n_bands=15)] for t in np.linspace(0, 1, 301)]
            assert all(b >= a for a, b in zip(sel, sel[1:]))


def test_deterministic_and_pure():
    args = [(0.3, 0.2), (0.8, 0.9), (0.55, 0.5)]
    d0 = D.copy()
    assert [select_index(D, t, u, **KW) for t, u in args] == [select_index(d0, t, u, **KW) for t, u in args]
    assert np.array_equal(d0, D)


def test_target_band_and_u_choice():
    # h = 0.025; t = 0.2 -> target 0.05 -> band [0.05, 0.075) = entries 3, 4, 5; u picks among them in distance order
    assert [select_index(D, 0.2, u, **KW) for u in (0.0, 0.34, 0.67, 0.9999, 1.0)] == [3, 4, 5, 5, 5]
    # t = 0.36 -> target 0.09: band 3 empty; band 4 (gap 0.01) is nearer than band 2 (gap 0.015)
    assert select_index(D, 0.36, 0.0, **KW) == 6
    # exact tie between two occupied bands -> the lower band
    assert select_index([0.0, 0.1, 0.8], 0.5, 0.99, w_txt=1.0, d_max=1.0, n_bands=4) == 1
    # target beyond the pool -> the top band (ceiling)
    assert select_index(D, 1.0, 0.0, w_txt=2.0, d_max=0.25, n_bands=10) == 9


def test_w_txt_scales_the_target():
    for t in np.linspace(0, 0.5, 51):
        for u in (0.0, 0.5, 0.9):
            assert select_index(D, t, u, w_txt=2.0, d_max=0.25, n_bands=10) == select_index(D, 2 * t, u, **KW)


def test_scale():
    g = np.array([0.4, 0.7])
    assert np.allclose(scale(g, 0.5), [0.2, 0.7]) and np.allclose(scale(g, 0.0), [0.0, 0.7])
    assert np.allclose(scale(g, 5.0), [1.0, 0.7]) and np.allclose(g, [0.4, 0.7])     # clipped; input untouched


@pytest.fixture
def pool(tmp_path):
    p = tmp_path / "pool.json"
    q = "Is there a dog in the image?"
    p.write_text(json.dumps({"meta": {"d_max": 0.25, "n_bands": 10}, "questions": {q: {"pool": [
        {"text": f"Q{i} dog image?", "dist": float(x)} for i, x in enumerate(D[1:], 1)]}}}))
    load_pool.cache_clear()
    yield str(p)
    load_pool.cache_clear()


def test_operator_apply(pool):
    op = QuestionPoolOperator(pool=pool)
    t = _tokens("Is there a dog in the image?")
    ctx = op.prepare(t)
    assert op.gene_dim(ctx) == 2 and list(op.gene_bounds(ctx)) == [1.0, 1.0]
    assert op.apply(ctx, np.array([0.0, 0.8]), t) is t
    assert op.apply(ctx, np.array([0.2, 0.5]), t).text == "Q4 dog image?"
    assert op.render(ctx, [0.2, 0.5]) == ("Q4 dog image?", 0.051)
    assert QuestionPoolOperator(pool=pool, w_txt=2.0).render(ctx, [0.1, 0.5]) == ("Q4 dog image?", 0.051)
    # unknown question: no genes
    assert op.gene_dim(op.prepare(_tokens("Is there a cat in the image?"))) == 0


def test_composite_keeps_float_genes(pool):
    import spacy
    from mmb.manipulator.text.composite import CompositeTextManipulator
    cm = CompositeTextManipulator(spacy.blank("en"), [QuestionPoolOperator(pool=pool)])
    ctx = cm.prepare("Is there a dog in the image?")
    assert ctx.genotype_dim == 2
    assert cm.apply(ctx, np.array([0.2, 0.5])) == "Q4 dog image?"     # an int cast would give t = 0
    assert cm.apply(ctx, np.zeros(2)) == "Is there a dog in the image?"


def test_versioned_pool():
    p = Path(__file__).parents[1] / "configs" / "text" / "question_pool.json"
    if not p.exists():
        pytest.skip("pool not built")
    P = json.loads(p.read_text())
    alld = []
    for q, e in P["questions"].items():
        d = [x["dist"] for x in e["pool"]]
        assert d == sorted(d) and all(x > 0 for x in d)
        keys = [re.sub(r"\s+", " ", x["text"].lower()).rstrip("?").strip() for x in e["pool"]]
        assert len(keys) == len(set(keys)) and q.lower().rstrip("?") not in keys
        for x in e["pool"]:
            assert x["text"].endswith("?") and x["nli"] >= 0.8 and x["cola"] >= e["cola_threshold"]
            assert all(re.search(r"\b" + re.escape(w) + r"\b", x["text"].lower()) for w in e["protect"])
        alld += d
    assert P["meta"]["d_max"] == pytest.approx(np.quantile(alld, P["meta"]["d_max_quantile"]), abs=1e-4)


# --- composite integration (continuous text block) ---------------------------------------------------------------

def _cm(pool, extra_ops=()):
    import spacy
    from mmb.manipulator.text.composite import CompositeTextManipulator
    return CompositeTextManipulator(spacy.blank("en"), [QuestionPoolOperator(pool=pool), *extra_ops])


def test_composite_continuous_bounds_and_genotypes(pool):
    cm = _cm(pool)
    ctx = cm.prepare("Is there a dog in the image?")
    assert cm.is_continuous and ctx.is_continuous
    assert np.array_equal(ctx.xl, [0.0, 0.0]) and np.array_equal(ctx.xu, [1.0, 1.0])
    z = ctx.zero_genotype()
    assert z.dtype == np.float64 and np.array_equal(z, [0.0, 0.0])
    assert cm.apply(ctx, z) == "Is there a dog in the image?"
    g = np.stack([ctx.random_genotype(np.random.default_rng(i)) for i in range(200)])
    assert g.dtype == np.float64 and (g >= 0).all() and (g < 1).all()          # u upper-open
    assert np.array_equal(ctx.random_genotype(np.random.default_rng(3)), ctx.random_genotype(np.random.default_rng(3)))
    assert cm.apply(ctx, [0.3, 1.0]) == cm.apply(ctx, [0.3, np.nextafter(1.0, 0.0)])   # u = 1 is the last member


def test_composite_scale_and_w_txt(pool):
    cm = _cm(pool)
    ctx = cm.prepare("Is there a dog in the image?")
    assert np.allclose(cm.scale([0.6, 0.4], 0.5), [0.3, 0.4])
    assert np.allclose(cm.scale([0.6, 0.4], 0.5, ctx), [0.3, 0.4])
    assert cm.w_txt == 1.0
    before = cm.apply(ctx, [0.2, 0.5])
    cm.w_txt = 2.0
    assert cm.w_txt == 2.0 and cm.apply(ctx, [0.1, 0.5]) == before        # w doubles the target
    with pytest.raises(ValueError):
        cm.scale([0.6, 0.4, 0.1], 0.5, ctx)


def test_composite_unknown_question_is_empty(pool):
    cm = _cm(pool)
    ctx = cm.prepare("Is there a cat in the image?")
    assert ctx.genotype_dim == 0 and not ctx.is_continuous and ctx.xu.size == 0
    assert ctx.random_genotype(np.random.default_rng(0)).size == 0
    assert cm.apply(ctx, ctx.zero_genotype()) == "Is there a cat in the image?"


def test_composite_mixed_int_and_continuous(pool):
    from mmb.manipulator.text.operators.base import OperatorContext

    class IntOp:                                     # one integer gene (0..2) per token
        name = "int_stub"

        def prepare(self, tokens, exclude_words=None):
            return OperatorContext(positions=np.arange(tokens.n_tokens, dtype=np.intp))

        def gene_dim(self, ctx):
            return ctx.n_positions

        def gene_bounds(self, ctx):
            return np.full(ctx.n_positions, 3, dtype=np.int64)

        def apply(self, ctx, genes, current):
            return current

    cm = _cm(pool, [IntOp()])
    assert not cm.is_continuous
    ctx = cm.prepare("Is there a dog in the image?")
    di = ctx.op_gene_dims[ctx.op_order.index("int_stub")]
    assert di > 0
    assert ctx.genotype_dim == 2 + di and ctx.xu.size == ctx.genotype_dim
    g = ctx.random_genotype(np.random.default_rng(0))
    assert g.dtype == np.float64 and (g <= ctx.xu).all() and (g >= ctx.xl).all()
    with pytest.raises((TypeError, ValueError)):
        cm.scale(g, 0.5, ctx)


def test_profile_question_pool_only():
    from mmb.manipulator.text.profiles import build_operators_from_specs, load_profile_library, resolve_profile
    lib = load_profile_library(Path(__file__).parents[1] / "configs" / "templates" / "text_profiles.yaml")
    ops = build_operators_from_specs(resolve_profile(lib, "question_pool_only"))
    assert [o.name for o in ops] == ["question_pool"] and ops[0].continuous and ops[0].w_txt == 1.0
