"""T-B paraphrase gene (mmb.manipulator.text.operators.paraphrase)."""
import json

import numpy as np
import pytest

from mmb.manipulator.text.operators.paraphrase import ParaphraseOperator, _load_bank, _tokens


@pytest.fixture
def bank(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({"Is there a dog in the image?": {"protect": ["dog", "image"], "paraphrases": [
        {"text": f"P{i} dog image?", "dist": 0.01 * i} for i in range(1, 11)]}}))
    _load_bank.cache_clear()
    return str(p)


def test_ladder_and_apply(bank):
    op = ParaphraseOperator(ladder=(1, 2, 4, 8, 16), bank=bank)
    t = _tokens("Is there a dog in the image?")
    ctx = op.prepare(t)
    assert ctx.candidates == ("P1 dog image?", "P2 dog image?", "P4 dog image?", "P8 dog image?")   # rank 16 > 10 dropped
    assert list(op.gene_bounds(ctx)) == [5] and op.gene_dim(ctx) == 1
    assert op.apply(ctx, np.array([0]), t) is t
    assert op.apply(ctx, np.array([3]), t).text == "P4 dog image?"


def test_unknown_prompt_has_no_gene(bank):
    op = ParaphraseOperator(bank=bank)
    ctx = op.prepare(_tokens("Is there a cat in the image?"))
    assert op.gene_dim(ctx) == 0 and len(op.gene_bounds(ctx)) == 0


def test_versioned_bank_keeps_protected_words():
    from pathlib import Path
    import re
    p = Path(__file__).parents[1] / "configs" / "text" / "paraphrase_bank.json"
    if not p.exists():
        pytest.skip("bank not built")
    b = json.loads(p.read_text())
    for q, e in b.items():
        for x in e["paraphrases"]:
            assert x["text"].endswith("?") and all(re.search(r"\b" + re.escape(w) + r"\b", x["text"].lower()) for w in e["protect"])
        d = [x["dist"] for x in e["paraphrases"]]
        assert d == sorted(d)
