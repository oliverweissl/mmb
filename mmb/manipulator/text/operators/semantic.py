"""Embedding-ranked word substitution (T-A, 2026-11-28): the text analogue of the image channel's codebook ladder.

The image channel replaces a region's VQGAN codes by the r-th nearest live codeword; strength = rank r. Here a word is
replaced by the r-th nearest *meaning-preserving* substitute, where "nearest" is measured on the whole question:

1. **Candidates** -- contextual masked-LM proposals for the masked word (ModernBERT, as the synonym operator), with
   the synonym operator's filters: alphabetic, same fine-grained PoS tag in context, different lemma, no negation
   prefix. Sites: every alphabetic, non-protected word (`all_words`, default), or content words only.
2. **Meaning and grammar filters** -- the substituted question must be a mutual entailment of the original (NLI
   cross-encoder, entailment probability >= `nli_min` (0.8) in both directions; "main" -> "primary" passes, "main"
   -> "only" and "in" -> "like" fail) and grammatical (CoLA acceptability >= max(min(`cola_min`, a0), `cola_rel` x a0), a0 = the original's;
   i.e. at most 10% less acceptable; "in" -> "at" in "Is there a broccoli at the image?" fails). Substitutes must be dictionary words (wamerican; drops subword
   fragments such as "th") with the same coarse PoS in context ("in" -> "if" changes ADP to SCONJ); articles must agree
   with the next word (no "an broccoli"); auxiliaries are never edited.
3. **Ranking** -- by the sentence-encoder distance d = 1 - cos(enc(original), enc(substituted)) (all-MiniLM-L6-v2,
   the encoder the validity proxies use), nearest first (ties: MLM rank).
4. **Ladder** -- gene k >= 1 picks the `ladder[k-1]`-th nearest substitute (1-based), e.g. ladder [1, 2, 4, 8];
   gene 0 keeps the word. A site keeps as many levels as it has substitutes for.

No typos, homoglyphs or zero-width characters: every edited question consists of real words. The text strength of a
crossing is the same sentence distance d of the full edited question (validity column txt_minilm_cos).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import TokenSequence
from .synonym import _LOAD_LOCK, SynonymContext, SynonymOperator

logger = logging.getLogger(__name__)

# sites with all_words: every open/closed class except auxiliaries (they carry the question form: "Is" -> "Does"
# turns "Is there a ..." ungrammatical) and proper nouns
ALL_UPOS = frozenset({"ADJ", "ADP", "ADV", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "SCONJ", "VERB"})
WORDLIST = "/usr/share/dict/american-english"   # Debian/Ubuntu package wamerican: substitutes must be dictionary words
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
NLI = "cross-encoder/nli-deberta-v3-base"
COLA = "textattack/roberta-base-CoLA"           # grammatical acceptability (label 1 = acceptable)
VOWEL_SOUND_EXCEPTIONS = ("hour", "honest", "honour", "honor", "heir")
CONSONANT_SOUND_EXCEPTIONS = ("uni", "use", "user", "usu", "eu", "one", "once")


@dataclass(frozen=True)
class SemanticContext(SynonymContext):
    """Synonym context plus the sentence distance of each kept substitute (same layout as candidates)."""

    distances: tuple[tuple[float, ...], ...] = ()


class SemanticSubstitutionOperator(SynonymOperator):
    """gene 0 = keep the word; gene k = the ladder[k-1]-th nearest meaning-preserving substitute."""

    name = "semantic_substitution"

    def __init__(self, ladder=(1, 2, 4, 8), *, all_words: bool = True, nli_min: float = 0.8, cola_min: float = 0.5,
                 cola_rel: float = 0.9,
                 encoder_name: str = ENCODER, nli_name: str = NLI, cola_name: str = COLA, **kw: Any) -> None:
        if all_words:
            kw["content_pos"] = ALL_UPOS
        kw.setdefault("severity", 1.0)
        super().__init__(**kw)
        self._ladder = tuple(int(r) for r in ladder)
        if not self._ladder or any(r < 1 for r in self._ladder) or list(self._ladder) != sorted(set(self._ladder)):
            raise ValueError(f"ladder must be strictly increasing ranks >= 1, got {ladder}")
        self._k_max = len(self._ladder)
        self._nli_min, self._cola_min, self._cola_rel = float(nli_min), float(cola_min), float(cola_rel)
        self._encoder_name, self._nli_name, self._cola_name = encoder_name, nli_name, cola_name
        self._cola: Any = None
        self._cola_tok: Any = None
        self._encoder: Any = None
        self._nli: Any = None
        self._nli_tok: Any = None
        self._ent_idx: int | None = None

    @property
    def ladder(self) -> tuple[int, ...]:
        return self._ladder

    # -- models -----------------------------------------------------------------------------------------------
    def _ensure_semantic_loaded(self) -> None:
        if self._nli is not None:
            return
        with _LOAD_LOCK:
            if self._nli is not None:
                return
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            dev = torch.device(self._device_str)
            self._encoder = SentenceTransformer(self._encoder_name, device=str(dev))
            tok = AutoTokenizer.from_pretrained(self._nli_name)
            nli = AutoModelForSequenceClassification.from_pretrained(self._nli_name).to(dev).eval()
            lab = {v.lower(): k for k, v in nli.config.id2label.items()}
            self._ent_idx = int(lab["entailment"])
            if self._cola_min > 0:
                self._cola_tok = AutoTokenizer.from_pretrained(self._cola_name)
                self._cola = AutoModelForSequenceClassification.from_pretrained(self._cola_name).to(dev).eval()
            self._nli_tok = tok
            self._nli = nli

    def _distances(self, original: str, texts: list[str]) -> np.ndarray:
        e = self._encoder.encode([original] + texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.clip(1.0 - e[1:] @ e[0], 0.0, 2.0)

    def _entail(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(premises), 32):
            b = self._nli_tok(premises[i:i + 32], hypotheses[i:i + 32], return_tensors="pt", padding=True,
                              truncation=True).to(self._nli.device)
            with torch.no_grad():
                out.append(self._nli(**b).logits.softmax(-1)[:, self._ent_idx].float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def _same_upos(self, tokens: TokenSequence, p: int, cand: str, upos: str) -> bool:
        doc = self._tag_in_context(tokens, p, cand)
        return doc is not None and doc[p].pos_ == upos

    def acceptability(self, texts: list[str]) -> np.ndarray:
        """CoLA P(acceptable) per text (1.0 when the grammar filter is off)."""
        import torch

        if self._cola is None or not texts:
            return np.ones(len(texts))
        out = []
        for i in range(0, len(texts), 32):
            b = self._cola_tok(texts[i:i + 32], return_tensors="pt", padding=True, truncation=True).to(self._cola.device)
            with torch.no_grad():
                out.append(self._cola(**b).logits.softmax(-1)[:, 1].float().cpu().numpy())
        return np.concatenate(out)

    def mutual_entailment(self, original: str, texts: list[str]) -> np.ndarray:
        """min(P(orig => text), P(text => orig)) per text."""
        if not texts:
            return np.empty(0)
        return np.minimum(self._entail([original] * len(texts), texts), self._entail(texts, [original] * len(texts)))

    # -- operator -----------------------------------------------------------------------------------------------
    def prepare(self, tokens: TokenSequence, exclude_words: frozenset[str] | None = None) -> SemanticContext:
        # candidate pool: the synonym operator's filtered MLM proposals, unlimited depth (topk_pre_filter)
        k_keep = self._k_max
        self._k_max = self._topk_pre
        try:
            base = super().prepare(tokens, exclude_words)
        finally:
            self._k_max = k_keep
        if base.n_positions == 0:
            return SemanticContext(positions=base.positions, candidates=(), original_words=(), distances=())
        self._ensure_semantic_loaded()
        original = "".join(t + w for t, w in zip(tokens.tokens, tokens.whitespace))
        if self._cola is not None:      # at most a 10% loss of acceptability, floor cola_min (or the original's own score)
            a0 = float(self.acceptability([original])[0])
            cola_thr = max(min(self._cola_min, a0), self._cola_rel * a0)
        else:
            cola_thr = 0.0
        pos_out, cand_out, orig_out, dist_out = [], [], [], []
        for p, cands, ow in zip(base.positions, base.candidates, base.original_words):
            nxt = tokens.tokens[int(p) + 1] if int(p) + 1 < tokens.n_tokens else ""
            upos = tokens.pos_tags[int(p)]
            words = _wordlist()
            cands = tuple(c for c in cands if (len(c) > 1 or c.lower() == "a") and _article_ok(c, nxt)
                          and (words is None or c.lower() in words) and self._same_upos(tokens, int(p), c, upos))
            texts = []
            for c in cands:
                sub = list(tokens.tokens)
                sub[int(p)] = _match_case(ow, c)
                texts.append("".join(t + w for t, w in zip(sub, tokens.whitespace)))
            ent = self.mutual_entailment(original, texts)
            acc = self.acceptability(texts)
            keep = [i for i in range(len(texts)) if ent[i] >= self._nli_min and acc[i] >= cola_thr]
            if not keep:
                continue
            d = self._distances(original, [texts[i] for i in keep])
            order = sorted(range(len(keep)), key=lambda j: (round(float(d[j]), 6), keep[j]))     # nearest, then MLM rank
            ranked = [(_match_case(ow, cands[keep[j]]), float(d[j])) for j in order]
            lv = [ranked[r - 1] for r in self._ladder if r <= len(ranked)]
            pos_out.append(int(p)); orig_out.append(ow)
            cand_out.append(tuple(w for w, _ in lv)); dist_out.append(tuple(x for _, x in lv))
        return SemanticContext(positions=np.array(pos_out, dtype=np.intp), candidates=tuple(cand_out),
                               original_words=tuple(orig_out), distances=tuple(dist_out))


_WORDS: frozenset[str] | None = None


def _wordlist() -> frozenset[str] | None:
    global _WORDS
    if _WORDS is None:
        from pathlib import Path
        p = Path(WORDLIST)
        if not p.exists():
            logger.warning("%s missing: dictionary filter off", WORDLIST)
            _WORDS = frozenset()
        else:
            # lowercase entries only: capitalised ones are names and symbols ("Th" would admit the fragment "th")
            _WORDS = frozenset(w.strip() for w in p.read_text(errors="ignore").splitlines()
                               if w.strip().isalpha() and w.strip().islower())
    return _WORDS or None


def _article_ok(cand: str, next_word: str) -> bool:
    """a/an agree with the sound of the next word (spelling heuristic with common exceptions)."""
    c, w = cand.lower(), next_word.lower()
    if c not in ("a", "an") or not w[:1].isalpha():
        return True
    vowel = w[0] in "aeiou"
    if w.startswith(VOWEL_SOUND_EXCEPTIONS):
        vowel = True
    elif w.startswith(CONSONANT_SOUND_EXCEPTIONS):
        vowel = False
    return (c == "an") == vowel


def _match_case(orig: str, cand: str) -> str:
    if orig[:1].isupper() and not orig.isupper():
        return cand[:1].upper() + cand[1:]
    return cand.lower() if orig.islower() else cand
