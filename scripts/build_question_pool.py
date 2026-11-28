#!/usr/bin/env python
"""Question pool for the continuous text block (2026-11-28; operator mmb/manipulator/text/operators/question_pool.py).

Per distinct question of cells/*_calib*.yaml and cells/*_cells.yaml, a pool of WHOLE edited questions:
1. substitutions: the SemanticSubstitutionOperator's per-word substitutes (each already NLI/CoLA-filtered alone,
   all ranks) combined over 1..3 word sites (every combination, capped at --max-combos per question, seeded sample);
2. bank: the reviewed paraphrases of configs/text/paraphrase_bank.json;
3. t5: extra T5 paraphrases (humarin/chatgpt_paraphraser_on_T5_base): seeded samples of the original at two
   temperatures, plus paraphrases of every bank paraphrase (second order);
4. growth (default: every question, so all pools are built the same way; --grow-below N limits it to pools with fewer than N entries after 1-3): word substitutions (1..2 sites)
   on top of every accepted paraphrase (source "paraphrase+substitution"), and more T5 samples (seeds 3-5,
   temperatures 1.0/1.3/1.6, plus second-order paraphrases of every accepted paraphrase; source "t5").
   Growth candidates may only use words that occur in the originals or in an accepted pass-1 entry of any question
   (recombination of vetted vocabulary; others rejected as "growth: word(s) outside ...").
Every candidate is validated AS A WHOLE question (first failing reason recorded): ends with "?"; protected words kept;
dictionary words; a/an agreement; the T-B meaning rules (no presupposed object, identity question, whole-image
reading, second medium word); NLI mutual entailment >= 0.8; CoLA >= max(min(0.5, a0), 0.9 a0). De-duplicated
(case, whitespace, final "?"); ordered by MiniLM distance d = 1 - cos. meta.d_max = the --d-max-quantile of all pool
distances over all questions (the text block's global scale), meta.n_bands = the band count of the (t, u) mapping.

    ../.venv/bin/python scripts/build_question_pool.py --out configs/text/question_pool.json
"""
import argparse, itertools, json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import build_paraphrase_bank as B  # noqa: E402  (meaning_rule, prompts(), GEN)
from mmb.manipulator.text.operators import semantic as S  # noqa: E402
from mmb.manipulator.text.types import tokenize  # noqa: E402

CELL_GLOBS = ("cells/*_calib*.yaml", "cells/*_cells.yaml")
NLI_MIN = 0.8


def prompts() -> dict[str, list[str]]:
    import glob, yaml
    out: dict[str, set] = {}
    for f in sorted({p for g in CELL_GLOBS for p in glob.glob(g)}):
        for it in yaml.safe_load(open(f))["seeds"]["slot_items"]["items"]:
            out.setdefault(it["prompt"], set()).update(str(w).lower() for w in (it.get("meta") or {}).get("protect_words") or [])
    # a phrase ("dining table") subsumes its words
    return {q: sorted(p for p in s if not any(p != o and re.search(r"\b" + re.escape(p) + r"\b", o) for o in s)) for q, s in out.items()}


# extra meaning rules (C1, 2026-11-28; for user review): patterns that passed NLI but change what is asked
EXTRA_RULES = (
    (r"\b(is|be|was)\s+(the|this|that)\s+image\s+(a|an|of|from)\b|\bimage\s+(is|was)\s+(a|an|of|from)\b"
     r"|\brepresentation of\b|\bimage below\b", "whole-image reading (the image IS / represents the object)"),
    (r"\b(an|another|any|some)\s+image\b", "changes the referent ('an image' instead of the given image)"),
    (r"\b(with|for|across|about|into|under|by|like|over|through|onto|upon|via|toward|towards|at)\s+(the|this|that|an|a)?\s*(\w+\s+)?image\b",
     "odd preposition before 'image'"),
    (r"\babove image\b|\bpictured image\b|\ban above\b", "odd reference to the image ('above image', 'pictured image')"),
    (r"\banother\b", "'another' presupposes a first object"),
    (r"\b(can|could|do|did|would|will)\s+(me|they|he|she|them|him|her|us)\b|\bcan i\b", "changes who is asked (they / me / I)"),
    (r"\b(successfully|actually|really)\b", "adverb that changes the question"),
    (r"\b(during|against|off|amidst|behind|beyond|throughout)\s+(the|this|that|an|a)?\s*(\w+\s+)?image\b",
     "odd preposition before 'image'"),
    (r"\bsubject\s+(label|text|material|field|information|character|area|topic|content)\b|\bwhatever\b",
     "changes what 'subject' asks for"),
    (r"\b(real|okay|credible|probability|sentence|ye|reflect|reflects|reflected|reflecting)\b",
     "changes the question (realness, permission, probability, 'reflect')"),
    (r"\bcan (it|one|someone)\b", "changes who is asked (it / one / someone)"),
    (r"\b(can|could)\s+(you|we)\s+know\b|\bexpect\b|\bvisible to\b|\bthat presence\b|\bis it (the case|true) if\b"
     r"|\baccurate to (detect|confirm|verify|tell)\b", "odd question form ('can you know', 'expect', 'is it the case if')"),
    (r"\bto\s+(include|display|add|put|insert|place|integrate|show)\s+(a|an|any)\b",
     "asks about putting the object into the image"),
    (r"\b(do|does|did)\s+(you|we|i)\s+(include|have|display|show|feature|contain|capture|depict|present|portray|showcase)\b",
     "the asker (not the image) contains the object ('do you include a dog')"),
    (r"\b(help me|assist me|can we use|use this image|identify it|name it)\b", "a request or task instead of the question"),
    (r"\b(do you (believe|think)|might it be|feasible|possible to come across|factual occurrence)\b",
     "hedged / opinion question"),
    (r"\b(piece|stalk|mass|kind|type|sort|pair|bunch|head|bit|speck|glimpse|produce)\s+of\s+\w+", "quantity or kind of the object"),
    (r"\b(is|be|being|can|could)\s+(the\s+|this\s+|that\s+)?image\b|\bimage\s+(be|being|appear)\b"
     r"|\b(depiction|representative|product|identity|indicative)\s+of\b|\btaken with\b",
     "the image as the subject of a copula / modal (whole-image reading)"),
    (r"\bas\s+(a|an|its|the)\s+(\w+\s+)?(subject|component|objective|object|focus)\b", "adds a role for the object"),
    (r"\b(be|being)\s+(used|operated|interpreted|considered)\b", "asks about use / interpretation, not presence"),
    (r"\b(\w+)\s+\1\b", "repeated word"),
    (r"\b(resembles?|look like|looks like|take the form|made (out )?of|appearance of|figment|instance of|indications? of"
     r"|considered to be|in question is|image of)\b", "resemblance / whole-image reading"),
    (r"\bis the (object|subject|device|item|thing)\b", "identity question (is the object a ...)"),
    (r"\b(tell me about|information about|explain|describe|let me know|talk about)\b",
     "a request to describe (not the question)"),
    (r"\bsuch a thing\b|\bexists?\b|\bexistence\b", "existential reading (does X exist)"),
    (r"[()]", "parenthetical (odd reading)"),
    (r"\b(what is|what's)\s+(a|an|that|this)\s+(\w+\s+)?subject\b",
     "indefinite or demonstrative 'subject' (a subject, not THE main subject)"),
    (r"\bthat\s+image\b", "distal demonstrative 'that image'"),
    (r"\b(controlled|control|use of|using|captured with|working|attached|vicinity|come with|comes with|accessible"
     r"|available|as a feature|as an indication|as well|what is it|if so|identify this image)\b",
     "adds a use, property or second question"),
)


PRESENCE_VERBS = {"show", "contain", "include", "have", "has", "depict", "display", "feature", "portray", "showcase",
                  "present", "exhibit", "reveal", "capture", "illustrate"}
PERCEPTION_VERBS = {"see", "find", "spot", "observe", "detect", "identify", "notice", "locate", "tell", "confirm", "verify",
                    "determine"}


def _lemma_v(w: str) -> str:
    w = w.lower()
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ed", "")):
        if w.endswith(suf) and (w[: -len(suf)] + rep) in PRESENCE_VERBS | PERCEPTION_VERBS:
            return w[: -len(suf)] + rep
    return w


def extra_rule(text: str, q: str, prot: list[str], nlp) -> str | None:
    low = text.lower()
    for pat, why in EXTRA_RULES:
        if re.search(pat, low):
            return why
    for o in (p for p in prot if p not in B.GENERIC_PROTECTED):
        obj = re.escape(o.split()[0])
        m = re.search(r"\bimage\s+(\w+)\s+(?:a|an|any|some)?\s*" + obj + r"\b", low)
        if m and _lemma_v(m.group(1)) not in PRESENCE_VERBS:
            return f"'image {m.group(1)} <object>': not a presence verb"
        m = re.search(r"\b(?:can|could)\s+(?:you|we|i)\s+(\w+)\s+(?:a|an|any|some)?\s*" + obj + r"\b", low)
        if m and _lemma_v(m.group(1)) not in PERCEPTION_VERBS:
            return f"'can you {m.group(1)} <object>': not a perception verb"
    for o in (p for p in prot if p not in B.GENERIC_PROTECTED):
        n = lambda s: len(re.findall(r"\b" + re.escape(o.split()[-1]) + r"\w*", s.lower()))
        if n(text) > n(q):
            return "object word repeated (self-definition)"
        if re.search(r"\b" + re.escape(o) + r"-", low):
            return "adds content about the object (hyphen compound: 'broccoli-like', 'remote-controlled')"
        doc = nlp(text)
        for i, tk in enumerate(doc):
            if tk.lower_ == o.split()[0] and i > 0 and (doc[i - 1].tag_ in ("JJ", "JJR", "JJS", "NN", "NNS", "VBN", "HYPH")) \
                    and doc[i - 1].lower_ not in o.split():
                return "modifier before the object ('lone dog', 'actual person')"
        for i, tk in enumerate(doc[:-1]):
            if tk.lower_ == o.split()[-1] and (doc[i + 1].text in ("'s", "’s") or doc[i + 1].tag_ in ("NN", "NNS", "NNP")
                                                or doc[i + 1].lower_ in ("which", "that", "who")):
                return "adds content about the object (possessive, compound noun or relative clause after it)"
    return None


def key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip().rstrip("?").strip()


def articles_ok(text: str) -> bool:
    w = re.findall(r"[A-Za-z']+", text)
    return all(S._article_ok(a, b) for a, b in zip(w, w[1:]))


def t5_extra(tok, mod, q: str, bank_texts: list[str], dev) -> list[str]:
    def gen(src, n_beam, n_samp, seeds, temps):
        enc = tok(f"paraphrase: {src}", return_tensors="pt", truncation=True, max_length=128).to(dev)
        outs = []
        with torch.no_grad():
            if n_beam:
                outs += tok.batch_decode(mod.generate(**enc, num_beams=n_beam, num_return_sequences=n_beam,
                                                      repetition_penalty=1.5, max_length=64), skip_special_tokens=True)
            for s in seeds:
                for tp in temps:
                    torch.manual_seed(s)
                    outs += tok.batch_decode(mod.generate(**enc, do_sample=True, top_p=0.95, temperature=tp,
                                                          num_return_sequences=n_samp, max_length=64), skip_special_tokens=True)
        return outs
    outs = gen(q, 32, 256, (1, 2), (1.0, 1.3))
    for p in bank_texts:
        outs += gen(p, 16, 64, (0,), (1.0,))
    return [re.sub(r"\s+", " ", s).strip() for s in outs if s.strip()]


def t5_more(tok, mod, q: str, paras: list[str], dev) -> list[str]:
    """Growth: more seeds and temperatures for the original, second-order paraphrases of every accepted paraphrase."""
    def gen(src, n_beam, n_samp, seeds, temps):
        enc = tok(f"paraphrase: {src}", return_tensors="pt", truncation=True, max_length=128).to(dev)
        outs = []
        with torch.no_grad():
            if n_beam:
                outs += tok.batch_decode(mod.generate(**enc, num_beams=n_beam, num_return_sequences=n_beam,
                                                      repetition_penalty=1.5, max_length=64), skip_special_tokens=True)
            for s_ in seeds:
                for tp in temps:
                    torch.manual_seed(s_)
                    outs += tok.batch_decode(mod.generate(**enc, do_sample=True, top_p=0.95, temperature=tp,
                                                          num_return_sequences=n_samp, max_length=64), skip_special_tokens=True)
        return outs
    outs = gen(q, 0, 256, (3, 4, 5), (1.0, 1.3, 1.6))
    for p in paras:
        outs += gen(p, 16, 64, (1,), (1.0, 1.3))
    return [re.sub(r"\s+", " ", s).strip() for s in outs if s.strip()]


def substitutions(op, nlp, q: str, prot: list[str], max_combos: int, rng, max_sites: int = 3):
    toks = tokenize(nlp, q)
    excl = frozenset(prot) | frozenset(w for p in prot for w in p.split())
    ctx = op.prepare(toks, exclude_words=excl)
    sites = [(int(p), ow, list(c)) for p, ow, c in zip(ctx.positions, ctx.original_words, ctx.candidates)]
    combos = []
    for k in range(1, max_sites + 1):
        for ss in itertools.combinations(range(len(sites)), k):
            for choice in itertools.product(*[sites[i][2] for i in ss]):
                combos.append(tuple(zip(ss, choice)))
    if len(combos) > max_combos:
        idx = np.sort(rng.choice(len(combos), max_combos, replace=False))
        combos = [combos[i] for i in idx]
    out = []
    for c in combos:
        t = list(toks.tokens)
        for i, w in c:
            t[sites[i][0]] = S._match_case(sites[i][1], w)
        out.append(("".join(a + b for a, b in zip(t, toks.whitespace)), [[sites[i][1], w] for i, w in c]))
    return out, {ow: c for _, ow, c in sites}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("configs/text/question_pool.json"))
    ap.add_argument("--bank", type=Path, default=Path("configs/text/paraphrase_bank.json"))
    ap.add_argument("--max-combos", type=int, default=20000)
    ap.add_argument("--max-combos-paraphrase", type=int, default=300)
    ap.add_argument("--grow-below", type=int, default=10**9,
                    help="grow pools with fewer accepted entries (step 4); default: grow every question (same construction for all)")
    ap.add_argument("--d-max-quantile", type=float, default=0.95)
    ap.add_argument("--n-bands", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    import spacy
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    rng = np.random.default_rng(0)
    bank = json.loads(a.bank.read_text())
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])   # tagger kept (extra_rule)
    op = S.SemanticSubstitutionOperator(ladder=tuple(range(1, 51)), device=a.device)
    op._ensure_semantic_loaded()
    words = S._wordlist() or frozenset()
    tok = AutoTokenizer.from_pretrained(B.GEN); mod = AutoModelForSeq2SeqLM.from_pretrained(B.GEN).to(a.device).eval()
    questions = {}

    def validate(q, prot, cands, a0, thr):
        rejected, pre = [], []
        for c in cands:
            t, low = c["text"], c["text"].lower()
            why = None
            if not t.endswith("?"):
                why = "not a question"
            elif not all(re.search(r"\b" + re.escape(p) + r"\b", low) for p in prot):
                why = "protected word lost"
            elif any(w.lower() not in words and w.lower() not in prot and w not in words for w in re.findall(r"[A-Za-z]+", t)):
                why = "non-dictionary word"
            elif not articles_ok(t):
                why = "a/an disagreement"
            else:
                why = B.meaning_rule(t, prot) or extra_rule(t, q, prot, nlp)
            if why:
                rejected.append(dict(c, reason=why))
            else:
                pre.append(c)
        texts = [c["text"] for c in pre]
        ent = op.mutual_entailment(q, texts) if texts else np.empty(0)
        acc = op.acceptability(texts) if texts else np.empty(0)
        ok = []
        for c, e, ac in zip(pre, ent, acc):
            if e < NLI_MIN:
                rejected.append(dict(c, reason=f"NLI mutual entailment {e:.2f} < {NLI_MIN}"))
            elif ac < thr:
                rejected.append(dict(c, reason=f"CoLA {ac:.2f} < {thr:.2f}"))
            else:
                ok.append(dict(c, nli=round(float(e), 3), cola=round(float(ac), 3)))
        d = op._distances(q, [c["text"] for c in ok]) if ok else []
        for c, x in zip(ok, d):
            c["dist"] = round(float(x), 4)
        return ok, rejected

    def merge(cands, q, src, items):
        new = []
        for t, ed in items:
            k = key(t)
            if k == key(q):
                continue
            if k in cands:
                if src not in cands[k]["sources"]:
                    cands[k]["sources"].append(src)
                continue
            cands[k] = dict(text=t, sources=[src], **({"edits": ed} if ed else {}))
            new.append(cands[k])
        return new

    state = {}
    for q, prot in prompts().items():          # pass 1: substitutions, bank, T5
        a0 = float(op.acceptability([q])[0]); thr = max(min(0.5, a0), 0.9 * a0)
        subs, sites = substitutions(op, nlp, q, prot, a.max_combos, rng)
        btexts = [x["text"] for x in bank.get(q, {}).get("paraphrases", [])]
        cands: dict[str, dict] = {}                 # key -> {text, sources, edits}
        for src, items in (("bank", [(t, None) for t in btexts]), ("substitution", subs),
                           ("t5", [(t, None) for t in t5_extra(tok, mod, q, btexts, a.device)])):
            merge(cands, q, src, items)
        ok, rejected = validate(q, prot, list(cands.values()), a0, thr)
        state[q] = dict(prot=prot, a0=a0, thr=thr, sites=sites, cands=cands, ok=ok, rejected=rejected, grown=False)
        print(f"pass 1 {q}: pool {len(ok)}", flush=True)
    # growth vocabulary: words of the originals and of every accepted pass-1 entry (all questions)
    vocab = {w.lower() for q, st in state.items() for t in [q] + [c["text"] for c in st["ok"]]
             for w in re.findall(r"[A-Za-z]+", t)}
    for q, st in state.items():                 # pass 2: growth
        if len(st["ok"]) >= a.grow_below:
            continue
        st["grown"] = True
        prot, cands = st["prot"], st["cands"]
        paras = [c["text"] for c in st["ok"] if c["sources"] != ["substitution"]]
        extra = []
        for p_ in paras:
            ps, _ = substitutions(op, nlp, p_, prot, a.max_combos_paraphrase, rng, max_sites=2)
            extra += merge(cands, q, "paraphrase+substitution", [(t, [p_] + ed) for t, ed in ps])
        extra += merge(cands, q, "t5", [(t, None) for t in t5_more(tok, mod, q, paras, a.device)])
        known, novel = [], []
        for c in extra:
            new = sorted({w.lower() for w in re.findall(r"[A-Za-z]+", c["text"])} - vocab - set(prot))
            (novel if new else known).append(c)
            if new:
                c["reason"] = "growth: word(s) outside the accepted pass-1 vocabulary: " + ", ".join(new[:4])
        ok2, rej2 = validate(q, prot, known, st["a0"], st["thr"])
        st["ok"] = st["ok"] + ok2
        st["rejected"] = st["rejected"] + rej2 + novel
    for q, st in state.items():
        ok, rejected = st["ok"], st["rejected"]
        ok.sort(key=lambda c: (c["dist"], c["text"]))
        counts: dict[str, int] = {}
        for r in rejected:
            rr = r["reason"]
            if rr.startswith(("NLI", "CoLA")):
                rr = rr.split(" 0.")[0].split(" 1.")[0]
            rr = rr.split(":")[0] if rr.startswith(("growth", "'image", "'can you")) else rr
            kk = f"{r['sources'][0]}: {rr}"
            counts[kk] = counts.get(kk, 0) + 1
        questions[q] = dict(protect=st["prot"], cola_original=round(st["a0"], 3), cola_threshold=round(st["thr"], 3),
                            grown=st["grown"], substitution_sites={k: v for k, v in st["sites"].items()},
                            n_candidates=len(st["cands"]), n_pool=len(ok),
                            n_by_source={s_: sum(s_ in c["sources"] for c in ok)
                                         for s_ in ("substitution", "bank", "t5", "paraphrase+substitution")},
                            ceiling=ok[-1]["dist"] if ok else 0.0,
                            rejected_counts=dict(sorted(counts.items())), pool=ok,
                            rejected=sorted(rejected, key=lambda r: (r["reason"], r["text"])))
        print(f"{q}: candidates {len(st['cands'])}, pool {len(ok)} {questions[q]['n_by_source']} ceiling {questions[q]['ceiling']}"
              + (" (grown)" if st["grown"] else ""), flush=True)
    alld = np.array([c["dist"] for e in questions.values() for c in e["pool"]])
    d_max = round(float(np.quantile(alld, a.d_max_quantile)), 4)
    meta = dict(d_max=d_max, d_max_quantile=a.d_max_quantile, n_bands=a.n_bands, nli_min=NLI_MIN,
                distance="1 - cos(all-MiniLM-L6-v2)", cells=list(CELL_GLOBS), max_combos=a.max_combos,
                generator=B.GEN, grow_below=a.grow_below, extra_rules=[w for _, w in EXTRA_RULES] + [
                "'image VERB <object>' with VERB outside " + "/".join(sorted(PRESENCE_VERBS)),
                "'can you VERB <object>' with VERB outside " + "/".join(sorted(PERCEPTION_VERBS)),"object word repeated (self-definition)",
                "adds content about the object (hyphen compound: 'broccoli-like', 'remote-controlled')",
                "modifier before the object ('lone dog', 'actual person')",
                "adds content about the object (possessive, compound noun or relative clause after it)"], note="mapping: mmb/manipulator/text/operators/question_pool.py select_index")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(meta=meta, questions=questions), indent=1, ensure_ascii=False))
    print("d_max", d_max, "pool total", alld.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
