"""RQ3 human validity study: item sampler and one self-contained rating page.

Usage::

    python -m mmb.validity.rating_set results/MAIN.csv --out study/rq3 \
        [--n 120] [--controls 24] [--arms a0_directed,a1_random_tiered,a5_mimicry_objective,a9_deepjanus] \
        [--proxies results/validity_MAIN.csv] [--agree results/agree_MAIN_qwen.csv ...] [--seed 42]

Stimuli are the MINIMAL crossings of the runs (min_cross.png/.json, else the minimiser's minimal_cross.*), i.e.
the method's output. Edited items are spread evenly over the strata method x task x SUT (one per (seed input,
SUT, method) at most); a method without crossing runs in the results is skipped with a warning (re-sample once
its runs exist). Controls are codec reconstructions of unedited seed inputs (anchor_recon.png), shown with the
original question; they measure how often raters call an unchanged input invalid.

writes into --out:
  rate.html        the rater view: one self-contained file (images embedded as base64), nothing else needed
  rater_view.json  what the page shows (no method, SUT or outcome)
  items/*.png      the stimuli under their blind ids (for the analyst)
  manifest.csv     PRIVATE key: item_id, stratum, run_id, run_dir, method (arm), task, sut, seed input (cell_id,
                   rng_seed), candidate answers, ground truth, original and shown question, proxies (img_lpips,
                   vis_char_dist of the minimal crossing; 0 for controls). Keep it with the analyst.

Per item the page shows the image, the original question and the question shown to the model side by side, and asks
  (a) imagenet: "Does the image still show a <seed class>?"   pope: "Is there a <object> in the image?"
      (Yes / No / I can't tell);
  (b) "Is the question on the right still the same question as the original on the left?" (Yes / No / I can't tell).
Valid = (a) matches the seed (imagenet: Yes; pope: the ground truth) and (b) is Yes. Ratings download as JSON
(primary) or CSV. Nothing is uploaded; a draft is kept in the browser's localStorage.

The consent text lives in the page; the default below MUST be replaced by the verbatim study consent text
before any rater sees it (--consent-file).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image

from mmb.analysis.crossing_files import minimal_crossing

DEFAULT_ARMS = ("a0_directed", "a1_random_tiered", "a5_mimicry_objective", "a9_deepjanus")   # hill climber dropped 2026-11-25
DEFAULT_CONSENT = (
    "CONSENT TEXT PLACEHOLDER - replace with the verbatim consent text before use (--consent-file).\n"
    "This study asks how well people understand images and short questions that are shown to an AI system. "
    "Taking part is voluntary and anonymous: no name, e-mail address or other identifying data is collected, "
    "and you may stop at any time by closing the page. Your answers are saved as a file on your own computer "
    "that you send back yourself. By ticking the box you confirm that you are at least 18 years old and agree to take part."
)
PROXY_COLS = ("img_lpips", "vis_char_dist")


def _stimulus_size(df: pd.DataFrame) -> tuple[int, int]:
    for d in df["run_dir"]:
        mc = minimal_crossing(Path(d))
        if mc is not None:
            return Image.open(mc[0]).size
    return (256, 256)


def sample_edited(df: pd.DataFrame, n: int, arms, seed: int, verified: set[str] | None = None) -> pd.DataFrame:
    """n runs with a minimal crossing, spread as evenly as possible over the (arm, task, sut) strata (a stratum that
    runs short passes its quota on), random seed inputs within; one per (cell, sut, arm). Arms with no crossing run
    are skipped with a warning."""
    rng = random.Random(seed)
    d = df[df["crossed"].astype(bool) & df["arm"].isin(arms)].copy()
    if verified is not None:
        d = d[d["run_dir"].isin(verified)]
    d = d[[minimal_crossing(Path(p)) is not None for p in d["run_dir"]]]
    for a in arms:
        if a not in set(d["arm"]):
            warnings.warn(f"method {a}: no runs with a minimal crossing; skipped (re-sample once its runs exist)")
    d = d.sort_values(["min_cross_eval" if "min_cross_eval" in d else "first_cross_eval", "run_dir"])
    d = d.drop_duplicates(["task", "sut", "arm", "cell_id"])
    strata = sorted({(a, t, s) for a, t, s in zip(d["arm"], d["task"], d["sut"])})
    if not strata:
        return d.iloc[0:0]
    pools = {}
    for a, t, s in strata:
        idx = sorted(d.index[(d["arm"] == a) & (d["task"] == t) & (d["sut"] == s)])
        rng.shuffle(idx)
        pools[(a, t, s)] = idx
    # exactly n items when the runs allow it: n // strata each, the remainder to randomly ordered strata, and the
    # quota of a stratum that runs short goes to the strata that still have runs
    quota = {k: 0 for k in strata}
    order = list(strata)
    rng.shuffle(order)
    left = n
    while left > 0:
        open_ = [k for k in order if quota[k] < len(pools[k])]
        if not open_:
            break
        for k in sorted(open_, key=lambda k: quota[k]):
            if left == 0:
                break
            if quota[k] == min(quota[j] for j in open_):
                quota[k] += 1
                left -= 1
    return pd.concat([d.loc[pools[k][:quota[k]]] for k in strata])


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build(results: Path, out: Path, *, n: int = 120, controls: int = 24, arms=DEFAULT_ARMS, seed: int = 42,
          agree: list[Path] | None = None, runs_root: Path | None = None, consent: str = DEFAULT_CONSENT,
          proxies: Path | None = None) -> dict:
    df = pd.read_csv(results)
    if runs_root is not None:
        df["run_dir"] = [str(p) if Path(p).exists() else str(Path(runs_root) / p) for p in df["run_dir"]]
    verified = None
    if agree:
        a = pd.concat([pd.read_csv(p) for p in agree])
        ok = a["cross_verified"].astype(str).str.lower().isin(("true", "1", "1.0"))
        verified = set(a.loc[ok, "run_dir"])
    prox = {}
    if proxies is not None and Path(proxies).exists():
        p = pd.read_csv(proxies)
        if "view" in p:
            p = p[p["view"] == "min"]
        prox = {Path(r).name: {c: r2.get(c) for c in PROXY_COLS} for r, r2 in zip(p["run_dir"], p.to_dict("records"))}
    size = _stimulus_size(df)
    edited = sample_edited(df, n, arms, seed, verified)
    rng = random.Random(seed + 1)
    # controls: seed inputs not among the edited ones when possible, even over tasks
    used = set(zip(edited["task"], edited["cell_id"]))
    pool = df.drop_duplicates(["task", "cell_id"])
    pool = pool.assign(_used=[(t, c) in used for t, c in zip(pool["task"], pool["cell_id"])]).sort_values("_used")
    ctrl_rows = []
    tasks = sorted(set(pool["task"]))
    for t in tasks:
        g = pool[pool["task"] == t]
        idx = list(g.index)
        need = max(0, controls // max(1, len(tasks)))
        free = [i for i in idx if not g.loc[i, "_used"]]
        rest = [i for i in idx if g.loc[i, "_used"]]
        rng.shuffle(free)
        rng.shuffle(rest)
        ctrl_rows += [g.loc[i] for i in (free + rest)[:need]]       # prefer seed inputs that have no edited item
    ids = [f"it{k:04d}" for k in rng.sample(range(1, 9999), len(edited) + len(ctrl_rows))]
    out = Path(out)
    (out / "items").mkdir(parents=True, exist_ok=True)
    items, key = [], []

    def item_from(run_dir: Path, item_id: str, stratum: str, row) -> None:
        stats = json.loads((run_dir / "stats.json").read_text())
        meta = stats.get("seed_metadata") or {}
        task = row["task"]
        ca, cb = stats.get("class_a"), stats.get("class_b")
        q0 = meta.get("prompt_template") or stats.get("prompt_template") or ""
        if stratum == "edited":
            png, js = minimal_crossing(run_dir)
            mc = json.loads(js.read_text())
            img, text = Image.open(png).convert("RGB"), mc.get("text") or q0
            pv = prox.get(run_dir.name, {})
        else:   # controls are shown as the SUT sees them: the codec reconstruction, not the raw photograph
            from mmb.validity.score import anchor_reference
            img, text = anchor_reference(run_dir)[0], q0
            pv = {c: 0.0 for c in PROXY_COLS}
        if img.size != size:
            img = img.resize(size, Image.BICUBIC)
        img.save(out / "items" / f"{item_id}.png")
        it = {"id": item_id, "task": "pope" if task == "pope" else "imagenet", "image": f"items/{item_id}.png",
              "original": q0, "question": text}
        if task == "pope":
            it["object"] = meta.get("object") or "object"
        else:
            it["label"] = ca
        items.append(it)
        cands = [ca, cb] if task != "pope" else ["Yes", "No"]
        key.append(dict(item_id=item_id, stratum=stratum, run_id=run_dir.name, task=task, sut=row["sut"],
                        arm=row["arm"] if stratum == "edited" else "",
                        cell_id=row["cell_id"], rng_seed=row["rng_seed"], run_dir=str(run_dir),
                        candidates="|".join(str(c) for c in cands),
                        anchor_label=ca if task != "pope" else "", gt=(meta.get("gt") or "") if task == "pope" else "",
                        object=meta.get("object") or "", original_question=q0, shown_question=text,
                        **{c: pv.get(c, "") for c in PROXY_COLS}))

    k = 0
    for _, row in edited.iterrows():
        item_from(Path(row["run_dir"]), ids[k], "edited", row); k += 1
    for row in ctrl_rows:
        item_from(Path(row["run_dir"]), ids[k], "control", row); k += 1
    rng.shuffle(items)
    view = {"version": 2, "n_items": len(items), "consent": consent, "items": items}
    (out / "rater_view.json").write_text(json.dumps(view, indent=1, ensure_ascii=False))
    pd.DataFrame(key).to_csv(out / "manifest.csv", index=False)
    embedded = dict(view, items=[dict(i, image=_data_uri(out / i["image"])) for i in items])
    (out / "rate.html").write_text(RATING_HTML.replace("__MANIFEST__", json.dumps(embedded, ensure_ascii=False).replace("</", "<\\/")))
    counts = edited.groupby(["arm", "task", "sut"]).size().to_dict() if len(edited) else {}
    return {"edited": len(edited), "controls": len(ctrl_rows), "strata": {"/".join(k): v for k, v in counts.items()},
            "out": str(out)}


RATING_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Image and question rating</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:1.5em auto;padding:0 1em;line-height:1.4}
img{max-width:100%;max-height:420px;border:1px solid #888;display:block;margin:.5em 0}
fieldset{margin:.8em 0;border:1px solid #bbb}legend{font-weight:600}
label{display:block;margin:.2em 0;cursor:pointer}
button{font-size:1em;padding:.4em 1em;margin:.3em .3em .3em 0}
.q{background:#f3f3f3;padding:.5em .8em;border-radius:4px;font-size:1.05em}
.pair{display:flex;gap:1em}.pair>div{flex:1}
.small{color:#555;font-size:.9em}
progress{width:100%}
</style></head><body>
<div id="app"></div>
<script>
"use strict";
var M = __MANIFEST__;
var KEY = "ratings_draft_v2";
var state = JSON.parse(localStorage.getItem(KEY) || "null");
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function hash(s){var h=2166136261;for(var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return h>>>0;}
function shuffle(a,r){a=a.slice();for(var i=a.length-1;i>0;i--){var j=Math.floor(r()*(i+1));var t=a[i];a[i]=a[j];a[j]=t;}return a;}
function newRater(){var c="abcdefghjkmnpqrstuvwxyz23456789",s="";for(var i=0;i<8;i++)s+=c[Math.floor(Math.random()*c.length)];return s;}
function save(){localStorage.setItem(KEY,JSON.stringify(state));}
function el(tag,attrs,kids){var e=document.createElement(tag);for(var k in (attrs||{})){if(k==="text")e.textContent=attrs[k];else e.setAttribute(k,attrs[k]);}(kids||[]).forEach(function(c){e.appendChild(c);});return e;}
function radios(name,opts){var f=el("fieldset");opts.forEach(function(o){var l=el("label");var i=el("input",{type:"radio",name:name,value:o.value});l.appendChild(i);l.appendChild(document.createTextNode(" "+o.label));f.appendChild(l);});return f;}
function chosen(name){var r=document.querySelector("input[name="+name+"]:checked");return r?r.value:null;}
var app=document.getElementById("app");
function consent(){
  app.innerHTML="";
  app.appendChild(el("h2",{text:"Image and question rating"}));
  String(M.consent).split("\n").forEach(function(p){app.appendChild(el("p",{text:p}));});
  var cb=el("input",{type:"checkbox",id:"agree"});var l=el("label");l.appendChild(cb);l.appendChild(document.createTextNode(" I have read the above and agree to take part."));app.appendChild(l);
  var go=el("button",{text:"Start"});go.onclick=function(){if(!cb.checked){alert("Please tick the box to continue.");return;}
    var id=newRater();var r=mulberry(hash(id));state={rater:id,order:shuffle(M.items.map(function(x){return x.id;}),r),pos:0,rows:[]};save();show();};
  app.appendChild(go);
  app.appendChild(el("button",{text:"I do not agree"})).onclick=function(){app.innerHTML="<p>Thank you. You can close this page.</p>";};
}
function item(id){for(var i=0;i<M.items.length;i++)if(M.items[i].id===id)return M.items[i];}
function show(){
  if(state.pos>=state.order.length){return done();}
  var it=item(state.order[state.pos]);var t0=Date.now();
  var r=mulberry(hash(state.rater+it.id));
  app.innerHTML="";
  var p=el("progress",{max:state.order.length,value:state.pos});app.appendChild(p);
  app.appendChild(el("p",{"class":"small",text:"Item "+(state.pos+1)+" of "+state.order.length}));
  app.appendChild(el("img",{src:it.image,alt:"stimulus"}));
  var pair=el("div",{"class":"pair"});
  var left=el("div");left.appendChild(el("p",{"class":"small",text:"Original question"}));left.appendChild(el("div",{"class":"q",text:it.original}));
  var right=el("div");right.appendChild(el("p",{"class":"small",text:"Question shown"}));right.appendChild(el("div",{"class":"q",text:it.question}));
  pair.appendChild(left);pair.appendChild(right);app.appendChild(pair);
  var yn=[{value:"yes",label:"Yes"},{value:"no",label:"No"},{value:"__cant",label:"I can't tell"}];
  app.appendChild(el("p",{text:it.task==="pope"?"Is there a "+it.object+" in the image?":"Does the image still show a "+it.label+"?"}));
  var q1=radios("a",yn);app.appendChild(q1);
  app.appendChild(el("p",{text:"Is the question shown still the same question as the original?"}));
  var q2=radios("b",yn);app.appendChild(q2);
  var next=el("button",{text:"Next"});next.onclick=function(){
    var a=chosen("a"),b=chosen("b");
    if(a===null||b===null){alert("Please answer every question.");return;}
    state.rows.push({rater_id:state.rater,item_id:it.id,order_index:state.pos,answer_content:a,answer_same_question:b,rt_ms:Date.now()-t0,client_time:new Date().toISOString()});
    state.pos++;save();show();};
  app.appendChild(next);
  app.appendChild(el("button",{text:"Download answers so far"})).onclick=function(){download("json");};
}
var COLS=["rater_id","item_id","order_index","answer_content","answer_same_question","rt_ms","client_time"];
function csv(){
  function q(v){v=String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;}
  return [COLS.join(",")].concat(state.rows.map(function(r){return COLS.map(function(c){return q(r[c]);}).join(",");})).join("\n");}
function download(kind){
  var body=kind==="csv"?csv():JSON.stringify({rater_id:state.rater,version:M.version,rows:state.rows},null,1);
  var b=new Blob([body],{type:kind==="csv"?"text/csv":"application/json"});var a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="ratings_"+state.rater+"."+kind;document.body.appendChild(a);a.click();a.remove();}
function done(){app.innerHTML="";app.appendChild(el("h2",{text:"Done - thank you"}));
  app.appendChild(el("p",{text:"Please download your answers and send the file back to the person who gave you this page."}));
  app.appendChild(el("button",{text:"Download answers (JSON)"})).onclick=function(){download("json");};
  app.appendChild(el("button",{text:"Download answers (CSV)"})).onclick=function(){download("csv");};}
if(state&&state.order&&state.pos<state.order.length){
  app.appendChild(el("p",{text:"A draft from this browser was found (item "+(state.pos+1)+" of "+state.order.length+")."}));
  var c=el("button",{text:"Continue"});c.onclick=show;app.appendChild(c);
  var n=el("button",{text:"Start over"});n.onclick=function(){localStorage.removeItem(KEY);state=null;consent();};app.appendChild(n);
}else if(state&&state.order){done();}else{consent();}
</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--controls", type=int, default=24)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--proxies", type=Path, default=Path("results/validity_MAIN.csv"),
                    help="mmb.validity.score CSV; img_lpips and vis_char_dist of the minimal crossing go into manifest.csv")
    ap.add_argument("--agree", type=Path, nargs="*", default=[], help="keep only single-item verified crossings")
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--consent-file", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    consent = a.consent_file.read_text() if a.consent_file else DEFAULT_CONSENT
    print(build(a.results, a.out, n=a.n, controls=a.controls, arms=a.arms.split(","), seed=a.seed,
                agree=a.agree or None, runs_root=a.runs_root, consent=consent, proxies=a.proxies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
