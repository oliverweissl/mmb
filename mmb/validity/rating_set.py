"""Human validity study on MAIN crossings: item sampler and a static rating page.

Usage::

    python -m mmb.validity.rating_set results/MAIN.csv --out study/main_validity \
        [--n 120] [--controls 24] [--arms a0_directed,a1_random_tiered,a8_hillclimb] \
        [--agree results/agree_MAIN_qwen.csv ...] [--seed 42] [--runs-root runs/MAIN]

writes into --out:
  rate.html        one self-contained page (manifest inlined); open it in any browser, no server
  items/*.png      the stimuli (blind ids); first-crossing images, plus unedited anchors as controls
  manifest.json    what the page shows (no arm, SUT or outcome)
  key.csv          PRIVATE key: item_id -> task, sut, arm, cell_id, rng_seed, stratum, run_dir, ground truth
Hand rate.html together with the items/ folder to raters; they download a ratings CSV at the end and send it
back. Nothing is uploaded, no cookies, no analytics; the page keeps a draft in the browser's localStorage so a
reload does not lose answers, and "Download" works at any time.

Protocol (the HS-01 pair phase, adapted to the two MAIN tasks):
  imagenet: image + the (edited) question + five options: the pair's two class words (display order randomised
            per rater and item), "Something else", "Nothing recognizable", "I can't tell"; valid = the
            anchor class word.
  pope:     "Does the image contain a <object>?" (Yes / No / I can't tell) and "Does the question below still
            ask whether the image contains a <object>?" (Yes / No / I can't tell); valid = the answer to
            the first equals the ground truth and the second is Yes.
  all:      5-point comprehensibility of the question shown.
Edited items are sampled evenly over (task, arm), spread over SUTs and cells; controls are the unedited
anchors (resized to the stimulus size) of other cells and are rated in the same page, in random order.

The consent text lives in manifest["consent"]; the default below follows the properties recorded for HS-01
(anonymous, voluntary, no names or contact data, can stop at any time) and MUST be replaced by the verbatim
HS-01 consent text before any rater sees it (--consent-file).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

DEFAULT_ARMS = ("a0_directed", "a1_random_tiered", "a8_hillclimb")
DEFAULT_CONSENT = (
    "CONSENT TEXT PLACEHOLDER - replace with the verbatim HS-01 consent text before use (--consent-file).\n"
    "This study asks how well people understand images and short questions that are shown to an AI system. "
    "Taking part is voluntary and anonymous: no name, e-mail address or other identifying data is collected, "
    "and you may stop at any time by closing the page. Your answers are saved as a file on your own computer "
    "that you send back yourself. By ticking the box you confirm that you are at least 18 years old and agree to take part."
)


def _stimulus_size(df: pd.DataFrame) -> tuple[int, int]:
    for d in df["run_dir"]:
        p = Path(d) / "first_cross.png"
        if p.exists():
            return Image.open(p).size
    return (256, 256)


def _load_run(run_dir: Path):
    stats = json.loads((run_dir / "stats.json").read_text())
    fc = json.loads((run_dir / "first_cross.json").read_text()) if (run_dir / "first_cross.json").exists() else {}
    return stats, fc


def sample_edited(df: pd.DataFrame, n: int, arms, seed: int, verified: set[str] | None = None) -> pd.DataFrame:
    """n crossing runs, even over (task, arm), round-robin over SUTs, random cells within; one per (cell, sut, arm)."""
    rng = random.Random(seed)
    d = df[df["crossed"].astype(bool) & df["arm"].isin(arms)].copy()
    if verified is not None:
        d = d[d["run_dir"].isin(verified)]
    d = d.sort_values(["first_cross_eval", "run_dir"]).drop_duplicates(["task", "sut", "arm", "cell_id"])
    strata = sorted({(t, a) for t, a in zip(d["task"], d["arm"])})
    if not strata:
        return d.iloc[0:0]
    per = max(1, n // len(strata))
    picks = []
    for t, a in strata:
        g = d[(d["task"] == t) & (d["arm"] == a)]
        by_sut = {s: list(x.index) for s, x in g.groupby("sut")}
        for idxs in by_sut.values():
            rng.shuffle(idxs)
        chosen, order = [], sorted(by_sut)
        while len(chosen) < per and any(by_sut[s] for s in order):
            for s in order:
                if by_sut[s] and len(chosen) < per:
                    chosen.append(by_sut[s].pop())
        picks.append(g.loc[chosen])
    return pd.concat(picks)


def build(results: Path, out: Path, *, n: int = 120, controls: int = 24, arms=DEFAULT_ARMS, seed: int = 42,
          agree: list[Path] | None = None, runs_root: Path | None = None, consent: str = DEFAULT_CONSENT) -> dict:
    df = pd.read_csv(results)
    if runs_root is not None:
        df["run_dir"] = [str(p) if Path(p).exists() else str(Path(runs_root) / p) for p in df["run_dir"]]
    verified = None
    if agree:
        a = pd.concat([pd.read_csv(p) for p in agree])
        ok = a["cross_verified"].astype(str).str.lower().isin(("true", "1", "1.0"))
        verified = set(a.loc[ok, "run_dir"])
    size = _stimulus_size(df)
    edited = sample_edited(df, n, arms, seed, verified)
    rng = random.Random(seed + 1)
    # controls: anchors of cells not among the edited ones when possible, even over tasks
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
        ctrl_rows += [g.loc[i] for i in (free + rest)[:need]]       # prefer cells that have no edited item
    ids = [f"it{k:04d}" for k in rng.sample(range(1, 9999), len(edited) + len(ctrl_rows))]
    out = Path(out)
    (out / "items").mkdir(parents=True, exist_ok=True)
    items, key = [], []

    def item_from(run_dir: Path, item_id: str, stratum: str, row) -> None:
        stats, fc = _load_run(run_dir)
        meta = stats.get("seed_metadata") or {}
        task = row["task"]
        ca, cb = stats.get("class_a"), stats.get("class_b")
        q0 = meta.get("prompt_template") or stats.get("prompt_template") or ""
        if stratum == "edited":
            src, text = run_dir / "first_cross.png", fc.get("text") or q0
        else:
            src, text = run_dir / "origin.png", q0
        img = Image.open(src).convert("RGB")
        if img.size != size:
            img = img.resize(size, Image.BICUBIC)
        img.save(out / "items" / f"{item_id}.png")
        it = {"id": item_id, "type": "pope" if task == "pope" else "pair", "image": f"items/{item_id}.png", "question": text}
        if it["type"] == "pair":
            it["options"] = [ca, cb]
        else:
            it["object"] = meta.get("object") or "object"
        items.append(it)
        key.append(dict(item_id=item_id, stratum=stratum, task=task, sut=row["sut"], arm=row["arm"] if stratum == "edited" else "",
                        cell_id=row["cell_id"], rng_seed=row["rng_seed"], run_dir=str(run_dir),
                        anchor_label=ca if task != "pope" else "", gt=(meta.get("gt") or "") if task == "pope" else "",
                        object=meta.get("object") or "", original_question=q0, shown_question=text,
                        g_cross=fc.get("g", "") if stratum == "edited" else ""))

    k = 0
    for _, row in edited.iterrows():
        item_from(Path(row["run_dir"]), ids[k], "edited", row); k += 1
    for row in ctrl_rows:
        item_from(Path(row["run_dir"]), ids[k], "control", row); k += 1
    rng.shuffle(items)
    manifest = {"version": 1, "n_items": len(items), "consent": consent, "items": items}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    (out / "rate.html").write_text(RATING_HTML.replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")))
    return {"edited": len(edited), "controls": len(ctrl_rows), "out": str(out)}


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
.small{color:#555;font-size:.9em}
progress{width:100%}
</style></head><body>
<div id="app"></div>
<script>
"use strict";
var M = __MANIFEST__;
var KEY = "ratings_draft_v1";
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
  app.appendChild(el("div",{"class":"q",text:it.question}));
  var order="";var q1,q2=null;
  if(it.type==="pair"){
    var two=shuffle(it.options,r);order=two.join("|");
    app.appendChild(el("p",{text:"What is the main subject of the image?"}));
    q1=radios("a",[{value:two[0],label:two[0]},{value:two[1],label:two[1]},{value:"__else",label:"Something else"},{value:"__nothing",label:"Nothing recognizable"},{value:"__cant",label:"I can't tell"}]);
  }else{
    app.appendChild(el("p",{text:"Does the image contain a "+it.object+"?"}));
    q1=radios("a",[{value:"yes",label:"Yes"},{value:"no",label:"No"},{value:"__cant",label:"I can't tell"}]);
    q2=radios("b",[{value:"yes",label:"Yes"},{value:"no",label:"No"},{value:"__cant",label:"I can't tell"}]);
  }
  app.appendChild(q1);
  if(q2){app.appendChild(el("p",{text:"Does the question above still ask whether the image contains a "+it.object+"?"}));app.appendChild(q2);}
  app.appendChild(el("p",{text:"How easy is the question above to understand? (1 = not at all, 5 = perfectly)"}));
  var q3=radios("c",[1,2,3,4,5].map(function(v){return{value:String(v),label:String(v)};}));app.appendChild(q3);
  var next=el("button",{text:"Next"});next.onclick=function(){
    var a=chosen("a"),b=q2?chosen("b"):"",c=chosen("c");
    if(a===null||c===null||(q2&&b===null)){alert("Please answer every question.");return;}
    state.rows.push({rater_id:state.rater,item_id:it.id,order_index:state.pos,answer:a,answer_question_still_asked:b,comprehensibility:c,shown_option_order:order,rt_ms:Date.now()-t0,client_time:new Date().toISOString()});
    state.pos++;save();show();};
  app.appendChild(next);
  app.appendChild(el("button",{text:"Download answers so far"})).onclick=download;
}
function csv(){var cols=["rater_id","item_id","order_index","answer","answer_question_still_asked","comprehensibility","shown_option_order","rt_ms","client_time"];
  function q(v){v=String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;}
  return [cols.join(",")].concat(state.rows.map(function(r){return cols.map(function(c){return q(r[c]);}).join(",");})).join("\n");}
function download(){var b=new Blob([csv()],{type:"text/csv"});var a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="ratings_"+state.rater+".csv";document.body.appendChild(a);a.click();a.remove();}
function done(){app.innerHTML="";app.appendChild(el("h2",{text:"Done - thank you"}));
  app.appendChild(el("p",{text:"Please download your answers and send the file back to the person who gave you this page."}));
  app.appendChild(el("button",{text:"Download answers (CSV)"})).onclick=download;}
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
    ap.add_argument("--agree", type=Path, nargs="*", default=[], help="keep only single-item verified crossings")
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--consent-file", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    consent = a.consent_file.read_text() if a.consent_file else DEFAULT_CONSENT
    print(build(a.results, a.out, n=a.n, controls=a.controls, arms=a.arms.split(","), seed=a.seed,
                agree=a.agree or None, runs_root=a.runs_root, consent=consent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
