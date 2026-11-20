
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const js = /<script>([\s\S]*)<\/script>/.exec(html)[1];
function N(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent="";this.onclick=null;this.checked=false;}
N.prototype.appendChild=function(c){this.children.push(c);return c;};
N.prototype.setAttribute=function(k,v){this.attrs[k]=v;if(k==="name")this.name=v;if(k==="value")this.value=v;if(k==="id")this.id=v;if(k==="type")this.type=v;};
N.prototype.remove=function(){};
N.prototype.click=function(){downloads.push(this.download);};
Object.defineProperty(N.prototype,"innerHTML",{set:function(v){this.children=[];},get:function(){return "";}});
const downloads=[]; let blobs=[]; const alerts=[];
const root=new N("div"); root.id="app";
function walk(n,f){f(n);n.children.forEach(c=>walk(c,f));}
const ctx={document:{createElement:t=>new N(t),createTextNode:t=>({textNode:t,children:[]}),getElementById:()=>root,
 querySelector:(sel)=>{const m=/name=(\w+)\]:checked/.exec(sel);let r=null;walk(root,n=>{if(n.name===m[1]&&n.checked)r=n;});return r;},
 body:new N("body")},
 localStorage:{d:{},getItem(k){return this.d[k]||null;},setItem(k,v){this.d[k]=v;},removeItem(k){delete this.d[k];}},
 alert:m=>alerts.push(m),Blob:function(p){blobs.push(p.join(""));},URL:{createObjectURL:()=>"blob:x"},Math,JSON,Date,String,Array,Object,RegExp,Number,console};
ctx.window=ctx; vm.createContext(ctx); vm.runInContext(js,ctx);
function buttons(){const b=[];walk(root,n=>{if(n.tag==="button")b.push(n);});return b;}
function byText(t){return buttons().find(b=>b.textContent===t);}
// refuse without ticking
byText("Start").onclick(); if(!alerts.length) throw new Error("started without consent");
walk(root,n=>{if(n.id==="agree")n.checked=true;}); byText("Start").onclick();
let steps=0;
while(byText("Next")){
  const seen={};
  walk(root,n=>{if(n.tag==="input"&&n.type==="radio"&&!seen[n.name]){seen[n.name]=1;n.checked=true;}});
  const before=ctx.localStorage.getItem("ratings_draft_v1");
  byText("Next").onclick(); steps++; if(steps>1000) throw new Error("no end");
}
byText("Download answers (CSV)").onclick();
process.stdout.write(JSON.stringify({steps,csv:blobs[blobs.length-1],downloads,alerts}));
