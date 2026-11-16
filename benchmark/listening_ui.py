"""Escucha ciega en HTML local autocontenido (sin servidor, sin red, sin dependencias).

Entrada: paquete generado por ``listening.prepare_from_indexes`` (``stimuli/``,
``blind_key.json``). Salida: carpeta APARTE con ``index.html`` + ``stimuli/`` (copia
con rutas relativas) + ``INSTRUCCIONES.md`` + ``listening_ui_manifest.json``.

Ciego: el HTML y la carpeta de salida NO contienen nombres de sistema/modelo, ids de
ítem, nombres de archivo originales, textos de referencia ni ``blind_key.json``. Solo
``S####``/``R####``, el ``reference_id`` y qué dimensiones son ``NA`` fijas (misma
información que la plantilla CSV que ya recibe el evaluador). ``build`` verifica la
fuga buscando esas cadenas en cada archivo de salida y falla si aparece alguna.

Aleatorización por evaluador: el navegador baraja con ``mulberry32(fnv1a("seed:rater"))``;
:func:`rater_order` es la misma función en Python (reproducible y probada contra node).

No fabrica puntuaciones: todo arranca vacío; ``NA`` solo donde ``blind_key`` lo fija o si
el evaluador lo elige. Puntuación humana 1-5, no MOS formal (ITU-T P.800).
Formato de exportación ``opentraductor-listening-ratings/v1`` (JSON) y CSV con las
mismas columnas que ``ratings_template.csv``; ambos los lee ``listening.summarize_ratings``.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Sequence

from . import listening as L

SCHEMA = "opentraductor-listening-ratings/v1"
_BLIND_NAME = re.compile(r"^[SR]\d{4}\.wav$")


# ---- aleatorización reproducible (espejo exacto del JS) -----------------------------
def _fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def rater_order(ids: Sequence[str], seed: int, rater: str) -> list[str]:
    """Orden de presentación para ``rater`` (Fisher-Yates con PRNG sembrado). Idéntico al JS."""
    out = list(ids)
    rnd = _mulberry32(_fnv1a(f"{seed}:{rater.strip()}"))
    for i in range(len(out) - 1, 0, -1):
        j = int(rnd() * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out


# ---- HTML ---------------------------------------------------------------------------
_CSS = """
:root{--fg:#111;--bg:#fff;--muted:#444;--accent:#0b57d0;--ok:#146c2e;--warn:#8a4b00;--line:#767676}
@media (prefers-color-scheme:dark){:root{--fg:#f2f2f2;--bg:#121212;--muted:#c4c4c4;--accent:#8ab4f8;--ok:#7bd88f;--warn:#ffb74d;--line:#9a9a9a}}
body{font:1.1rem/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg);max-width:46rem;margin:0 auto;padding:1rem}
a.skip{position:absolute;left:-999px}a.skip:focus{left:1rem;top:1rem;background:var(--bg);padding:.5rem;z-index:9}
:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
button,input[type=text]{font:inherit;padding:.5rem .9rem;color:var(--fg);background:var(--bg);border:2px solid var(--line);border-radius:.4rem}
button{cursor:pointer}button.primary{border-color:var(--accent);font-weight:600}button:disabled{opacity:.5;cursor:not-allowed}
fieldset{border:2px solid var(--line);border-radius:.5rem;margin:.8rem 0;padding:.6rem .9rem}
legend{font-weight:600;padding:0 .4rem}
.opts{display:flex;flex-wrap:wrap;gap:.4rem 1.1rem}.opts label{display:inline-flex;align-items:center;gap:.35rem;min-height:2.4rem;cursor:pointer}
.opts input{width:1.4rem;height:1.4rem}
.hint{color:var(--muted);font-size:.95rem;margin:.1rem 0 .4rem}
.fixed{color:var(--muted);font-style:italic}
audio{width:100%;margin:.3rem 0}
progress{width:100%;height:1.2rem}
nav.row,.row{display:flex;flex-wrap:wrap;gap:.6rem;margin:.8rem 0}
#status{min-height:1.5rem;font-weight:600}.warn{color:var(--warn)}.ok{color:var(--ok)}
textarea{width:100%;font:inherit;min-height:4rem;color:var(--fg);background:var(--bg);border:2px solid var(--line);border-radius:.4rem;box-sizing:border-box}
"""

_JS = r"""
"use strict";
const D=JSON.parse(document.getElementById("data").textContent);
const DIMS=D.dimensions, $=s=>document.querySelector(s);
const NAMES={naturalidad:"Naturalidad",inteligibilidad:"Inteligibilidad",acento:"Acento (cercanía al inglés estadounidense general)",expresividad:"Expresividad (intención y énfasis frente al original del ítem)",similitud_voz:"Similitud de voz con la referencia (original o muestra de voz)"};
const REFLABEL={source_utterance:["Referencia: original del ítem (audio fuente en español). Escuchalo primero:","Audio original del ítem"],
 speaker_sample:["Muestra de voz (NO es el original de lo que dice el estímulo; idioma no verificado). Úsala solo para la similitud de voz:","Muestra de voz de referencia"],
 unknown:["Audio de referencia (tipo no declarado; ver instrucciones):","Audio de referencia"]};
const LOW="1 = muy mala", HIGH="5 = excelente";
function fnv1a(s){let h=0x811c9dc5;for(const b of new TextEncoder().encode(s)){h^=b;h=Math.imul(h,0x01000193)>>>0}return h>>>0}
function mulberry32(a){return function(){a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function order(ids,seed,rater){const o=ids.slice(),r=mulberry32(fnv1a(seed+":"+rater.trim()));for(let i=o.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[o[i],o[j]]=[o[j],o[i]]}return o}
let rater="",ord=[],pos=0,R={};
const storeKey=()=>"opentraductor-listening:v2:"+D.package_id+":"+rater.trim();
function save(){try{localStorage.setItem(storeKey(),JSON.stringify({package_id:D.package_id,ratings:R}))}catch(e){}}
function load(){try{const v=JSON.parse(localStorage.getItem(storeKey())||"null");return(v&&v.package_id===D.package_id&&v.ratings)?v.ratings:{}}catch(e){return{}}}
function blank(sid){const na=D.stimuli[sid].na,r={reference_id:D.stimuli[sid].ref||"",comentario:"",escuchas_completas:0};for(const d of DIMS)r[d]=na.includes(d)?"NA":"";return r}
function say(msg,cls){const s=$("#status");s.textContent=msg;s.className=cls||""}
function isDone(sid){const r=R[sid];return DIMS.every(d=>r[d]!=="")}
function counts(){const t=ord.length;let c=0;for(const s of ord)if(isDone(s))c++;return[c,t]}
function start(){
  const v=$("#rater").value.trim();
  if(!v){say("Escribí tu nombre o código de evaluador.","warn");$("#rater").focus();return}
  rater=v;ord=order(Object.keys(D.stimuli).sort(),D.seed,rater);
  const saved=load();R={};for(const s of ord){R[s]=Object.assign(blank(s),saved[s]||{});for(const d of DIMS)if(D.stimuli[s].na.includes(d))R[s][d]="NA"}
  pos=0;$("#login").hidden=true;$("#main").hidden=false;render();$("#title").focus();
}
function render(){
  const sid=ord[pos],st=D.stimuli[sid],r=R[sid],[c,t]=counts();
  $("#title").textContent="Estímulo "+(pos+1)+" de "+t;
  $("#prog").value=c;$("#prog").max=t;$("#progtxt").textContent=c+" de "+t+" estímulos completos";
  const box=$("#stim");box.textContent="";
  if(st.ref){const L=REFLABEL[st.refkind]||REFLABEL.unknown;const p=document.createElement("p");p.textContent=L[0];box.append(p);
    const a=document.createElement("audio");a.controls=true;a.preload="none";a.src="stimuli/"+st.ref+".wav";a.setAttribute("aria-label",L[1]);box.append(a)}
  const p2=document.createElement("p");p2.textContent="Estímulo a evaluar:";box.append(p2);
  const a2=document.createElement("audio");a2.controls=true;a2.preload="auto";a2.src="stimuli/"+sid+".wav";a2.setAttribute("aria-label","Audio a evaluar");
  a2.addEventListener("ended",()=>{r.escuchas_completas++;save()});box.append(a2);
  const f=$("#dims");f.textContent="";
  for(const d of DIMS){
    const fs=document.createElement("fieldset"),lg=document.createElement("legend");lg.textContent=NAMES[d];fs.append(lg);
    const fixed=st.na.includes(d);
    if(fixed){const p=document.createElement("p");p.className="fixed";p.textContent="No aplica a este estímulo (valor NA fijo).";fs.append(p)}
    else{const h=document.createElement("p");h.className="hint";h.textContent=LOW+" · "+HIGH+" · NA si no se puede juzgar";fs.append(h);
      const o=document.createElement("div");o.className="opts";
      for(const v of ["1","2","3","4","5","NA"]){const l=document.createElement("label"),i=document.createElement("input");
        i.type="radio";i.name=sid+"_"+d;i.value=v;i.checked=r[d]===v;
        i.addEventListener("change",()=>{r[d]=v;save();upd()});l.append(i,document.createTextNode(v==="NA"?"No sé / NA":v));o.append(l)}
      fs.append(o)}
    f.append(fs)}
  $("#coment").value=r.comentario;
  $("#prev").disabled=pos===0;$("#next").disabled=pos===t-1;
}
function upd(){const[c,t]=counts();$("#prog").value=c;$("#progtxt").textContent=c+" de "+t+" estímulos completos";say("Guardado en este navegador.","ok")}
function go(n){pos=Math.max(0,Math.min(ord.length-1,pos+n));render();$("#title").focus();window.scrollTo(0,0)}
function rows(){return ord.map(s=>Object.assign({stimulus_id:s},R[s]))}
function exportJSON(){const[c,t]=counts();
  return JSON.stringify({schema:D.schema,rater:rater,package_id:D.package_id,package_seed:D.seed,scale:D.scale,exported_at:new Date().toISOString(),n_complete:c,n_total:t,
   disclaimer:"Puntuación humana subjetiva 1-5; no es MOS formal (ITU-T P.800). Celdas vacías = sin dato (no se rellenan).",
   ratings:rows().map(x=>({stimulus_id:x.stimulus_id,reference_id:x.reference_id,naturalidad:cell(x.naturalidad),inteligibilidad:cell(x.inteligibilidad),acento:cell(x.acento),expresividad:cell(x.expresividad),similitud_voz:cell(x.similitud_voz),comentario:x.comentario,escuchas_completas:x.escuchas_completas}))},null,2)}
function cell(v){return(v===""||v==="NA")?v:Number(v)}
function q(s){s=String(s);return/[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function exportCSV(){const h=["rater","stimulus_id","reference_id",...DIMS,"comentario"];
  return[h.join(",")].concat(rows().map(x=>[rater,x.stimulus_id,x.reference_id,...DIMS.map(d=>x[d]),x.comentario].map(q).join(","))).join("\n")+"\n"}
function dl(name,text,type){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;document.body.append(a);a.click();a.remove()}
function safe(){return rater.trim().replace(/[^A-Za-z0-9_-]+/g,"_")||"evaluador"}
function doExport(kind){const[c,t]=counts();
  if(c<t)say("Aviso: exportás con "+(t-c)+" estímulos incompletos; sus celdas quedan vacías (sin dato).","warn");else say("Exportado: "+t+" estímulos completos.","ok");
  if(kind==="json")dl("ratings_"+D.package_id.slice(0,8)+"_"+safe()+".json",exportJSON(),"application/json");else dl("ratings_"+D.package_id.slice(0,8)+"_"+safe()+".csv",exportCSV(),"text/csv")}
document.addEventListener("DOMContentLoaded",()=>{
  $("#start").addEventListener("click",start);
  $("#rater").addEventListener("keydown",e=>{if(e.key==="Enter")start()});
  $("#prev").addEventListener("click",()=>go(-1));$("#next").addEventListener("click",()=>go(1));
  $("#coment").addEventListener("input",e=>{R[ord[pos]].comentario=e.target.value;save()});
  $("#exp-json").addEventListener("click",()=>doExport("json"));$("#exp-csv").addEventListener("click",()=>doExport("csv"));
  $("#reset").addEventListener("click",()=>{if(confirm("¿Borrar tus puntuaciones guardadas en este navegador?")){try{localStorage.removeItem(storeKey())}catch(e){}R={};for(const s of ord)R[s]=blank(s);render();say("Puntuaciones borradas.","warn")}});
});
"""

_PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Escucha ciega</title><style>__CSS__</style></head><body>
<a class="skip" href="#main">Saltar al contenido</a>
<header><h1>Escucha ciega</h1></header>
<section id="login" aria-labelledby="h-ins">
<h2 id="h-ins">Instrucciones</h2>
__INSTRUCTIONS__
<p><label for="rater">Tu nombre o código de evaluador:</label><br><input id="rater" type="text" autocomplete="off" aria-describedby="rhelp">
<button id="start" class="primary" type="button">Empezar</button></p>
<p id="rhelp" class="hint">El orden de los estímulos depende de este nombre. Usá siempre el mismo para retomar donde dejaste.</p>
</section>
<main id="main" hidden>
<h2 id="title" tabindex="-1"></h2>
<progress id="prog" value="0" max="1" aria-labelledby="progtxt"></progress><div id="progtxt" class="hint"></div>
<div id="stim"></div>
<form id="dims" onsubmit="return false"></form>
<p><label for="coment">Comentario (opcional)</label><br><textarea id="coment"></textarea></p>
<nav class="row" aria-label="Navegación"><button id="prev" type="button">Anterior</button><button id="next" type="button" class="primary">Siguiente</button></nav>
<div class="row"><button id="exp-json" type="button">Exportar JSON</button><button id="exp-csv" type="button">Exportar CSV</button><button id="reset" type="button">Borrar mis puntuaciones</button></div>
<p class="hint">Tus puntuaciones se guardan solo en este navegador (localStorage) y no salen de tu equipo hasta que exportes y envíes el archivo.</p>
</main>
<div id="status" role="status" aria-live="polite"></div>
<script type="application/json" id="data">__DATA__</script>
<script>__JS__</script>
</body></html>
"""


def _md_to_html(md: str) -> str:
    """Conversión mínima (títulos omitidos, listas, párrafos, `código`) del texto de instrucciones."""
    esc = lambda s: (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    code = lambda s: re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(s))
    out, para, items = [], [], []

    def flush():
        nonlocal para, items
        if para:
            out.append("<p>" + code(" ".join(para)) + "</p>")
            para = []
        if items:
            out.append("<ul>" + "".join(f"<li>{code(i)}</li>" for i in items) + "</ul>")
            items = []

    for line in md.splitlines():
        if line.startswith("## "):
            flush()
            out.append(f"<h3>{code(line[3:].strip())}</h3>")
            continue
        if line.startswith("#"):
            continue
        if line.startswith("- "):
            if para:
                flush()
            items.append(line[2:].strip())
        elif line.startswith("  ") and items:
            items[-1] += " " + line.strip()
        elif not line.strip():
            flush()
        else:
            if items:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


UI_NOTES = """
Esta página no necesita internet ni servidor: abrila con doble clic (`index.html`). Cada dimensión se
puntúa con botones 1 a 5 (no hay valor inicial: nada está puntuado hasta que elijas) y `NA` si no se
puede juzgar. Podés usar solo el teclado (Tab y flechas). Al terminar, `Exportar JSON` o `Exportar CSV`
y enviá el archivo. Si exportás antes de terminar, lo no puntuado queda vacío y no se rellena.
"""


def _leak_terms(key: dict) -> set[str]:
    terms: set[str] = set()
    for v in key.values():
        for f in ("system", "item_id", "audio_ref"):
            val = v.get(f)
            if val and len(val) >= 3:
                terms.add(val)
        ar = v.get("audio_ref")
        if ar:
            terms.add(Path(ar).stem)
    return terms


def package_id(seed: int, stimuli: dict, hashes: dict) -> str:
    """Identidad estable del paquete: semilla + dimensiones + escala + por estímulo (id, referencia, tipo, NA) + sha256 de cada WAV.
    Dos paquetes con la misma semilla e ids S#### pero audios o referencias distintas NO comparten identidad;
    regenerar el mismo contenido da el mismo id (el avance guardado sigue valiendo)."""
    canon = json.dumps({"seed": seed, "dimensions": list(L.DIMENSIONS), "scale": L.SCALE, "stimuli": stimuli, "sha256": hashes},
                       sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:20]


def build(
    package_dir: str | Path,
    out_dir: str | Path,
    *,
    seed: int | None = None,
) -> dict:
    """Genera el HTML de escucha ciega desde un paquete de ``listening-prepare``."""
    pkg, out = Path(package_dir), Path(out_dir)
    meta = json.loads((pkg / "blind_key.json").read_text(encoding="utf-8"))
    key = meta["key"]
    seed = int(meta.get("seed", 0) if seed is None else seed)
    if out.resolve() == pkg.resolve() or pkg.resolve() in out.resolve().parents:
        raise ValueError("la salida debe ser una carpeta aparte del paquete (contiene blind_key.json)")
    stim_src = pkg / "stimuli"
    stimuli: dict[str, dict] = {}
    needed_refs: set[str] = set()
    for sid, v in key.items():
        if not (stim_src / f"{sid}.wav").is_file():
            raise ValueError(f"falta estímulo {sid}.wav en {stim_src}")
        rid = v.get("reference_id")
        if rid:
            if not (stim_src / f"{rid}.wav").is_file():
                raise ValueError(f"falta referencia {rid}.wav en {stim_src}")
            needed_refs.add(rid)
        stimuli[sid] = {"ref": rid or None, "refkind": (v.get("reference_kind") or "unknown") if rid else None,
                        "na": sorted(v.get("na_dims", []))}
    extra = [p.name for p in stim_src.iterdir() if p.name not in {f"{s}.wav" for s in key} | {f"{r}.wav" for r in needed_refs}]
    if extra:
        raise ValueError(f"archivos no declarados en stimuli/: {extra}")

    if out.exists():
        shutil.rmtree(out)
    (out / "stimuli").mkdir(parents=True)
    hashes = {}
    for name in sorted([f"{s}.wav" for s in key] + [f"{r}.wav" for r in needed_refs]):
        shutil.copyfile(stim_src / name, out / "stimuli" / name)
        hashes[name] = hashlib.sha256((out / "stimuli" / name).read_bytes()).hexdigest()

    base_md = (pkg / "INSTRUCCIONES.md").read_text(encoding="utf-8") if (pkg / "INSTRUCCIONES.md").is_file() else L.INSTRUCTIONS
    instructions_md = base_md.rstrip() + "\n\n" + UI_NOTES.strip() + "\n"
    pid = package_id(seed, stimuli, hashes)
    data = {"schema": SCHEMA, "package_id": pid, "seed": seed, "scale": L.SCALE, "dimensions": list(L.DIMENSIONS), "stimuli": stimuli}
    data_json = json.dumps(data, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    html = (_PAGE.replace("__CSS__", _CSS).replace("__INSTRUCTIONS__", _md_to_html(instructions_md))
            .replace("__DATA__", data_json).replace("__JS__", _JS))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "INSTRUCCIONES.md").write_text(instructions_md, encoding="utf-8")
    manifest = {"schema": "opentraductor-listening-ui/v1", "package_id": pid, "seed": seed, "n_stimuli": len(key),
                "n_references": len(needed_refs), "sha256": hashes,
                "nota": "Sin nombres de sistema ni clave ciega; solo ids S####/R####."}
    (out / "listening_ui_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    problems = check_blind(out, key)
    if problems:
        shutil.rmtree(out)
        raise ValueError(f"fuga de información ciega: {problems}")
    return {"out": str(out), "n_stimuli": len(key), "n_references": len(needed_refs), "seed": seed, "package_id": pid}


def check_blind(out_dir: str | Path, key: dict) -> list[str]:
    """Busca nombres de sistema/ítem/archivo original en TODOS los archivos de salida
    (incluidos los bytes de los WAV) y valida los nombres de ``stimuli/``."""
    out = Path(out_dir)
    problems: list[str] = []
    if (out / "blind_key.json").exists():
        problems.append("blind_key.json presente")
    for p in (out / "stimuli").iterdir():
        if not _BLIND_NAME.match(p.name):
            problems.append(f"nombre no ciego: {p.name}")
    terms = _leak_terms(key)
    for p in out.rglob("*"):
        if p.is_file():
            blob = p.read_bytes()
            for t in terms:
                if p.suffix == ".wav" and len(t) < 8:
                    continue  # cadenas cortas darían falsos positivos en bytes de audio
                if t.encode("utf-8") in blob:
                    problems.append(f"{p.relative_to(out)} contiene {t!r}")
    return problems
