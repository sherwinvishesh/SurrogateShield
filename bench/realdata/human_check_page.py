"""A review page for the 100-row human check (HUMAN_CHECK.md): offline HTML, git-ignored.

    python -m bench.realdata.human_check_page                      # human_check.jsonl -> build/human_check/index.html
    python -m bench.realdata.human_check_page --check bench/realdata/human_check.done.jsonl   # lint the export, no text

Open ``bench/realdata/build/human_check/index.html`` in a browser (a file://
page; it loads nothing from the network and sends nothing). It shows one
message at a time with Sonnet's labels marked in the text, lets you fix the
four lists by selecting text, lints every entry as you go with the rule of
``bench/realworld.py`` (exact whole-word substring, known type, listed once,
keep never overlapping protect), keeps your work in the browser's local
storage, and exports ``human_check.done.jsonl`` in the schema that
``label.py --agreement`` reads. The page and the exported file are private
(0600; both paths are git-ignored). ``--check`` runs the same lint from Python
on the exported file and prints row ids and problems only, never text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

from bench import realworld as rw
from bench.realdata.common import ROOT, TASKS, read_jsonl

RD = ROOT / "bench" / "realdata"
SOURCE = RD / "human_check.jsonl"
DONE = RD / "human_check.done.jsonl"
PAGE = RD / "build" / "human_check" / "index.html"
FIELDS = ("task", "service_query", "protect", "sensitive", "optional", "keep")

CSS = """
:root{--bg:#fafaf7;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddd;--pro:#ffd3d3;--sen:#e6d5ff;--opt:#ffe9b3;--keep:#cfeedd;--bad:#b00020;--ok:#1b7f3b}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:2}
header b{font-size:16px}header .sp{flex:1}button{font:inherit;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.primary{background:#1c1c1a;color:#fff;border-color:#1c1c1a}button:disabled{opacity:.5;cursor:default}
main{display:grid;grid-template-columns:260px 1fr;min-height:calc(100vh - 52px)}
nav{border-right:1px solid var(--line);overflow:auto;max-height:calc(100vh - 52px);background:#fff}
nav div{padding:6px 10px;border-bottom:1px solid #f0f0ee;cursor:pointer;display:flex;gap:8px;font-size:13px}
nav div.cur{background:#eef3ff}nav .st{width:14px}nav .ds{color:var(--mute);width:62px}nav .bad{color:var(--bad)}
section{padding:16px 22px;max-width:980px}
.meta{color:var(--mute);font-size:13px;margin-bottom:8px}
details{margin:8px 0}details pre{background:#f1f1ee;padding:8px;border-radius:6px;white-space:pre-wrap}
pre.msg{white-space:pre-wrap;word-wrap:break-word;background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;font:15px/1.5 ui-monospace,Menlo,Consolas,monospace;user-select:text}
mark{padding:0 1px;border-radius:3px}mark.protect{background:var(--pro)}mark.sensitive{background:var(--sen)}mark.optional{background:var(--opt)}mark.keep{background:var(--keep)}
.tool{position:absolute;background:#fff;border:1px solid #bbb;border-radius:8px;padding:8px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;gap:6px;align-items:center;flex-wrap:wrap;z-index:3;max-width:560px}
.tool.on{display:flex}.tool code{background:#f1f1ee;padding:2px 6px;border-radius:4px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lists{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.list{border:1px solid var(--line);border-radius:8px;padding:10px;background:#fff}.list h3{margin:0 0 6px;font-size:14px;display:flex;align-items:center;gap:8px}
.list h3 i{display:inline-block;width:12px;height:12px;border-radius:3px}.list h3 .n{color:var(--mute);font-weight:normal}
.entry{display:flex;gap:6px;align-items:center;margin:4px 0}.entry input[type=text]{flex:1;font:inherit;padding:4px 6px;border:1px solid #ccc;border-radius:5px}
.entry select{font:inherit;padding:4px;border:1px solid #ccc;border-radius:5px}.entry button{padding:2px 7px}
.err{color:var(--bad);font-size:12.5px;margin:2px 0 4px}
.row{display:flex;gap:18px;align-items:center;margin-top:14px;flex-wrap:wrap}.row label{display:flex;gap:6px;align-items:center}
textarea{width:100%;font:inherit;padding:6px;border:1px solid #ccc;border-radius:6px;min-height:52px}
.hint{color:var(--mute);font-size:13px}.ok{color:var(--ok)}.warn{color:var(--bad)}
.foot{display:flex;gap:10px;margin:16px 0}
.banner{background:#fff6d6;border:1px solid #e8d48a;border-radius:6px;padding:6px 10px;font-size:13px;margin-bottom:10px}
kbd{font:12px ui-monospace,Menlo,monospace;background:#f1f1ee;border:1px solid #ccc;border-radius:4px;padding:0 4px}
"""

# Pure functions, kept together so they can be run under node for a check.
CORE_JS = r"""
/*CORE-START*/
const HC = (() => {
  const PROTECT = %(protect)s, SENSITIVE = %(sensitive)s, TASKS = %(tasks)s;
  const LISTS = ["protect", "sensitive", "optional", "keep"];
  function wordchar(c) {               // bench/realworld.py _wordchar
    if (!/[\p{L}\p{N}]/u.test(c)) return false;
    const cp = c.codePointAt(0);
    return !((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF)
      || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x0E00 && cp <= 0x0E7F) || (cp >= 0x20000 && cp <= 0x2FA1F));
  }
  function occurrences(text, value) {   // bench/realworld.py occurrences
    const out = []; if (!value) return out;
    let i = text.indexOf(value);
    while (i !== -1) {
      const j = i + value.length;
      const before = i ? text[i - 1] : " ", after = j < text.length ? text[j] : " ";
      const v0 = [...value][0], v1 = [...value].slice(-1)[0];
      if (!(wordchar(before) && wordchar(v0)) && !(wordchar(after) && wordchar(v1))) out.push([i, j]);
      i = text.indexOf(value, i + 1);
    }
    return out;
  }
  function entries(row, name) { return (row[name] || []).map(x => name === "keep" ? [x, null] : [x.value, x.type]); }
  function lint(row) {                  // bench/realworld.py lint + label.py validate, per row
    const errs = [], spans = {}, seen = new Map();
    if (!TASKS.includes(row.task)) errs.push("task not in the list");
    if (typeof row.service_query !== "boolean") errs.push("service_query must be true or false");
    for (const name of LISTS) {
      entries(row, name).forEach(([value, typ], k) => {
        const where = `${name}[${k}]`;
        if (typeof value !== "string" || !value.trim()) { errs.push(`${where} empty value`); return; }
        if (name === "protect" && !PROTECT.includes(typ)) { errs.push(`${where} bad type`); return; }
        if (name === "sensitive" && !SENSITIVE.includes(typ)) { errs.push(`${where} bad type`); return; }
        if (name === "optional" && (typeof typ !== "string" || !typ.trim())) { errs.push(`${where} needs a type`); return; }
        const occ = occurrences(row.text, value);
        if (!occ.length) errs.push(`${where} is not an exact whole-word substring of the message`);
        if (seen.has(value)) errs.push(`${where} repeats ${seen.get(value)}`); else seen.set(value, where);
        spans[where] = occ;
      });
    }
    for (const [w1, o1] of Object.entries(spans)) for (const [w2, o2] of Object.entries(spans))
      if (w1.startsWith("protect[") && w2.startsWith("keep[") && o1.some(([a, b]) => o2.some(([c, d]) => a < d && c < b)))
        errs.push(`${w1} overlaps ${w2}`);
    return errs;
  }
  function sortedKeys(o) {               // json.dumps(sort_keys=True) shape
    if (Array.isArray(o)) return o.map(sortedKeys);
    if (o && typeof o === "object") { const r = {}; for (const k of Object.keys(o).sort()) r[k] = sortedKeys(o[k]); return r; }
    return o;
  }
  function exportLines(rows) { return rows.map(r => JSON.stringify(sortedKeys(r))).join("\n") + "\n"; }
  function parseJsonl(text) { return text.split("\n").filter(l => l.trim()).map(l => JSON.parse(l)); }
  return { PROTECT, SENSITIVE, TASKS, LISTS, wordchar, occurrences, lint, exportLines, parseJsonl, sortedKeys };
})();
/*CORE-END*/
"""

APP_JS = r"""
const SRC = JSON.parse(document.getElementById("rows").textContent);
const KEY = "human-check:" + document.body.dataset.sha;
let rows = SRC.map(r => JSON.parse(JSON.stringify(r))), cur = 0, restored = false, pending = null;
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && Array.isArray(s.rows) && s.rows.length === SRC.length) { rows = s.rows; cur = s.cur || 0; restored = true; } } catch (e) {}
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
function save() { try { localStorage.setItem(KEY, JSON.stringify({ rows, cur, at: Date.now() })); } catch (e) {} }
function errorsOf(r) { return HC.lint(r); }
function renderNav() {
  const nav = $("nav"); nav.innerHTML = "";
  rows.forEach((r, i) => {
    const d = document.createElement("div"); if (i === cur) d.className = "cur";
    const bad = errorsOf(r).length, np = r.protect.length, ns = r.sensitive.length;
    d.innerHTML = `<span class="st">${r.checked ? "✓" : "○"}</span><span>${i + 1}</span><span class="ds">${esc(r.dataset)}</span>`
      + `<span class="${bad ? "bad" : ""}">${bad ? "lint" : (np + ns ? `${np}P ${ns}S` : "—")}</span>`;
    d.onclick = () => { cur = i; render(); };
    nav.appendChild(d);
  });
  const done = rows.filter(r => r.checked).length, bad = rows.filter(r => errorsOf(r).length).length;
  $("progress").textContent = `checked ${done}/${rows.length}` + (bad ? ` · ${bad} with lint errors` : "");
}
function marked(r) {
  const spans = [];
  for (const name of HC.LISTS) for (const [v] of (r[name] || []).map(x => name === "keep" ? [x] : [x.value]))
    for (const [a, b] of HC.occurrences(r.text, v || "")) spans.push([a, b, name]);
  spans.sort((x, y) => x[0] - y[0] || y[1] - x[1]);
  let out = "", pos = 0;
  for (const [a, b, name] of spans) { if (a < pos) continue; out += esc(r.text.slice(pos, a)) + `<mark class="${name}" title="${name}">` + esc(r.text.slice(a, b)) + "</mark>"; pos = b; }
  return out + esc(r.text.slice(pos));
}
function typeOptions(name, chosen) {
  const types = name === "protect" ? HC.PROTECT : name === "sensitive" ? HC.SENSITIVE : null;
  if (!types) return `<input type="text" class="typ" value="${esc(chosen || "")}" placeholder="type" style="width:120px">`;
  return `<select class="typ">${types.map(t => `<option ${t === chosen ? "selected" : ""}>${t}</option>`).join("")}</select>`;
}
function renderLists(r) {
  const errs = errorsOf(r), box = $("lists"); box.innerHTML = "";
  for (const name of HC.LISTS) {
    const div = document.createElement("div"); div.className = "list";
    const items = r[name] || [];
    div.innerHTML = `<h3><i class="${name}" style="background:var(--${{protect: "pro", sensitive: "sen", optional: "opt", keep: "keep"}[name]})"></i>${name} <span class="n">(${items.length})</span></h3>`;
    items.forEach((x, k) => {
      const e = document.createElement("div"); e.className = "entry";
      const value = name === "keep" ? x : x.value;
      e.innerHTML = `<input type="text" class="val" value="${esc(value || "")}">` + (name === "keep" ? "" : typeOptions(name, x.type)) + `<button title="remove">✕</button>`;
      e.querySelector(".val").oninput = ev => { if (name === "keep") r.keep[k] = ev.target.value; else r[name][k].value = ev.target.value; soft(); };
      const t = e.querySelector(".typ"); if (t) t.onchange = ev => { r[name][k].type = ev.target.value; soft(); };
      e.querySelector("button").onclick = () => { r[name].splice(k, 1); render(); };
      div.appendChild(e);
      const mine = errs.filter(m => m.startsWith(`${name}[${k}]`)); if (mine.length) { const p = document.createElement("div"); p.className = "err"; p.textContent = mine.join("; "); div.appendChild(p); }
    });
    const add = document.createElement("button"); add.textContent = "+ add"; add.style.marginTop = "6px";
    add.onclick = () => { if (name === "keep") r.keep.push(""); else r[name].push({ value: "", type: name === "protect" ? "PERSON" : name === "sensitive" ? "HEALTH" : "PERSON" }); render(); };
    div.appendChild(add); box.appendChild(div);
  }
  const other = errs.filter(m => !/^(protect|sensitive|optional|keep)\[/.test(m));
  $("rowerr").textContent = other.join("; ");
}
function soft() { save(); const r = rows[cur]; $("msg").innerHTML = marked(r); renderNav(); const errs = errorsOf(r);
  document.querySelectorAll("#lists .err").forEach(e => e.remove());
  document.querySelectorAll("#lists .list").forEach((div, li) => { const name = HC.LISTS[li]; div.querySelectorAll(".entry").forEach((e, k) => { const mine = errs.filter(m => m.startsWith(`${name}[${k}]`)); if (mine.length) { const p = document.createElement("div"); p.className = "err"; p.textContent = mine.join("; "); e.after(p); } }); }); }
function render() {
  const r = rows[cur];
  $("meta").textContent = `${cur + 1} / ${rows.length} · ${r.dataset} · ${r.id}` + (r.earlier_turns && r.earlier_turns.length ? ` · ${r.earlier_turns.length} earlier turn(s)` : "");
  const et = $("earlier"); if (r.earlier_turns && r.earlier_turns.length) { et.style.display = ""; et.querySelector("pre").textContent = r.earlier_turns.join("\n\n— — —\n\n"); } else et.style.display = "none";
  $("msg").innerHTML = marked(r);
  $("task").value = r.task; $("sq").checked = !!r.service_query; $("checked").checked = !!r.checked; $("notes").value = r.notes || "";
  renderLists(r); renderNav(); hideTool(); save();
  document.querySelector("nav .cur") && document.querySelector("nav .cur").scrollIntoView({ block: "nearest" });
}
function hideTool() { $("tool").classList.remove("on"); pending = null; }
document.addEventListener("mousedown", e => { if (!$("tool").contains(e.target) && !$("msg").contains(e.target)) hideTool(); });
$("msg").addEventListener("mouseup", ev => {
  const sel = window.getSelection(); const s = sel ? sel.toString() : "";
  const v = s.replace(/^\s+|\s+$/g, "");
  if (!v || v.includes("\n")) { hideTool(); return; }
  pending = v; $("selv").textContent = v; const tool = $("tool"); tool.classList.add("on");
  const rect = $("msg").getBoundingClientRect(); tool.style.left = Math.max(8, ev.clientX - rect.left) + "px"; tool.style.top = (ev.clientY - rect.top + 18) + "px";
  const occ = HC.occurrences(rows[cur].text, v); $("selhint").textContent = occ.length ? `${occ.length} whole-word match${occ.length > 1 ? "es" : ""}` : "not a whole word here: extend or trim the selection";
});
function addPending(name) {
  if (!pending) return; const r = rows[cur];
  if (name === "keep") r.keep.push(pending); else r[name].push({ value: pending, type: name === "optional" ? $("opttype").value || "PERSON" : $(name === "protect" ? "ptype" : "stype").value });
  window.getSelection() && window.getSelection().removeAllRanges(); render();
}
$("addp").onclick = () => addPending("protect"); $("adds").onclick = () => addPending("sensitive"); $("addo").onclick = () => addPending("optional"); $("addk").onclick = () => addPending("keep"); $("tclose").onclick = hideTool;
$("task").onchange = e => { rows[cur].task = e.target.value; soft(); };
$("sq").onchange = e => { rows[cur].service_query = e.target.checked; soft(); };
$("checked").onchange = e => { rows[cur].checked = e.target.checked; renderNav(); save(); };
$("notes").oninput = e => { rows[cur].notes = e.target.value; save(); };
$("prev").onclick = () => { cur = (cur + rows.length - 1) % rows.length; render(); };
$("next").onclick = () => { cur = (cur + 1) % rows.length; render(); };
$("nextun").onclick = () => { for (let k = 1; k <= rows.length; k++) { const i = (cur + k) % rows.length; if (!rows[i].checked) { cur = i; break; } } render(); };
$("export").onclick = () => {
  const done = rows.filter(r => r.checked).length, bad = rows.filter(r => errorsOf(r).length).length;
  if (bad && !confirm(`${bad} row(s) have lint errors; label.py --agreement counts them as they are. Export anyway?`)) return;
  if (done < rows.length && !confirm(`${rows.length - done} row(s) are not marked checked and will not count. Export anyway?`)) return;
  const blob = new Blob([HC.exportLines(rows)], { type: "application/x-ndjson" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "human_check.done.jsonl"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  $("exportmsg").textContent = `exported ${rows.length} rows, ${done} checked. Move the file to bench/realdata/human_check.done.jsonl, then run label.py --agreement.`;
};
$("import").onchange = e => {
  const f = e.target.files[0]; if (!f) return; const rd = new FileReader();
  rd.onload = () => { try { const got = HC.parseJsonl(rd.result); const byId = new Map(got.map(r => [r.id, r])); let n = 0; rows = rows.map(r => { const g = byId.get(r.id); if (g && g.text === r.text) { n++; return g; } return r; }); save(); render(); $("exportmsg").textContent = `imported ${n} row(s) by id`; } catch (err) { $("exportmsg").textContent = "could not read that file: " + err; } };
  rd.readAsText(f); e.target.value = "";
};
$("reset").onclick = () => { if (!confirm("Discard every change kept in this browser and start again from the file?")) return; try { localStorage.removeItem(KEY); } catch (e) {} rows = SRC.map(r => JSON.parse(JSON.stringify(r))); cur = 0; render(); };
document.addEventListener("keydown", e => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "j" || e.key === "ArrowRight") $("next").click(); else if (e.key === "k" || e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "c") { rows[cur].checked = !rows[cur].checked; $("checked").checked = rows[cur].checked; renderNav(); save(); }
  else if (e.key === "n") $("nextun").click();
});
if (restored) $("banner").style.display = "";
render();
"""

PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human check · %(n)d messages</title><meta name="robots" content="noindex"><style>%(css)s</style></head>
<body data-sha="%(sha)s">
<header><b>Human check</b><span id="progress" class="hint"></span><span class="sp"></span>
<span class="hint"><kbd>j</kbd>/<kbd>k</kbd> next/previous · <kbd>n</kbd> next unchecked · <kbd>c</kbd> toggle checked</span>
<button id="reset" title="discard the browser's saved work">Reset to file</button>
<label><input type="file" id="import" accept=".jsonl,.txt,application/x-ndjson" style="display:none"><button onclick="document.getElementById('import').click()">Import done file</button></label>
<button id="export" class="primary">Export human_check.done.jsonl</button></header>
<main><nav id="nav"></nav><section>
<div id="banner" class="banner" style="display:none">Restored your work from this browser's storage (it never leaves this machine). "Reset to file" starts again from Sonnet's labels.</div>
<div id="meta" class="meta"></div>
<details id="earlier"><summary class="hint">earlier turns of the same user (context only; label the message below)</summary><pre></pre></details>
<div style="position:relative"><pre class="msg" id="msg"></pre>
<div class="tool" id="tool"><code id="selv"></code><span class="hint" id="selhint"></span>
<select id="ptype">%(ptypes)s</select><button id="addp">protect</button>
<select id="stype">%(stypes)s</select><button id="adds">sensitive</button>
<input id="opttype" type="text" placeholder="type" value="PERSON" style="width:90px;font:inherit;padding:4px;border:1px solid #ccc;border-radius:5px"><button id="addo">optional</button>
<button id="addk">keep</button><button id="tclose" title="close">✕</button></div></div>
<p class="hint">Select text in the message to add it to a list. Values must be exact whole words of the message, listed once; keep must not overlap protect; sensitive is only a special-category fact about a private person. Rules: <code>bench/realdata/HUMAN_CHECK.md</code>.</p>
<div id="lists" class="lists"></div><div id="rowerr" class="err"></div>
<div class="row"><label>task <select id="task">%(tasks)s</select></label><label><input type="checkbox" id="sq"> service_query (the message is about the service itself)</label>
<label><input type="checkbox" id="checked"> <b>checked</b> (counts in the agreement)</label></div>
<div class="row" style="display:block"><label class="hint">notes (why, without quoting the text)</label><textarea id="notes"></textarea></div>
<div class="foot"><button id="prev">← previous</button><button id="next">next →</button><button id="nextun">next unchecked</button><span id="exportmsg" class="hint"></span></div>
</section></main>
<script id="rows" type="application/json">%(rows)s</script>
<script>%(core)s</script>
<script>%(app)s</script>
</body></html>
"""


def build(rows: List[dict]) -> str:
    sha = hashlib.sha256("".join(r["id"] + "\n" + r["text"] + "\n" for r in rows).encode()).hexdigest()[:12]
    protect, sensitive = sorted(rw.PROTECT_TYPES), sorted(rw.SENSITIVE_TYPES)
    core = CORE_JS % {"protect": json.dumps(protect), "sensitive": json.dumps(sensitive), "tasks": json.dumps(list(TASKS))}
    opts = lambda xs: "".join(f"<option>{x}</option>" for x in xs)
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return PAGE_HTML % {"n": len(rows), "css": CSS, "sha": sha, "ptypes": opts(protect), "stypes": opts(sensitive),
                        "tasks": opts(TASKS), "rows": data, "core": core, "app": APP_JS}


def write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def check(path: Path, source: Path = SOURCE) -> int:
    """Lint an exported file the way ``label.py validate`` does; ids and
    problems only. Returns the number of rows with problems."""
    from bench.realdata.label import validate
    rows = read_jsonl(path)
    src = {r["id"]: r for r in read_jsonl(source)} if source.exists() else None
    bad, checked = 0, 0
    for r in rows:
        problems = []
        if "sonnet" not in r or "text" not in r:
            problems.append("missing sonnet or text (export from the page, do not retype rows)")
        elif src is not None:
            if r["id"] not in src:
                problems.append("id not in the source file")
            elif src[r["id"]]["text"] != r["text"] or src[r["id"]]["sonnet"] != r["sonnet"]:
                problems.append("text or sonnet differs from the source file")
        if not problems:
            problems = validate({f: r.get(f) for f in FIELDS}, r["text"])
        checked += bool(r.get("checked"))
        if problems:
            bad += 1
            print(f"{r.get('id', '?')}: {'; '.join(problems)}")
    print(f"{len(rows)} rows, {checked} checked, {bad} with problems"
          + ("" if src is not None else " (source file absent: ids and texts not compared)"))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", type=Path, default=SOURCE, help="the human-check file (default human_check.jsonl)")
    ap.add_argument("--out", type=Path, default=PAGE, help="where to write the page (default build/human_check/index.html)")
    ap.add_argument("--check", type=Path, metavar="DONE", help="lint an exported done file instead; prints ids and problems only")
    args = ap.parse_args(argv)
    if args.check:
        return 1 if check(args.check, args.src) else 0
    if not args.src.exists():
        print(f"{args.src} is missing: run  python -m bench.realdata.label --human-check  first", file=sys.stderr)
        return 2
    rows = read_jsonl(args.src)
    write_private(args.out, build(rows))
    print(f"{len(rows)} rows -> {args.out} (0600, git-ignored). Open it in a browser; export when done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
