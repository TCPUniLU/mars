/* MARS web GUI — single-page front end (vanilla JS, no build step).
 *
 * Talks to the local server API (/api/...) served by `mars gui`.
 * 3D: 3Dmol.js, plots: Plotly (both from cdnjs). Everything degrades
 * gracefully if the CDN is unreachable (jobs still run).
 */
"use strict";

// ------------------------------------------------------------------ helpers
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
};
const HAS_3DMOL = () => typeof window.$3Dmol !== "undefined";
const HAS_PLOTLY = () => typeof window.Plotly !== "undefined";

async function api(path, { method = "GET", body } = {}) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  let data = null;
  const ctype = resp.headers.get("Content-Type") || "";
  if (ctype.includes("application/json")) data = await resp.json();
  else data = await resp.text();
  if (!resp.ok) {
    const msg = data && data.error ? data.error : `${resp.status} ${resp.statusText}`;
    throw new Error(msg);
  }
  return data;
}

function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), isErr ? 6000 : 2500);
}

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

function fmtDuration(s) {
  if (s === null || s === undefined || !isFinite(s) || s < 0) return "";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return `${sec}s`;
}

function fmtTime(t) {
  return t ? new Date(t * 1000).toLocaleString() : "";
}

const fileUrl = (jobId, path, download = false) =>
  `/api/jobs/${encodeURIComponent(jobId)}/files/` +
  path.split("/").map(encodeURIComponent).join("/") + (download ? "?download=1" : "");

function readFileB64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",", 2)[1] || "");
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

// ------------------------------------------------------------------ XYZ utils
function parseXYZFrames(text) {
  const lines = text.split(/\r?\n/);
  const frames = [];
  let i = 0;
  while (i < lines.length) {
    if (!lines[i].trim()) { i++; continue; }
    const n = parseInt(lines[i].trim().split(/\s+/)[0], 10);
    if (!Number.isFinite(n) || n <= 0 || i + 1 + n >= lines.length + 1) break;
    const block = lines.slice(i, i + 2 + n);
    if (block.length < 2 + n) break;
    frames.push({ n, comment: block[1] || "", text: block.join("\n") + "\n", atoms: block.slice(2) });
    i += 2 + n;
  }
  return frames;
}

function parseEnergy(comment) {
  const m = /Energy\s*=\s*([-+0-9.eE]+)\s*(kcal\/mol|eV)?/i.exec(comment || "");
  if (!m) return null;
  let v = parseFloat(m[1]);
  if ((m[2] || "").toLowerCase() === "ev") v *= 23.060547830619;
  return v;
}

// ------------------------------------------------------------------ theme
function currentTheme() {
  const forced = document.documentElement.dataset.theme;
  if (forced) return forced;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else if (theme === "") delete document.documentElement.dataset.theme;
  try { if (theme !== null && theme !== undefined) localStorage.setItem("mars-gui-theme", theme); } catch (e) { /* ignore */ }
  for (const v of MolView.all) v.setBackground();
  if (HAS_PLOTLY()) {
    for (const id of ["ir-plot", "energy-plot"]) {
      const node = document.getElementById(id);
      if (node && node.data) Plotly.relayout(node, plotLayoutColors());
    }
  }
}
function plotLayoutColors() {
  return {
    article_bgcolor: cssVar("--panel"),
    plot_bgcolor: cssVar("--panel"),
    "font.color": cssVar("--text"),
    "xaxis.gridcolor": cssVar("--border"),
    "yaxis.gridcolor": cssVar("--border"),
  };
}
function basePlotLayout(extra = {}) {
  return Object.assign({
    margin: { l: 60, r: 20, t: 20, b: 50 },
    article_bgcolor: cssVar("--panel"),
    plot_bgcolor: cssVar("--panel"),
    font: { color: cssVar("--text"), size: 12 },
    showlegend: false,
    hovermode: "closest",
  }, extra);
}

// ------------------------------------------------------------------ 3D viewer
class MolView {
  constructor(node) {
    this.node = node;
    this.v = null;
    this.style = "ballstick";
    this.labels = false;
    this.hasModel = false;
    MolView.all.push(this);
  }
  ensure() {
    if (!HAS_3DMOL()) return false;
    if (!this.v) {
      this.v = $3Dmol.createViewer(this.node, { backgroundColor: cssVar("--viewer-bg") || "white" });
    }
    return true;
  }
  setBackground() {
    if (this.v) { this.v.setBackgroundColor(cssVar("--viewer-bg") || "white"); this.v.render(); }
  }
  clear() {
    if (!this.v) return;
    this.v.stopAnimate();
    this.v.removeAllLabels();
    this.v.clear();
    this.v.render();
    this.hasModel = false;
  }
  /** Show one XYZ frame. keepView: keep the camera (frame playback). */
  show(xyzText, { keepView = false } = {}) {
    if (!this.ensure()) return null;
    this.v.stopAnimate();
    this.v.removeAllModels();
    this.v.removeAllLabels();
    const model = this.v.addModel(xyzText, "xyz");
    this.applyStyle();
    if (!keepView || !this.hasModel) this.v.zoomTo();
    this.hasModel = true;
    this.v.render();
    return model;
  }
  applyStyle() {
    if (!this.v) return;
    const styles = {
      ballstick: { stick: { radius: 0.14, colorscheme: "Jmol" }, sphere: { scale: 0.25, colorscheme: "Jmol" } },
      stick: { stick: { radius: 0.18, colorscheme: "Jmol" } },
      sphere: { sphere: { scale: 0.9, colorscheme: "Jmol" } },
      line: { line: { colorscheme: "Jmol" } },
    };
    this.v.setStyle({}, styles[this.style] || styles.ballstick);
    this.v.removeAllLabels();
    if (this.labels) {
      const atoms = this.v.selectedAtoms({});
      atoms.forEach((a, i) => {
        this.v.addLabel(`${a.elem}${i}`, {
          position: { x: a.x, y: a.y, z: a.z }, fontSize: 11, showBackground: true,
          backgroundOpacity: 0.55, backgroundColor: "black", fontColor: "white", inFront: true,
        });
      });
    }
    this.v.render();
  }
  setStyle(style) { this.style = style; this.applyStyle(); }
  setLabels(on) { this.labels = on; this.applyStyle(); }
  resetView() { if (this.v) { this.v.zoomTo(); this.v.render(); } }
  /** Animate a normal mode: xyz frame with dx dy dz columns. */
  vibrate(xyzText, amplitude) {
    if (!this.ensure()) return;
    this.v.stopAnimate();
    this.v.removeAllModels();
    this.v.removeAllLabels();
    const model = this.v.addModel(xyzText, "xyz");
    this.applyStyle();
    if (!this.hasModel) this.v.zoomTo();
    this.hasModel = true;
    model.vibrate(14, amplitude, true);
    this.v.animate({ loop: "backAndForth", reps: 0, interval: 45 });
    this.v.render();
  }
  resize() { if (this.v) { this.v.resize(); this.v.render(); } }
}
MolView.all = [];

function bindStyleControls(view, segSel, labelsSel) {
  $$(`${segSel} button`).forEach((b) => b.addEventListener("click", () => {
    $$(`${segSel} button`).forEach((x) => x.classList.toggle("active", x === b));
    view.setStyle(b.dataset.style);
  }));
  $(labelsSel).addEventListener("change", (e) => view.setLabels(e.target.checked));
}

// ------------------------------------------------------------------ state
const S = {
  schema: null,
  task: "conformer",
  structure: null, // upload record
  structFrames: [],
  potential: "so3lr",
  potValues: {}, // flag -> value (for the current potential)
  taskValues: {}, // task -> {flag: value}
  jobs: [],
  selectedJob: null,
  jobDetail: null,
  skew: 0,
  log: { jobId: null, offset: 0, raw: "" },
  result: { key: null, frames: [], playing: null },
  ir: null,
};

const structView = new MolView($("#struct-viewer"));
const resView = new MolView($("#res-viewer"));
const modeView = new MolView($("#mode-viewer"));

// ------------------------------------------------------------------ tabs
function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-page").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  setTimeout(() => MolView.all.forEach((v) => v.resize()), 30);
  if (name === "jobs") refreshJobs();
  if (name === "models" && typeof loadModels === "function") loadModels();
  if (location.hash.slice(1).split("/")[0] !== name) history.replaceState(null, "", `#${name}`);
}
function bindNav(root = document) {
  $$(".tab", root).forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $$("[data-goto]", root).forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); showTab(b.dataset.goto); }));
}
bindNav();

function addPlugins(list) {
  for (const p of list || []) {
    const btn = el("button", { class: "tab", "data-tab": `plugin-${p.id}`, role: "tab" },
      el("span", { class: "nav-ico", text: "◇" }), p.title);
    $("#plugin-nav").append(btn);
    btn.addEventListener("click", () => showTab(`plugin-${p.id}`));
    $("#plugin-pages").append(el("section", { id: `tab-plugin-${p.id}`, class: "tab-page" },
      el("iframe", { class: "plugin-frame", src: `/plugins/${encodeURIComponent(p.id)}/`, title: p.title })));
  }
}

function openJob(id) {
  S.selectedJob = id; S.jobDetail = null; renderJobList(); loadJobDetail(id);
  history.replaceState(null, "", `#jobs/${encodeURIComponent(id)}`);
}

function renderHomeJobs() {
  const ul = $("#home-jobs");
  if (!ul) return;
  ul.innerHTML = "";
  $("#home-jobs-empty").classList.toggle("hidden", S.jobs.length > 0);
  for (const j of S.jobs.slice(0, 6)) {
    ul.append(el("li", {
      class: "job-item",
      onclick: () => { showTab("jobs"); openJob(j.id); },
    },
    el("div", { class: "j-top" }, el("span", { class: "j-name", text: j.name, title: j.id }),
      el("span", { class: `status ${j.status}`, text: j.status })),
    el("div", { class: "j-sub" }, el("span", { text: `${j.task_label} · ${j.potential}` }),
      el("span", { text: elapsed(j) }))));
  }
}

function showSub(name) {
  $$(".subtab").forEach((t) => t.classList.toggle("active", t.dataset.sub === name));
  $$(".sub-page").forEach((p) => p.classList.toggle("active", p.id === `sub-${name}`));
  setTimeout(() => {
    MolView.all.forEach((v) => v.resize());
    if (HAS_PLOTLY()) ["ir-plot", "energy-plot"].forEach((id) => {
      const n = document.getElementById(id);
      if (n && n.data) Plotly.Plots.resize(n);
    });
  }, 30);
  if (name === "log") scrollLog();
  if (name === "cite") loadCite();
}

async function loadCite() {
  const id = S.selectedJob, ol = $("#cite-list");
  if (!id) return;
  ol.innerHTML = "";
  let d;
  try { d = await api(`/api/jobs/${encodeURIComponent(id)}/cite`); } catch (e) { ol.textContent = e.message; return; }
  for (const r of d.refs) {
    ol.append(el("li", {}, r.text + " ", r.url ? el("a", { href: r.url, target: "_blank", rel: "noopener", text: r.doi ? `doi:${r.doi}` : r.url }) : null));
  }
  $("#cite-notes").textContent = (d.notes || []).join(" ");
  $("#cite-bib").href = `/api/jobs/${encodeURIComponent(id)}/cite?format=bib`;
  $("#cite-bib").download = `${id}.bib`;
  $("#cite-copy").onclick = () => navigator.clipboard.writeText(d.refs.map((r, i) => `${i + 1}. ${r.text}${r.doi ? " doi:" + r.doi : ""}`).concat(d.notes || []).join("\n"))
    .then(() => toast("References copied"));
}
$$(".subtab").forEach((t) => t.addEventListener("click", () => showSub(t.dataset.sub)));

// ------------------------------------------------------------------ structure panel
async function uploadStructure(filename, payload) {
  try {
    const rec = await api("/api/upload", { method: "POST", body: Object.assign({ filename, kind: "structure" }, payload) });
    setStructure(rec);
    toast(`Loaded ${rec.filename}`);
  } catch (e) {
    toast(`Could not read structure: ${e.message}`, true);
  }
}

function setStructure(rec) {
  S.structure = rec;
  S.structFrames = parseXYZFrames(rec.xyz || "");
  const s = rec.summary || {};
  $("#struct-summary").innerHTML = "";
  $("#struct-summary").append(
    el("strong", { text: s.formula || "" }), ` · ${s.n_atoms} atoms`,
    s.n_frames > 1 ? ` · ${s.n_frames} frames` : "", ` · ${rec.filename}`,
  );
  $("#struct-empty").classList.add("hidden");
  const multi = S.structFrames.length > 1;
  $("#struct-frames").classList.toggle("hidden", !multi);
  if (multi) {
    const r = $("#struct-frame");
    r.max = S.structFrames.length - 1;
    r.value = 0;
  }
  showStructFrame(0, false);
  if (!HAS_3DMOL()) $("#struct-empty").textContent = `${s.formula} loaded (3D viewer unavailable)`;
  updateRunState();
  schedulePreview();
}

function showStructFrame(i, keepView = true) {
  const fr = S.structFrames[i];
  if (!fr) return;
  structView.show(fr.text, { keepView });
  const e = parseEnergy(fr.comment);
  $("#struct-frame-label").textContent = `${i + 1}/${S.structFrames.length}` + (e !== null ? ` · ${e.toFixed(3)} kcal/mol` : "");
}

$("#struct-frame").addEventListener("input", (e) => showStructFrame(+e.target.value));
bindStyleControls(structView, "#style-seg", "#labels-toggle");
$("#reset-view").addEventListener("click", () => structView.resetView());

async function handleStructureFile(file) {
  if (!file) return;
  if (file.size > 150 * 1048576) { toast("File too large", true); return; }
  const b64 = await readFileB64(file);
  await uploadStructure(file.name, { content_b64: b64 });
}
$("#file-input").addEventListener("change", (e) => handleStructureFile(e.target.files[0]));
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => handleStructureFile(e.dataTransfer.files[0]));
// Dropping anywhere on the setup page also works.
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.target.closest && e.target.closest("#dropzone")) return;
  if ($("#tab-setup").classList.contains("active") && e.dataTransfer.files.length) {
    handleStructureFile(e.dataTransfer.files[0]);
  }
});
$("#paste-use").addEventListener("click", () => {
  const text = $("#paste-xyz").value.trim();
  if (!text) { toast("Paste an XYZ block first", true); return; }
  uploadStructure("pasted.xyz", { content: text + "\n" });
});

// ------------------------------------------------------------------ task / form
// Every CLI flag of the task is in the schema, grouped in sections. `show_if`
// rules (mirrored server-side) hide options that do not apply to the current
// choices; the filter box finds any option, greying out the inactive ones.
const isEmptyVal = (v) => v === null || v === undefined || v === "" || v === false || (Array.isArray(v) && !v.length);
const asBool = (v) => v === true || v === "true" || v === "on" || v === 1 || v === "1";

function condHolds(c, values, fields) {
  const f = fields[c.flag];
  let v = c.flag in values ? values[c.flag] : (f ? f.default : null);
  if (f && f.type === "bool") v = asBool(v);
  if ("is" in c) return typeof c.is === "boolean" ? !!v === c.is : String(v) === String(c.is);
  if ("in" in c) return v !== null && v !== undefined && c.in.map(String).includes(String(v));
  if ("not_in" in c) return v === null || v === undefined || !c.not_in.map(String).includes(String(v));
  if ("empty" in c) return isEmptyVal(v) === c.empty;
  return true;
}
function condText(c) {
  if ("is" in c) return c.is === true ? c.flag : c.is === false ? `no ${c.flag}` : `${c.flag} ${c.is}`;
  if ("in" in c) return `${c.flag} ${c.in.join(" or ")}`;
  if ("not_in" in c) return `${c.flag} not ${c.not_in.join("/")}`;
  if ("empty" in c) return c.empty ? `${c.flag} unset` : `${c.flag} set`;
  return "";
}
function sameValue(field, v) {
  const d = field.default;
  if (field.type === "bool") return asBool(v) === asBool(d);
  if (isEmptyVal(v)) return isEmptyVal(d) || field.type === "optional";
  if (field.type === "list") {
    const items = Array.isArray(v) ? v : String(v).trim().split(/[\s,]+/);
    return Array.isArray(d) && items.length === d.length && items.every((x, i) => String(+x) === String(+d[i]));
  }
  if (field.type === "int" || field.type === "float") return d !== null && d !== undefined && +v === +d;
  return String(v) === String(d === null || d === undefined ? "" : d);
}

function fieldInput(field, value, onChange) {
  const id = `f-${field.flag.replace(/[^a-z0-9]/gi, "")}-${Math.random().toString(36).slice(2, 7)}`;
  const help = field.help || "";
  const flagTag = el("span", { class: "flag-tag mono", text: field.flag });
  if (field.type === "bool") {
    const input = el("input", { type: "checkbox", id, "data-flag": field.flag });
    input.checked = asBool(value);
    input.addEventListener("change", onChange);
    return el("label", { class: "check field-check", title: help }, input, el("span", {}, field.label, flagTag));
  }
  if (field.type === "optional") {
    // `--flag` alone, or `--flag VALUE`
    const box = el("input", { type: "checkbox", "data-flag": field.flag, "data-kind": "optional" });
    const num = el("input", { type: "number", step: "any", placeholder: "default", class: "opt-value" });
    box.checked = !isEmptyVal(value);
    if (value !== true && !isEmptyVal(value)) num.value = String(value);
    num.disabled = !box.checked;
    box.addEventListener("change", () => { num.disabled = !box.checked; onChange(); });
    num.addEventListener("input", onChange);
    return el("div", { class: "field", title: help },
      el("span", {}, field.label, flagTag), el("div", { class: "opt-row" }, box, num));
  }
  let input;
  if (field.type === "choice") {
    input = el("select", { id, "data-flag": field.flag });
    if (field.default === null || field.default === undefined) input.append(el("option", { value: "", text: "(auto)" }));
    for (const c of field.choices || []) input.append(el("option", { value: String(c), text: String(c) }));
    input.value = value === null || value === undefined ? "" : String(value);
  } else if (field.type === "file" || field.type === "files") {
    return fileFieldInput(field, value, onChange);
  } else {
    const isNum = field.type === "int" || field.type === "float";
    const dflt = Array.isArray(field.default) ? field.default.join(" ") : field.default;
    input = el("input", {
      type: isNum ? "number" : "text", id, "data-flag": field.flag,
      step: field.type === "int" ? "1" : "any",
      placeholder: dflt === null || dflt === undefined || dflt === "" ? "auto" : String(dflt),
    });
    if (Array.isArray(value)) input.value = value.join(" ");
    else if (value !== null && value !== undefined) input.value = String(value);
  }
  input.addEventListener("input", onChange);
  input.addEventListener("change", onChange);
  return el("label", { class: "field", for: id, title: help }, el("span", {}, field.label, flagTag), input);
}

function fileFieldInput(field, value, onChange) {
  const multi = field.type === "files";
  const names = el("span", { class: "muted small" });
  const ids = Array.isArray(value) ? value : value ? [value] : [];
  const holder = el("label", { class: "field", title: field.help || "" },
    el("span", {}, field.label, el("span", { class: "flag-tag mono", text: field.flag })));
  holder.dataset.flag = field.flag;
  holder.dataset.ids = JSON.stringify(ids);
  names.textContent = ids.length ? `${ids.length} file(s) uploaded` : "no file";
  const input = el("input", { type: "file", multiple: multi });
  input.addEventListener("change", async () => {
    const out = [];
    for (const f of input.files) {
      try {
        const rec = await api("/api/upload", {
          method: "POST", body: { filename: f.name, content_b64: await readFileB64(f), kind: "file" },
        });
        out.push(rec.id);
      } catch (e) { toast(`Upload failed: ${e.message}`, true); }
    }
    holder.dataset.ids = JSON.stringify(out);
    names.textContent = out.length ? Array.from(input.files).map((f) => f.name).join(", ") : "no file";
    onChange();
  });
  holder.append(input, names);
  return holder;
}

function readForm(container) {
  const values = {};
  $$("[data-flag]", container).forEach((node) => {
    const flag = node.dataset.flag;
    if (node.tagName === "LABEL") {
      const ids = JSON.parse(node.dataset.ids || "[]");
      if (ids.length) values[flag] = node.querySelector("input").multiple ? ids : ids[0];
      return;
    }
    if (node.dataset.kind === "optional") {
      const num = node.parentElement.querySelector(".opt-value").value;
      values[flag] = node.checked ? (num === "" ? true : num) : false;
    } else if (node.type === "checkbox") values[flag] = node.checked;
    else values[flag] = node.value;
  });
  return values;
}

/** Render `fields` (grouped in `sections`) into `container`; returns {flag: field}. */
function renderOptionForm(container, sections, values, onChange, openState) {
  container.innerHTML = "";
  const byFlag = {};
  for (const sec of sections) {
    const grid = el("div", { class: "form-grid" });
    for (const f of sec.fields) {
      byFlag[f.flag] = f;
      const v = f.flag in values ? values[f.flag] : f.default;
      const wrap = el("div", { class: "fwrap", "data-field": f.flag }, fieldInput(f, v, onChange));
      if (f.help) wrap.append(el("div", { class: "help-line", title: f.help, text: f.help }));
      if (f.show_if && f.show_if.length) wrap.append(el("div", { class: "cond-note", text: `needs ${f.show_if.map(condText).join(", ")}` }));
      grid.append(wrap);
    }
    if (sec.title === null) { container.append(grid); continue; }
    const badge = el("span", { class: "sec-badge" });
    const det = el("details", { class: "opt-section", "data-title": sec.title },
      el("summary", {}, el("span", { text: sec.title }), badge), grid);
    det.open = openState && sec.title in openState ? openState[sec.title] : !!sec.open;
    if (openState) det.addEventListener("toggle", () => { openState[sec.title] = det.open; });
    container.append(det);
  }
  return byFlag;
}

/** Show / hide fields by their show_if rules and the option filter; mark changed values. */
function applyVisibility(container, byFlag, filter = "") {
  const values = readForm(container);
  const q = filter.trim().toLowerCase();
  let hiddenMatches = 0;
  $$(".fwrap", container).forEach((w) => {
    const f = byFlag[w.dataset.field];
    const active = (f.show_if || []).every((c) => condHolds(c, values, byFlag));
    const match = !q || `${f.label} ${f.flag} ${f.help}`.toLowerCase().includes(q);
    w.classList.toggle("hidden", !match || (!active && !q));
    w.classList.toggle("inactive", !active);
    $$("input, select", w).forEach((i) => { if (!i.classList.contains("opt-value")) i.disabled = !active; });
    w.classList.toggle("changed", active && f.type !== "file" && f.type !== "files" && !sameValue(f, values[f.flag]));
    if (q && match && !active) hiddenMatches += 1;
  });
  $$(".opt-section", container).forEach((d) => {
    const shown = $$(".fwrap", d).filter((w) => !w.classList.contains("hidden"));
    d.classList.toggle("hidden", !shown.length);
    if (q && shown.length) d.open = true;
    const n = shown.filter((w) => w.classList.contains("changed")).length;
    const badge = $(".sec-badge", d);
    badge.textContent = n ? `${n} set` : `${shown.length}`;
    badge.classList.toggle("on", n > 0);
  });
  return hiddenMatches;
}

S.openSections = {};
S.taskFields = {};
S.potFields = {};

function refreshTaskVisibility() {
  const q = $("#opt-filter").value;
  const inactive = applyVisibility($("#task-form"), S.taskFields, q);
  const note = $("#opt-filter-note");
  note.textContent = q && inactive ? `${inactive} matching option(s) are inactive with the current choices (greyed).` : "";
}

function onTaskChange() {
  S.taskValues[S.task] = readForm($("#task-form"));
  refreshTaskVisibility();
  schedulePreview();
}

function renderTasks() {
  const list = $("#task-list");
  list.innerHTML = "";
  for (const [key, t] of Object.entries(S.schema.tasks)) {
    list.append(el("button", {
      class: "task-card" + (key === S.task ? " active" : ""), type: "button",
      onclick: () => { S.taskValues[S.task] = readForm($("#task-form")); S.task = key; renderTasks(); },
    }, el("div", { class: "t-name", text: t.label }), el("div", { class: "t-cmd", text: t.command.trim() })));
  }
  const t = S.schema.tasks[S.task];
  $("#task-desc").textContent = t.description;
  const n = t.fields.length;
  $("#opt-count").textContent = `${n} options · ${t.sections.length} sections`;
  S.openSections[S.task] = S.openSections[S.task] || {};
  S.taskFields = renderOptionForm($("#task-form"), t.sections, S.taskValues[S.task] || {}, onTaskChange, S.openSections[S.task]);
  refreshTaskVisibility();
  const shown = $$(".opt-section", $("#task-form")).filter((d) => !d.classList.contains("hidden"));
  $("#opt-expand").textContent = shown.length && shown.every((d) => d.open) ? "Collapse all" : "Expand all";
  renderPotentialForm();
  schedulePreview();
}

function renderPotentialForm() {
  const t = S.schema.tasks[S.task];
  const fields = t.potential_fields[S.potential] || [];
  const form = $("#potential-form");
  const onChange = () => { S.potValues = readForm(form); applyVisibility(form, S.potFields); schedulePreview(); };
  S.potFields = renderOptionForm(form, [{ title: null, fields }], S.potValues, onChange, null);
  applyVisibility(form, S.potFields);
  if (!fields.length) form.append(el("p", { class: "muted small", text: "No extra options for this potential." }));
}

$("#potential").addEventListener("change", (e) => {
  S.potential = e.target.value;
  S.potValues = {};
  renderPotentialForm();
  renderEnvSelect(true);
  schedulePreview();
});
$("#opt-filter").addEventListener("input", refreshTaskVisibility);
$("#opt-reset").addEventListener("click", () => {
  S.taskValues[S.task] = {};
  S.potValues = {};
  renderTasks();
  toast("Options reset to the CLI defaults");
});
$("#opt-expand").addEventListener("click", () => {
  const dets = $$(".opt-section", $("#task-form")).filter((d) => !d.classList.contains("hidden"));
  const open = !dets.every((d) => d.open);
  dets.forEach((d) => { d.open = open; });
  $("#opt-expand").textContent = open ? "Collapse all" : "Expand all";
});

// ------------------------------------------------------------------ config files (mars --config)
function downloadText(filename, text, type = "application/toml") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("#cfg-save").addEventListener("click", async () => {
  try {
    const res = await api("/api/config/export", { method: "POST", body: buildPayload() });
    downloadText(res.filename, res.text);
  } catch (e) { toast(`Cannot save the config: ${e.message}`, true); }
});

$("#cfg-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  try {
    const res = await api("/api/config/import", { method: "POST", body: { filename: file.name, text: await file.text() } });
    applyLoadedConfig(res, file.name);
  } catch (err) { toast(`Cannot load ${file.name}: ${err.message}`, true); }
});

function applyLoadedConfig(res, name) {
  const p = res.payload;
  S.task = p.task;
  S.taskValues[p.task] = p.options;
  S.potential = p.potential;
  $("#potential").value = p.potential;
  S.potValues = p.potential_options;
  $("#charge").value = p.charge;
  $("#spin").value = p.spin;
  $("#float64").checked = !!p.float64;
  $("#device").value = p.device;
  $("#advanced").value = p.advanced || "";
  if (p.name) $("#job-name").value = p.name;
  renderTasks();
  renderEnvSelect(true);
  const notes = [...res.warnings];
  if (res.input && (!S.structure || (S.structure.filename !== res.input && S.structure.run_name !== res.input))) {
    notes.unshift(`The config names the input ${res.input}: load that structure in step 1.`);
  }
  const box = $("#cfg-notice");
  box.innerHTML = "";
  box.append(el("strong", { text: `Loaded ${name}` }), ...notes.map((n) => el("div", { text: n })));
  box.classList.remove("hidden");
  toast(`Loaded ${name}`);
}
$("#cfg-notice").addEventListener("click", (e) => e.currentTarget.classList.add("hidden"));

$$("#preview-seg button").forEach((b) => b.addEventListener("click", () => {
  $$("#preview-seg button").forEach((x) => x.classList.toggle("active", x === b));
  $("#cmd-preview").classList.toggle("hidden", b.dataset.view !== "cmd");
  $("#cfg-preview").classList.toggle("hidden", b.dataset.view !== "config");
}));
$("#preview-copy").addEventListener("click", async () => {
  const cfg = !$("#cfg-preview").classList.contains("hidden");
  const text = (cfg ? $("#cfg-preview") : $("#cmd-preview")).textContent;
  try { await navigator.clipboard.writeText(text); toast(cfg ? "Config copied" : "Command copied"); }
  catch (e) { toast("Copy failed: select the text instead", true); }
});

// ------------------------------------------------------------------ models & environments
const STATUS_TXT = { ok: "✓ installed", missing: "✗ not installed", broken: "⚠ installed, fails to import",
  no_weights: "⚠ installed, weights missing", no_mars: "⚠ installed, but MARS is not installed in this environment" };
const famOf = (pot) => (S.envs ? S.envs.families.find((f) => f.potentials.includes(pot)) : null);
const envStatus = (env, famId) => (env.check && env.check.families ? (env.check.families[famId] || {}).status : null);

async function loadEnvs() {
  try { S.envs = await api("/api/envs"); } catch (e) { toast(e.message, true); return; }
  renderEnvSelect(false);
  if ($("#tab-models").classList.contains("active")) renderModels();
}
async function loadModels() { await loadEnvs(); renderModels(); }

function renderEnvSelect(autoPick) {
  const sel = $("#env-select");
  if (!sel || !S.envs) return;
  const fam = famOf(S.potential);
  const prev = sel.value;
  sel.innerHTML = "";
  let best = null;
  for (const e of S.envs.envs) {
    const st = fam ? envStatus(e, fam.id) : "ok";
    const mark = st === "ok" ? "✓" : st ? (st === "missing" ? "✗" : "⚠") : "…";
    sel.append(el("option", { value: e.python, text: `${mark} ${e.name}${e.default ? " (default)" : ""}` }));
    if (st === "ok" && (best === null || e.default)) best = e.python;
  }
  sel.value = autoPick && best ? best : (prev || best || S.envs.default);
  updateEnvHint();
}
function updateEnvHint() {
  const sel = $("#env-select"), fam = famOf(S.potential), hint = $("#env-hint");
  if (!sel || !S.envs || !hint) return;
  const env = S.envs.envs.find((e) => e.python === sel.value);
  const st = env && fam ? envStatus(env, fam.id) : null;
  hint.innerHTML = "";
  if (!fam || !env) return;
  if (st === "ok") {
    hint.append(`${fam.label} is installed in this environment.`);
    const v = env.check && env.check.mars;
    if (v && S.version && v !== S.version) {
      hint.classList.add("warn-text");
      hint.append(` It has MARS ${v} (this GUI: ${S.version}); some options may differ.`);
      return;
    }
  }
  else if (!st) hint.append("Checking this environment…");
  else {
    hint.classList.add("warn-text");
    hint.append(`${fam.label}: ${STATUS_TXT[st]} here. `,
      el("a", { href: "#models", class: "link", text: "Install it", onclick: (ev) => { ev.preventDefault(); showTab("models"); } }),
      " or pick another environment.");
    return;
  }
  hint.classList.remove("warn-text");
}
$("#env-select").addEventListener("change", () => { updateEnvHint(); schedulePreview(); });

function renderModels() {
  const body = $("#models-body");
  if (!S.envs) { body.textContent = "Loading…"; return; }
  body.className = "";
  body.innerHTML = "";
  const envs = S.envs.envs;
  const head = el("tr", {}, el("th", { text: "Model" }),
    ...envs.map((e) => el("th", { class: "env-col", title: e.python },
      el("div", { text: e.name }), el("div", { class: "muted small", text: e.default ? "default" : e.source }))));
  const rows = S.envs.families.map((f) => el("tr", {},
    el("td", {}, el("b", { text: f.label }), el("div", { class: "muted small", text: f.potentials.join(", ") })),
    ...envs.map((e) => {
      const st = envStatus(e, f.id);
      const fam = e.check && e.check.families ? e.check.families[f.id] : null;
      const tip = fam ? [fam.missing.length ? "missing: " + fam.missing.join(", ") : "",
        Object.entries(fam.errors || {}).map(([m, err]) => `${m}: ${err}`).join("\n"),
        fam.missing_weights && fam.missing_weights.length ? "weights missing: " + fam.missing_weights.join(", ") : ""]
        .filter(Boolean).join("\n") : "not checked yet";
      return el("td", { class: `cell ${st || "unknown"}`, title: tip || STATUS_TXT[st] || "",
        text: st === "ok" ? "✓" : st === "missing" ? "✗" : st === "no_mars" ? "–" : st ? "⚠" : "…" });
    })));
  body.append(el("div", { class: "card-sub" },
    el("h3", { text: "What is installed where" }),
    el("p", { class: "muted small", text: "✓ ready · ✗ not installed · ⚠ installed but failing (hover for details) · – installed, but MARS is missing in that environment · … not checked yet. " +
      "Some models cannot share an environment (dxtb conflicts with parts of the JAX stack), so keep several." }),
    el("div", { class: "table-wrap" }, el("table", { class: "data matrix" }, el("thead", {}, head), el("tbody", {}, ...rows)))));

  const envCards = envs.map((e) => {
    const c = e.check || {};
    return el("div", { class: "env-card" + (e.default ? " default" : "") },
      el("div", { class: "j-top" }, el("b", { text: e.name }), el("span", { class: "muted small", text: e.source })),
      el("code", { class: "small", text: e.python }),
      el("div", { class: "muted small", text: c.error ? `⚠ ${c.error}` : c.python_version ? `Python ${c.python_version} · MARS ${c.mars || "not installed"}` +
        (c.devices ? ` · ${c.devices.join(", ")}` : c.devices_error ? ` · JAX: ${c.devices_error}` : "") : "not checked yet" }),
      el("div", { class: "btn-row" },
        el("button", { class: "btn small", text: "Verify", title: "Import every model and list the JAX devices",
          onclick: async (ev) => { ev.target.disabled = true; ev.target.textContent = "Verifying…";
            try { await api("/api/envs/check", { method: "POST", body: { python: e.python, full: true } }); } catch (x) { toast(x.message, true); }
            await loadEnvs(); renderModels(); } }),
        e.default ? null : el("button", { class: "btn small ghost", text: "Make default",
          onclick: async () => { await api("/api/envs/default", { method: "POST", body: { python: e.python } }); await loadModels(); } })));
  });
  const addInput = el("input", { type: "text", placeholder: "/path/to/env/bin/python or the environment folder", class: "mono" });
  body.append(el("div", { class: "card-sub" }, el("h3", { text: "Environments" }),
    el("div", { class: "env-grid" }, ...envCards),
    el("div", { class: "row" }, el("label", { class: "field grow", text: "Add an environment" }, addInput),
      el("button", { class: "btn", text: "Add", onclick: async () => {
        try { await api("/api/envs/add", { method: "POST", body: { python: addInput.value } }); await loadModels(); }
        catch (x) { toast(x.message, true); } } })),
    el("details", {}, el("summary", { text: "Create a new environment (commands)" }),
      el("pre", { class: "log", text: "conda create -n mars-dxtb python=3.12\nconda activate mars-dxtb\n" +
        "# then add it above and install models into it from this page" }))));

  const famCards = S.envs.families.map((f) => {
    const envSel = el("select", {}, ...envs.map((e) => el("option", { value: e.python, text: e.name + (e.default ? " (default)" : "") })));
    envSel.value = S.envs.default;
    const jaxSel = el("select", {}, ...S.envs.jax_choices.map((j) => el("option", { value: j, text: { cpu: "CPU only", cuda12: "NVIDIA, CUDA 12", cuda13: "NVIDIA, CUDA 13" }[j] || j })));
    const logBox = el("pre", { class: "log hidden" });
    const jaxSpec = { cpu: "jax<0.11", cuda12: "jax[cuda12]<0.11", cuda13: "jax[cuda13]<0.11" };
    const cmdText = () => f.install.join("\n").replace("{jax}", jaxSpec[jaxSel.value] || jaxSel.value);
    const cmdBox = el("pre", { class: "cmds" });
    jaxSel.value = "cuda12";
    cmdBox.textContent = cmdText();
    jaxSel.addEventListener("change", () => { cmdBox.textContent = cmdText(); });
    return el("div", { class: "family" },
      el("div", { class: "j-top" }, el("b", { text: f.label }), el("span", { class: "muted small", text: f.potentials.join(", ") })),
      el("p", { class: "muted small", text: f.about }),
      cmdBox,
      f.manual ? el("ul", { class: "small" }, ...f.manual.map((m) => el("li", { text: "Manual step: " + m }))) : null,
      el("div", { class: "row" },
        el("label", { class: "field", text: "Environment" }, envSel),
        f.id === "core" ? el("label", { class: "field", text: "JAX build" }, jaxSel) : null,
        el("button", { class: "btn small", text: "Copy commands", onclick: () => navigator.clipboard.writeText(cmdText()).then(() => toast("Commands copied")) }),
        el("button", { class: "btn small primary", text: "Install", disabled: S.demo || undefined,
          title: S.demo ? "Not available in the demo" : undefined, onclick: async () => {
          const env = envs.find((x) => x.python === envSel.value);
          if (!confirm(`Install ${f.label} into "${env.name}"?\n\nThese commands run with ${env.python}:\n\n${cmdText()}`)) return;
          try {
            const rec = await api("/api/envs/install", { method: "POST", body: { family: f.id, python: envSel.value, jax: jaxSel.value, confirm: true } });
            logBox.classList.remove("hidden");
            followInstall(rec.id, logBox);
          } catch (x) { toast(x.message, true); }
        } })),
      logBox);
  });
  body.append(el("div", { class: "card-sub" }, el("h3", { text: "Install models" }),
    el("p", { class: "muted small", text: "Each model shows the commands it needs. Copy them, or run them here in an environment of your choice (you will be asked to confirm)." }),
    ...famCards));
}

async function followInstall(id, box) {
  for (;;) {
    let r;
    try { r = await api(`/api/envs/installs/${encodeURIComponent(id)}`); } catch (e) { box.textContent = e.message; return; }
    box.textContent = r.log;
    box.scrollTop = box.scrollHeight;
    if (r.status !== "running") { toast(r.status === "done" ? "Installation finished" : "Installation failed: see the log", r.status !== "done"); await loadEnvs(); return; }
    await new Promise((res) => setTimeout(res, 1500));
  }
}
["#charge", "#spin", "#device", "#float64", "#advanced", "#job-name"].forEach((s) => {
  $(s).addEventListener("input", schedulePreview);
  $(s).addEventListener("change", schedulePreview);
});

function buildPayload() {
  return {
    task: S.task,
    structure: S.structure ? S.structure.id : null,
    options: readForm($("#task-form")),
    potential: S.potential,
    potential_options: readForm($("#potential-form")),
    charge: parseFloat($("#charge").value || "0") || 0,
    spin: parseFloat($("#spin").value || "1") || 1,
    float64: $("#float64").checked,
    device: $("#device").value,
    python: $("#env-select").value || undefined,
    advanced: $("#advanced").value,
    name: $("#job-name").value,
    input_name: S.structure ? S.structure.run_name || S.structure.filename : null,
  };
}

let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updatePreview, 200);
}
async function updatePreview() {
  if (!S.schema) return;
  const err = $("#form-error");
  try {
    const res = await api("/api/preview", { method: "POST", body: buildPayload() });
    const inputName = S.structure ? S.structure.run_name || S.structure.filename : "<input>";
    $("#cmd-preview").textContent = res.command.replace("input.xyz", inputName);
    $("#cfg-preview").textContent = res.config;
    err.classList.add("hidden");
  } catch (e) {
    err.textContent = e.message;
    err.classList.remove("hidden");
  }
  updateRunState();
}
function updateRunState() {
  $("#run-btn").disabled = !S.structure;
  $("#run-btn").title = S.structure ? "" : "Load a structure first";
}

$("#run-btn").addEventListener("click", async () => {
  const btn = $("#run-btn");
  btn.disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: buildPayload() });
    toast(`Job ${job.name} queued`);
    S.selectedJob = job.id;
    S.jobDetail = null;
    showTab("jobs");
  } catch (e) {
    $("#form-error").textContent = e.message;
    $("#form-error").classList.remove("hidden");
  } finally {
    updateRunState();
  }
});

// ------------------------------------------------------------------ jobs
function elapsed(job) {
  if (!job.started) return job.status === "queued" ? "waiting" : "";
  const end = job.finished || (Date.now() / 1000 + S.skew);
  return fmtDuration(end - job.started);
}

async function refreshJobs() {
  try {
    const res = await api("/api/jobs");
    S.skew = res.now - Date.now() / 1000;
    S.jobs = res.jobs;
  } catch (e) { return; }
  renderJobList();
  renderHomeJobs();
  if (S.selectedJob) {
    const j = S.jobs.find((x) => x.id === S.selectedJob);
    if (!j) { S.selectedJob = null; renderJobDetail(); return; }
    const d = S.jobDetail;
    const active = j.status === "running" || j.status === "queued";
    if (!d || d.id !== j.id || d.status !== j.status || active) await loadJobDetail(j.id);
  } else {
    renderJobDetail();
  }
}

function renderJobList() {
  const ul = $("#job-list");
  ul.innerHTML = "";
  $("#job-list-empty").classList.toggle("hidden", S.jobs.length > 0);
  const nActive = S.jobs.filter((j) => j.status === "running" || j.status === "queued").length;
  const badge = $("#jobs-badge");
  badge.textContent = nActive;
  badge.classList.toggle("hidden", nActive === 0);
  for (const j of S.jobs) {
    ul.append(el("li", {
      class: "job-item" + (j.id === S.selectedJob ? " active" : ""),
      onclick: () => { openJob(j.id); },
    },
    el("div", { class: "j-top" }, el("span", { class: "j-name", text: j.name, title: j.id }),
      el("span", { class: `status ${j.status}`, text: j.status })),
    el("div", { class: "j-sub" },
      el("span", { text: `${j.task_label} · ${j.potential}` }),
      el("span", { text: elapsed(j) }))));
  }
}

async function loadJobDetail(id) {
  try {
    const d = await api(`/api/jobs/${encodeURIComponent(id)}`);
    const prev = S.jobDetail;
    S.jobDetail = d;
    renderJobDetail(prev);
    await pollLog();
  } catch (e) {
    if (String(e.message).includes("not found")) { S.selectedJob = null; renderJobDetail(); }
  }
}

function renderJobDetail(prev) {
  const d = S.jobDetail;
  $("#job-none").classList.toggle("hidden", !!d);
  $("#job-body").classList.toggle("hidden", !d);
  if (!d) return;
  $("#job-title").textContent = d.name;
  $("#job-meta").textContent = `${d.task_label} · ${d.potential} · ${d.id} · created ${fmtTime(d.created)}` +
    (d.started ? ` · ${d.finished ? "ran" : "running"} ${elapsed(d)}` : "");
  const st = $("#job-status");
  st.className = `status ${d.status}`;
  st.textContent = d.status;
  $("#job-command").textContent = d.command;
  const active = d.status === "running" || d.status === "queued";
  $("#job-cancel").classList.toggle("hidden", !active);
  $("#job-zip").href = `/api/jobs/${encodeURIComponent(d.id)}/zip`;
  $("#log-download").href = fileUrl(d.id, "run.log", true);
  const err = $("#job-error");
  err.classList.toggle("hidden", !d.error);
  err.textContent = d.error || "";

  const sameJob = prev && prev.id === d.id;
  const filesChanged = !sameJob || JSON.stringify(prev.files.map((f) => [f.path, f.size])) !==
    JSON.stringify(d.files.map((f) => [f.path, f.size]));
  if (!sameJob) {
    S.log = { jobId: d.id, offset: 0, raw: "" };
    $("#log").textContent = "";
    S.result.key = null;
    stopPlayback();
    showSub(active ? "log" : "results");
  } else if (prev.status !== d.status && !active) {
    showSub("results");
  }
  if (filesChanged || (sameJob && prev.status !== d.status)) {
    renderFileTable(d);
    renderResultSelect(d, !sameJob || prev.status !== d.status);
  }
}

$("#refresh-jobs").addEventListener("click", refreshJobs);
$("#job-cancel").addEventListener("click", async () => {
  try { await api(`/api/jobs/${S.selectedJob}/cancel`, { method: "POST", body: {} }); toast("Cancelling…"); refreshJobs(); }
  catch (e) { toast(e.message, true); }
});
$("#job-rerun").addEventListener("click", async () => {
  try {
    const j = await api(`/api/jobs/${S.selectedJob}/rerun`, { method: "POST", body: {} });
    S.selectedJob = j.id; S.jobDetail = null; toast("Job re-queued"); refreshJobs();
  } catch (e) { toast(e.message, true); }
});
$("#job-delete").addEventListener("click", async () => {
  const d = S.jobDetail;
  if (!d || !confirm(`Delete job “${d.name}” and all its files?`)) return;
  try {
    await api(`/api/jobs/${d.id}`, { method: "DELETE" });
    S.selectedJob = null; S.jobDetail = null; toast("Job deleted"); refreshJobs();
  } catch (e) { toast(e.message, true); }
});

// ------------------------------------------------------------------ log
const ANSI16 = ["#000", "#c0392b", "#1f8a4c", "#b7791f", "#2b6cb0", "#8e44ad", "#16a085", "#bbb",
  "#666", "#e74c3c", "#2ecc71", "#f1c40f", "#3498db", "#9b59b6", "#1abc9c", "#fff"];
function xterm256(n) {
  if (n < 16) return ANSI16[n];
  if (n >= 232) { const v = 8 + 10 * (n - 232); return `rgb(${v},${v},${v})`; }
  const lv = [0, 95, 135, 175, 215, 255], k = n - 16;
  return `rgb(${lv[Math.floor(k / 36)]},${lv[Math.floor(k / 6) % 6]},${lv[k % 6]})`;
}
function ansiToHtml(text) {
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let color = null, bold = false, out = "";
  for (const part of text.replace(/\x1b\[[0-9;]*$/, "").split(/(\x1b\[[0-9;]*m)/)) {
    const m = part.match(/^\x1b\[([0-9;]*)m$/);
    if (!m) {
      if (!part) continue;
      const style = (color ? `color:${color};` : "") + (bold ? "font-weight:700;" : "");
      out += style ? `<span style="${style}">${esc(part)}</span>` : esc(part);
      continue;
    }
    const c = (m[1] || "0").split(";").map(Number);
    for (let i = 0; i < c.length; i++) {
      if (c[i] === 0) { color = null; bold = false; }
      else if (c[i] === 1) bold = true;
      else if (c[i] === 22) bold = false;
      else if (c[i] === 39) color = null;
      else if (c[i] >= 30 && c[i] <= 37) color = ANSI16[c[i] - 30];
      else if (c[i] >= 90 && c[i] <= 97) color = ANSI16[c[i] - 82];
      else if (c[i] === 38 && c[i + 1] === 2) { color = `rgb(${c[i + 2]},${c[i + 3]},${c[i + 4]})`; i += 4; }
      else if (c[i] === 38 && c[i + 1] === 5) { color = xterm256(c[i + 2]); i += 2; }
    }
  }
  return out;
}

function collapseCR(text) {
  // tqdm progress bars rewrite the line with \r: keep only the last update.
  return text.split("\n").map((ln) => {
    const k = ln.lastIndexOf("\r", ln.length - 2);
    return k >= 0 ? ln.slice(k + 1) : ln;
  }).join("\n");
}
async function pollLog() {
  const d = S.jobDetail;
  if (!d) return;
  if (S.log.jobId !== d.id) S.log = { jobId: d.id, offset: 0, raw: "" };
  let start = S.log.offset;
  if (start === 0 && !S.log.raw) start = -400000; // first load: tail
  try {
    for (let guard = 0; guard < 8; guard++) {
      const res = await api(`/api/jobs/${encodeURIComponent(d.id)}/log?offset=${start}`);
      if (S.log.jobId !== d.id) return;
      if (res.text) S.log.raw += res.text;
      S.log.offset = res.offset;
      if (res.offset >= res.size) break;
      start = res.offset;
    }
  } catch (e) { return; }
  if (S.log.raw.length > 600000) S.log.raw = S.log.raw.slice(-500000);
  const pre = $("#log");
  pre.innerHTML = ansiToHtml(collapseCR(S.log.raw));
  scrollLog();
}
function scrollLog() {
  if ($("#log-follow").checked) { const pre = $("#log"); pre.scrollTop = pre.scrollHeight; }
}

// ------------------------------------------------------------------ files
function renderFileTable(d) {
  const t = $("#file-table");
  t.innerHTML = "";
  t.append(el("thead", {}, el("tr", {}, ["File", "Kind", "Size", "Modified", ""].map((h) => el("th", { text: h })))));
  const body = el("tbody");
  for (const f of d.files) {
    const viewable = optionKeyForFile(f);
    body.append(el("tr", {},
      el("td", { class: "mono" }, el("a", { href: fileUrl(d.id, f.path), target: "_blank", rel: "noopener", text: f.path })),
      el("td", { text: f.kind }),
      el("td", { class: "num", text: fmtBytes(f.size) }),
      el("td", { text: new Date(f.mtime * 1000).toLocaleTimeString() }),
      el("td", {},
        viewable ? el("button", { class: "btn ghost small", onclick: () => { selectResult(viewable); showSub("results"); } }, "View") : null,
        el("a", { class: "btn ghost small", href: fileUrl(d.id, f.path, true) }, "Download"))));
  }
  t.append(body);
}

// ------------------------------------------------------------------ results
function irPrefix(path) {
  const k = path.lastIndexOf("ir_spectrum");
  return k >= 0 ? path.slice(0, k) : null;
}
function optionKeyForFile(f) {
  if (f.kind === "spectrum") return `ir:${f.path}`;
  if (["structure", "ensemble", "trajectory"].includes(f.kind)) return `mol:${f.path}`;
  if (f.kind === "modes") {
    const d = S.jobDetail;
    const pre = f.path.replace(/normal_modes.*$/, "");
    const spec = d && d.files.find((x) => x.kind === "spectrum" && irPrefix(x.path) === pre);
    return spec ? `ir:${spec.path}` : `modes:${f.path}`;
  }
  if (f.kind === "image") return `img:${f.path}`;
  if (["analysis", "text", "log"].includes(f.kind)) return `text:${f.path}`;
  return null;
}

function renderResultSelect(d, chooseDefault) {
  const sel = $("#result-select");
  const prevKey = S.result.key;
  sel.innerHTML = "";
  const opts = [];
  const seen = new Set();
  const order = { spectrum: 0, ensemble: 1, structure: 2, trajectory: 3, modes: 4, image: 5, analysis: 6, text: 7, log: 8 };
  const files = d.files.filter((f) => f.path !== "command.sh").slice()
    .sort((a, b) => ((a.kind in order) ? order[a.kind] : 9) - ((b.kind in order) ? order[b.kind] : 9) || a.path.localeCompare(b.path));
  for (const f of files) {
    const key = optionKeyForFile(f);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    const isInput = f.path === d.input;
    const label = key.startsWith("ir:") ? `IR spectrum — ${f.path}` :
      `${f.path} (${isInput ? "input" : f.kind})`;
    opts.push({ key, label, f });
  }
  for (const o of opts) sel.append(el("option", { value: o.key, text: o.label }));
  $("#result-empty").classList.toggle("hidden", opts.length > 0);
  if (!opts.length) { hideResultViews(); return; }
  let key = prevKey && seen.has(prevKey) && !chooseDefault ? prevKey : defaultResult(d, opts);
  sel.value = key;
  if (key !== prevKey || chooseDefault) selectResult(key, true);
}

function defaultResult(d, opts) {
  const has = (k) => opts.find((o) => o.key === k);
  const byName = (names) => { for (const n of names) { const o = has(`mol:${n}`); if (o) return o.key; } return null; };
  if (d.task === "ir") { const o = opts.find((x) => x.key.startsWith("ir:")); if (o) return o.key; }
  const preferred = {
    conformer: ["auto_final_ensemble.xyz"], mtd: ["auto_final_ensemble.xyz", "mtd_ensemble.xyz"],
    md: ["auto_final_ensemble.xyz", "md_trajectory.xyz"], optimize: ["optimized.xyz"],
    solvation: ["solvated.xyz"], ir: ["optimized_structure.xyz"],
  }[d.task] || [];
  const p = byName(preferred);
  if (p) return p;
  const nonInput = opts.find((o) => o.key.startsWith("mol:") && o.f.path !== d.input && o.f.kind !== "trajectory");
  if (nonInput) return nonInput.key;
  const anyMol = opts.find((o) => o.key.startsWith("mol:") && o.f.path !== d.input);
  return (anyMol || opts.find((o) => o.key.startsWith("mol:")) || opts[0]).key;
}

$("#result-select").addEventListener("change", (e) => selectResult(e.target.value));

function hideResultViews() {
  ["#res-mol", "#res-ir", "#res-image", "#res-text"].forEach((s) => $(s).classList.add("hidden"));
  $("#use-as-input").classList.add("hidden");
  stopPlayback();
}

async function fetchText(path) {
  const resp = await fetch(fileUrl(S.jobDetail.id, path));
  if (!resp.ok) throw new Error(`${resp.status}`);
  return resp.text();
}

async function selectResult(key, quiet = false) {
  S.result.key = key;
  $("#result-select").value = key;
  hideResultViews();
  const [kind, path] = [key.slice(0, key.indexOf(":")), key.slice(key.indexOf(":") + 1)];
  try {
    if (kind === "mol" || kind === "modes") await showMolResult(path);
    else if (kind === "ir") await showIR(path);
    else if (kind === "img") {
      $("#res-img").src = fileUrl(S.jobDetail.id, path) + `${path.includes("?") ? "&" : "?"}t=${Date.now()}`;
      $("#res-image").classList.remove("hidden");
    } else if (kind === "text") {
      const txt = await fetchText(path);
      const pre = $("#res-text");
      pre.textContent = txt.length > 800000 ? txt.slice(0, 800000) + "\n… (truncated)" : txt;
      pre.classList.remove("hidden");
    }
  } catch (e) {
    if (!quiet) toast(`Could not show ${path}: ${e.message}`, true);
  }
}

// ---- structures / ensembles / trajectories
async function showMolResult(path) {
  const text = await fetchText(path);
  const frames = parseXYZFrames(text);
  S.result.frames = frames;
  S.result.path = path;
  $("#res-mol").classList.remove("hidden");
  $("#use-as-input").classList.remove("hidden");
  setTimeout(() => resView.resize(), 10);
  if (!frames.length) { toast("No frames found", true); return; }
  const multi = frames.length > 1;
  $("#res-frames").classList.toggle("hidden", !multi);
  const r = $("#res-frame");
  r.max = frames.length - 1;
  r.value = 0;
  resView.hasModel = false;
  showResFrame(0);
  const energies = frames.map((f) => parseEnergy(f.comment));
  const isEnsemble = multi && energies.some((e) => e !== null);
  $("#res-ensemble").classList.toggle("hidden", !isEnsemble);
  $("#res-play").classList.toggle("hidden", !multi);
  if (isEnsemble) renderEnsemble(energies);
}

function showResFrame(i) {
  const fr = S.result.frames[i];
  if (!fr) return;
  resView.show(fr.text, { keepView: true });
  const e = parseEnergy(fr.comment);
  $("#res-frame-label").textContent = `${i + 1}/${S.result.frames.length}` +
    (e !== null ? ` · ${e.toFixed(3)} kcal/mol` : "") + (fr.comment && e === null ? ` · ${fr.comment.slice(0, 60)}` : "");
  $$("#energy-table tbody tr").forEach((tr) => tr.classList.toggle("sel", +tr.dataset.i === i));
}

$("#res-frame").addEventListener("input", (e) => showResFrame(+e.target.value));
bindStyleControls(resView, "#res-style-seg", "#res-labels");

function stopPlayback() {
  if (S.result.playing) { clearInterval(S.result.playing); S.result.playing = null; }
  $("#res-play").textContent = "▶ Play";
}
$("#res-play").addEventListener("click", () => {
  if (S.result.playing) { stopPlayback(); return; }
  const fps = Math.max(1, Math.min(60, +$("#res-fps").value || 10));
  $("#res-play").textContent = "❚❚ Pause";
  S.result.playing = setInterval(() => {
    const r = $("#res-frame");
    const n = S.result.frames.length;
    const next = (+r.value + 1) % n;
    r.value = next;
    showResFrame(next);
  }, 1000 / fps);
});

function renderEnsemble(energies) {
  const valid = energies.filter((e) => e !== null);
  const emin = Math.min(...valid);
  const RT = 0.0019872043 * 298.15;
  const rel = energies.map((e) => (e === null ? null : e - emin));
  const w = rel.map((d) => (d === null ? 0 : Math.exp(-d / RT)));
  const Z = w.reduce((a, b) => a + b, 0) || 1;
  const t = $("#energy-table");
  t.innerHTML = "";
  t.append(el("thead", {}, el("tr", {}, ["#", "E (kcal/mol)", "ΔE (kcal/mol)", "Pop. 298 K (%)"].map((h) => el("th", { text: h })))));
  const body = el("tbody");
  energies.forEach((e, i) => {
    body.append(el("tr", { class: "clickable", "data-i": i, onclick: () => { $("#res-frame").value = i; showResFrame(i); } },
      el("td", { class: "num", text: i + 1 }),
      el("td", { class: "num", text: e === null ? "—" : e.toFixed(4) }),
      el("td", { class: "num", text: rel[i] === null ? "—" : rel[i].toFixed(3) }),
      el("td", { class: "num", text: ((100 * w[i]) / Z).toFixed(1) })));
  });
  t.append(body);
  if (body.firstChild) body.firstChild.classList.add("sel");
  if (!HAS_PLOTLY()) return;
  const node = $("#energy-plot");
  Plotly.newPlot(node, [{
    type: "bar", x: rel.map((_, i) => i + 1), y: rel, marker: { color: cssVar("--accent") },
    hovertemplate: "conformer %{x}<br>ΔE = %{y:.3f} kcal/mol<extra></extra>",
  }], basePlotLayout({
    xaxis: { title: "Conformer", gridcolor: cssVar("--border") },
    yaxis: { title: "ΔE (kcal/mol)", gridcolor: cssVar("--border") },
    margin: { l: 60, r: 20, t: 10, b: 45 },
  }), { responsive: true, displaylogo: false });
  node.removeAllListeners && node.removeAllListeners("plotly_click");
  node.on("plotly_click", (ev) => {
    const i = ev.points[0].x - 1;
    $("#res-frame").value = i;
    showResFrame(i);
  });
}

$("#use-as-input").addEventListener("click", async () => {
  const fr = S.result.frames;
  if (!fr || !fr.length) return;
  const i = +$("#res-frame").value || 0;
  const multi = fr.length > 1 && confirm("Use the whole multi-frame file as input?\n(Cancel = only the current frame)");
  const content = multi ? fr.map((f) => f.text).join("") : fr[i].text;
  const base = (S.result.path || "structure.xyz").split("/").pop();
  await uploadStructure(multi ? base : base.replace(/\.xyz$/, "") + (fr.length > 1 ? `_frame${i + 1}.xyz` : ".xyz"),
    { content });
  showTab("setup");
});

// ---- IR spectrum
function parseSpectrum(text) {
  const rows = [];
  let header = "";
  for (const ln of text.split(/\r?\n/)) {
    const s = ln.trim();
    if (!s) continue;
    if (s.startsWith("#")) { header += s + "\n"; continue; }
    const p = s.split(/\s+/).map(Number);
    if (p.length >= 2 && isFinite(p[0]) && isFinite(p[1])) rows.push([p[0], p[1]]);
  }
  return { rows, header };
}

function parseModes(text) {
  return parseXYZFrames(text).map((f) => {
    const m = /Freq:\s*([-+0-9.eE]+)/.exec(f.comment);
    return { freq: m ? parseFloat(m[1]) : NaN, text: f.text, comment: f.comment };
  });
}

function parseAnalysis(text) {
  // "ALL MODES" table of mode_analysis.txt (fixed-width columns).
  const lines = text.split(/\r?\n/);
  const out = {};
  const h = lines.findIndex((l) => /^Mode\s+Freq/.test(l));
  if (h < 0) return out;
  const header = lines[h];
  const cType = header.indexOf("Type"), cLabel = header.indexOf("Label"), cGroup = header.indexOf("Functional");
  for (let i = h + 1; i < lines.length; i++) {
    const l = lines[i];
    if (/^-{5,}/.test(l)) { if (i > h + 1 && Object.keys(out).length) break; continue; }
    if (/^={5,}/.test(l) || !l.trim()) { if (Object.keys(out).length) break; continue; }
    const m = /^\s*(\d+)\s/.exec(l);
    if (!m) continue;
    out[+m[1]] = {
      type: cType >= 0 ? l.slice(cType, cLabel > 0 ? cLabel : undefined).trim() : "",
      label: cLabel >= 0 ? l.slice(cLabel, cGroup > 0 ? cGroup : undefined).trim() : "",
      group: cGroup >= 0 ? l.slice(cGroup).trim() : "",
    };
  }
  return out;
}

async function showIR(specPath) {
  const d = S.jobDetail;
  const pre = irPrefix(specPath) || "";
  const modesFile = d.files.find((f) => f.kind === "modes" && f.path.startsWith(pre) && f.path.slice(pre.length).startsWith("normal_modes"));
  const anaFile = d.files.find((f) => f.kind === "analysis" && f.path.startsWith(pre));
  const [specText, modesText, anaText] = await Promise.all([
    fetchText(specPath),
    modesFile ? fetchText(modesFile.path) : Promise.resolve(null),
    anaFile ? fetchText(anaFile.path) : Promise.resolve(null),
  ]);
  const spec = parseSpectrum(specText);
  const modes = modesText ? parseModes(modesText) : [];
  const analysis = anaText ? parseAnalysis(anaText) : {};
  const opts = (d.payload && d.payload.options) || {};
  let range = [400, 4000];
  const fr = opts["--freq-range"];
  if (fr) {
    const p = String(Array.isArray(fr) ? fr.join(" ") : fr).split(/[\s,]+/).map(Number).filter(isFinite);
    if (p.length === 2) range = [Math.min(...p), Math.max(...p)];
  }
  if (opts["--broadening"]) $("#ir-width").value = opts["--broadening"];
  const md = !!opts["--md"] || /MD-based/.test(spec.header);
  const harmonic = !md && (modes.length > 0 || spec.rows.length < 600);
  S.ir = { spec, modes, analysis, range, harmonic, selected: null };
  $("#res-ir").classList.remove("hidden");
  const animated = harmonic && modes.length > 0;
  $("#mode-empty").classList.toggle("hidden", animated);
  $("#mode-amp").closest("label").classList.toggle("hidden", !animated);
  $("#ir-sticks").closest("label").classList.toggle("hidden", !harmonic);
  renderModeTable();
  drawIR();
  if (!animated) await showStaticIRStructure(d, md);
  // Select the most intense real mode by default.
  if (harmonic && spec.rows.length) {
    let best = -1, bi = -1;
    spec.rows.forEach(([f, I], i) => { if (f > 50 && I > best) { best = I; bi = i; } });
    if (bi >= 0) selectMode(bi);
  }
  setTimeout(() => { modeView.resize(); if (HAS_PLOTLY()) Plotly.Plots.resize($("#ir-plot")); }, 30);
}

function firstFrame(text) {
  const lines = text.split(/\r?\n/);
  const n = parseInt(lines[0], 10);
  return Number.isFinite(n) ? lines.slice(0, n + 2).join("\n") + "\n" : text;
}

async function showStaticIRStructure(d, md) {
  const optimized = d.files.find((f) => f.kind === "structure" && /optimi[sz]ed/.test(f.path));
  const input = d.files.find((f) => f.path === d.input);
  const src = md ? (input || optimized) : (optimized || input);
  if (!src) { modeView.clear(); return; }
  modeView.show(firstFrame(await fetchText(src.path)));
  $("#mode-empty").classList.add("hidden");
  $("#mode-label").textContent = md ? `initial structure · ${src.path}` : `structure · ${src.path} (no normal modes saved)`;
}

function drawIR() {
  const ir = S.ir;
  if (!ir || !HAS_PLOTLY()) return;
  const node = $("#ir-plot");
  const invert = $("#ir-invert").checked;
  const traces = [];
  const acc = cssVar("--accent"), mut = cssVar("--muted");
  let yTitle = "Intensity (normalised)";
  if (ir.harmonic) {
    const peaks = ir.spec.rows.map(([f, I], i) => ({ f, I, i })).filter((p) => p.f > 1);
    const fwhm = Math.max(0.5, +$("#ir-width").value || 10);
    const g = fwhm / 2;
    const xmax = Math.max(ir.range[1], ...peaks.map((p) => p.f + 5 * fwhm));
    const xs = [], ys = [];
    for (let x = 0; x <= xmax; x += 1) {
      let s = 0;
      for (const p of peaks) s += p.I * (g * g) / ((x - p.f) * (x - p.f) + g * g);
      xs.push(x); ys.push(s);
    }
    const ymax = Math.max(...ys, 1e-12);
    const yb = ys.map((y) => y / ymax);
    traces.push({
      x: xs, y: invert ? yb.map((y) => 1 - y) : yb, mode: "lines", line: { color: acc, width: 2 },
      hoverinfo: "skip", name: "broadened",
    });
    if ($("#ir-sticks").checked) {
      const imax = Math.max(...peaks.map((p) => p.I), 1e-12);
      const sx = [], sy = [], cd = [];
      for (const p of peaks) {
        const h = p.I / imax;
        sx.push(p.f, p.f, null);
        sy.push(invert ? 1 : 0, invert ? 1 - h : h, null);
        cd.push(p.i, p.i, null);
      }
      traces.push({
        x: sx, y: sy, customdata: cd, mode: "lines+markers", line: { color: mut, width: 1.5 },
        marker: { size: 5, color: mut }, name: "modes",
        hovertemplate: "mode %{customdata}<br>%{x:.1f} cm⁻¹<extra></extra>",
      });
    }
    if (ir.selected !== null && ir.spec.rows[ir.selected]) {
      const [f] = ir.spec.rows[ir.selected];
      traces.push({ x: [f, f], y: [0, 1], mode: "lines", line: { color: acc, width: 1, dash: "dot" }, hoverinfo: "skip" });
    }
  } else {
    const xs = ir.spec.rows.map((r) => r[0]), ys = ir.spec.rows.map((r) => r[1]);
    const ymax = Math.max(...ys, 1e-12);
    traces.push({ x: xs, y: ys.map((y) => (invert ? 1 - y / ymax : y / ymax)), mode: "lines", line: { color: acc, width: 1.8 } });
    yTitle = "Intensity (MD, normalised)";
  }
  const layout = basePlotLayout({
    xaxis: { title: "Wavenumber (cm⁻¹)", range: [ir.range[1], ir.range[0]], gridcolor: cssVar("--border"), zeroline: false },
    yaxis: { title: invert ? "Transmittance (arb.)" : yTitle, gridcolor: cssVar("--border"), zeroline: false, rangemode: "tozero" },
  });
  Plotly.react(node, traces, layout, { responsive: true, displaylogo: false });
  if (!node._marsClick) {
    node._marsClick = true;
    node.on("plotly_click", (ev) => {
      const pt = ev.points.find((p) => p.customdata !== undefined && p.customdata !== null);
      if (pt) { selectMode(pt.customdata); return; }
      if (S.ir && S.ir.harmonic) { // nearest mode to the clicked wavenumber
        const x = ev.points[0].x;
        let bi = -1, bd = Infinity;
        S.ir.spec.rows.forEach(([f], i) => { if (f > 1 && Math.abs(f - x) < bd) { bd = Math.abs(f - x); bi = i; } });
        if (bi >= 0) selectMode(bi);
      }
    });
  }
}
["#ir-width", "#ir-sticks", "#ir-invert"].forEach((s) => $(s).addEventListener("input", drawIR));

function renderModeTable() {
  const ir = S.ir;
  const t = $("#mode-table");
  t.innerHTML = "";
  if (!ir.harmonic) {
    t.append(el("tbody", {}, el("tr", {}, el("td", { class: "muted", text: "MD-based spectrum (no normal modes)." }))));
    return;
  }
  const imax = Math.max(...ir.spec.rows.map((r) => r[1]), 1e-12);
  t.append(el("thead", {}, el("tr", {}, ["Mode", "ν (cm⁻¹)", "Int.", "Type", "Assignment"].map((h) => el("th", { text: h })))));
  const body = el("tbody");
  const n = Math.max(ir.spec.rows.length, ir.modes.length);
  for (let i = 0; i < n; i++) {
    const row = ir.spec.rows[i];
    const f = row ? row[0] : ir.modes[i].freq;
    const a = ir.analysis[i] || {};
    const fTxt = f < 0 ? `${Math.abs(f).toFixed(1)}i` : f.toFixed(1);
    body.append(el("tr", { "data-i": i, onclick: () => selectMode(i) },
      el("td", { class: "num", text: i }),
      el("td", { class: "num", text: fTxt }),
      el("td", { class: "num", text: row ? (row[1] / imax).toFixed(3) : "" }),
      el("td", { text: a.type || "" }),
      el("td", { text: [a.label, a.group].filter(Boolean).join(" · ") })));
  }
  t.append(body);
}

function selectMode(i) {
  const ir = S.ir;
  if (!ir) return;
  ir.selected = i;
  $$("#mode-table tbody tr").forEach((tr) => {
    const on = +tr.dataset.i === i;
    tr.classList.toggle("sel", on);
    if (on) {
      const box = tr.closest(".table-wrap");
      if (box) {
        const top = tr.offsetTop - box.querySelector("thead").offsetHeight, bottom = tr.offsetTop + tr.offsetHeight;
        if (top < box.scrollTop) box.scrollTop = top;
        else if (bottom > box.scrollTop + box.clientHeight) box.scrollTop = bottom - box.clientHeight;
      }
    }
  });
  const row = ir.spec.rows[i];
  const mode = ir.modes[i];
  const f = row ? row[0] : mode ? mode.freq : NaN;
  const a = ir.analysis[i] || {};
  $("#mode-label").textContent = `mode ${i} · ${f < 0 ? Math.abs(f).toFixed(1) + "i" : f.toFixed(1)} cm⁻¹` +
    (a.label ? ` · ${a.label}` : a.type ? ` · ${a.type}` : "");
  if (mode) modeView.vibrate(mode.text, +$("#mode-amp").value);
  drawIR();
}
$("#mode-amp").addEventListener("change", () => { if (S.ir && S.ir.selected !== null) selectMode(S.ir.selected); });

// ------------------------------------------------------------------ init
async function init() {
  let saved = null;
  try { saved = localStorage.getItem("mars-gui-theme"); } catch (e) { /* ignore */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-select").value = saved || "";
  $("#theme-select").addEventListener("change", (e) => applyTheme(e.target.value));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(null));
  if (!HAS_3DMOL() || !HAS_PLOTLY()) $("#cdn-warning").classList.remove("hidden");
  window.addEventListener("resize", () => MolView.all.forEach((v) => v.resize()));

  try {
    const [info, schema] = await Promise.all([api("/api/info"), api("/api/schema")]);
    $("#workdir").textContent = info.workdir;
    $("#workdir").title = `Job directory · MARS ${info.version} · ${info.python}`;
    $("#version").textContent = `MARS ${info.version}`;
    S.version = info.version;
    S.demo = !!info.demo;
    if (S.demo) { $("#demo-banner").classList.remove("hidden"); document.body.classList.add("demo"); }
    $("#about-version").textContent = `MARS ${info.version} · ${info.python}`;
    $("#settings-workdir").textContent = info.workdir;
    $("#settings-python").textContent = info.python;
    addPlugins(info.plugins);
    S.schema = schema;
  } catch (e) {
    toast(`Cannot reach the MARS GUI server: ${e.message}`, true);
    return;
  }
  const potSel = $("#potential");
  for (const p of S.schema.potentials) potSel.append(el("option", { value: p.name, text: p.label }));
  potSel.value = S.potential = S.schema.default_potential;
  renderTasks();
  updateRunState();
  loadEnvs();
  setInterval(() => { if (!S.envs || S.envs.envs.some((e) => !e.check)) loadEnvs(); }, 5000);
  const [start, arg, sub] = location.hash.slice(1).split("/");
  if (start && document.getElementById(`tab-${start}`)) showTab(start);
  if (start === "jobs" && arg) {
    openJob(decodeURIComponent(arg));
    if (sub && document.getElementById(`sub-${sub}`)) setTimeout(() => showSub(sub), 300);
  }
  await refreshJobs();
  setInterval(async () => {
    await refreshJobs();
  }, 2000);
}
init();
