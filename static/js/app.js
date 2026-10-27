// DataLens single-page application.
import { api } from "./api.js";
import * as charts from "./charts.js";
import {
  correlationColor,
  escapeHtml as esc,
  formatBytes,
  formatCell,
  formatMetric,
  formatNumber,
  formatPercent,
  formatPValue,
  formatSpread,
  metricLabel,
  mixColor,
  pluralize,
  textOn,
  toCsv,
} from "./format.js";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const MODELABLE = new Set(["numeric", "categorical", "boolean"]);

const state = {
  config: null,
  id: null,
  dataset: null,
  profile: null,
  insights: [],
  summary: "",
  tab: "overview",
  rendered: new Set(),
  preview: { offset: 0, limit: 25, sort: null, order: "asc" },
  column: null,
  corrMethod: "pearson",
  stats: null,
  model: null,
  sqlResult: null,
  lastFile: null, // { id, file } for transparent re-upload
  lastTrainOptions: null,
};

// ------------------------------------------------------------------ utilities

let toastTimer;
function toast(message, { error = false } = {}) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("error", error);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, error ? 6000 : 3000);
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.label = button.textContent;
    button.disabled = true;
    button.textContent = label ?? "Working…";
  } else {
    button.disabled = false;
    button.textContent = button.dataset.label ?? button.textContent;
  }
}

function typeTag(type) {
  return `<span class="tag type-${esc(type)}">${esc(type)}</span>`;
}

function kpi(label, value, sub = "", accent = false) {
  return `<div class="kpi${accent ? " accent" : ""}">
    <div class="kpi-label">${esc(label)}</div>
    <div class="kpi-value">${esc(value)}</div>
    ${sub ? `<div class="kpi-sub">${esc(sub)}</div>` : ""}
  </div>`;
}

function findingsHtml(findings) {
  if (!findings?.length) return `<p class="muted">No notable findings.</p>`;
  return findings.map((f) => `<div class="finding">
      <div class="finding-title">${esc(f.title)}</div>
      <div class="finding-detail">${esc(f.detail)}</div>
    </div>`).join("");
}

function tableHtml(columns, rows, { numeric = [], className = "data", rowClass } = {}) {
  const numSet = new Set(numeric);
  const head = columns.map((c, i) => `<th class="${numSet.has(i) ? "num" : ""}">${esc(c)}</th>`).join("");
  const body = rows.map((row, r) => `<tr class="${rowClass ? rowClass(r) : ""}">${row.map((v, i) => {
    if (v === null || v === undefined) return `<td class="null">null</td>`;
    return `<td class="${numSet.has(i) ? "num" : ""}">${typeof v === "object" && v.html ? v.html : esc(v)}</td>`;
  }).join("")}</tr>`).join("");
  return `<table class="${className}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

const raw = (html) => ({ html });

function bar(pct, { neg = false } = {}) {
  const width = Math.max(0, Math.min(100, pct));
  return `<div class="bar${neg ? " neg" : ""}"><span style="width:${width.toFixed(1)}%"></span></div>`;
}

function download(filename, content, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function columnsOfType(...types) {
  const set = new Set(types);
  return state.profile.columns.filter((c) => set.has(c.semantic_type));
}

// ------------------------------------------------------------------ serverless recovery
// On serverless hosts a request can reach an instance that has never seen this dataset or model.
// Samples are rebuilt by the server; uploads are re-sent and models retrained with the same options.

async function recover(err) {
  if (/Dataset not found/.test(err.message) && state.lastFile?.id === state.id) {
    await api.upload(state.lastFile.file);
    return true;
  }
  if (/No model has been trained|Train a model before/.test(err.message) && state.lastTrainOptions) {
    await api.train(state.id, state.lastTrainOptions);
    return true;
  }
  return false;
}

async function withRecovery(call, attempts = 2) {
  try {
    return await call();
  } catch (err) {
    if (attempts > 0 && (await recover(err))) return withRecovery(call, attempts - 1);
    throw err;
  }
}

async function ensureServerState() {
  const payload = await withRecovery(() => api.dataset(state.id));
  if (state.lastTrainOptions && !payload.dataset.has_model) await api.train(state.id, state.lastTrainOptions);
}

function optionsFromSummary(summary) {
  return {
    target: summary.target,
    task: summary.task,
    models: summary.leaderboard.filter((r) => !r.baseline).map((r) => r.key),
    cv_folds: summary.cv_folds,
    tune: Boolean(summary.tuning),
  };
}

// ------------------------------------------------------------------ routing

function setHash(tab) {
  const hash = state.id ? `#/d/${state.id}/${tab}` : "";
  if (location.hash !== hash) history.replaceState(null, "", hash || location.pathname);
}

async function route() {
  const match = location.hash.match(/^#\/d\/([\w-]+)(?:\/(\w+))?/);
  if (!match) return showHome();
  if (match[1] === state.id) return setTab(match[2] || "overview");
  try {
    const payload = await api.dataset(match[1]);
    openWorkspace(payload, match[2] || "overview");
  } catch {
    history.replaceState(null, "", location.pathname);
    showHome();
  }
}

// ------------------------------------------------------------------ home

function showHome() {
  charts.destroyAll();
  state.id = null;
  state.profile = null;
  $("#view-home").hidden = false;
  $("#view-workspace").hidden = true;
  $("#topbar-context").hidden = true;
  $("#new-dataset-btn").hidden = true;
  $("#upload-progress").hidden = true;
  setHash();
  document.title = "DataLens · Analytics Workbench";
}

async function loadSamples() {
  const container = $("#samples");
  try {
    const { samples } = await api.samples();
    if (!samples.length) {
      container.innerHTML = `<p class="muted small">No sample files found. Run <code>python scripts/generate_samples.py</code>.</p>`;
      return;
    }
    container.innerHTML = samples.map((s) => `
      <button class="sample" data-sample="${esc(s.key)}">
        <span class="sample-title">${esc(s.title)}</span>
        <span class="sample-desc">${esc(s.description)}</span>
        <span class="tag">${esc(s.task)}</span>
      </button>`).join("");
  } catch (err) {
    container.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  }
}

async function ingest(promise, label) {
  $("#upload-progress").hidden = false;
  $("#upload-status").textContent = label;
  $$(".sample").forEach((b) => { b.disabled = true; });
  try {
    const payload = await promise;
    openWorkspace(payload, "overview");
  } catch (err) {
    toast(err.message, { error: true });
  } finally {
    $("#upload-progress").hidden = true;
    $$(".sample").forEach((b) => { b.disabled = false; });
  }
}

function handleFile(file) {
  if (!file) return;
  const limit = state.config?.max_upload_mb ?? 50;
  if (file.size > limit * 1024 * 1024) {
    toast(`File is larger than the ${limit} MB limit.`, { error: true });
    return;
  }
  ingest(api.upload(file).then((payload) => {
    state.lastFile = { id: payload.dataset.id, file };
    return payload;
  }), `Profiling ${file.name}…`);
}

// ------------------------------------------------------------------ workspace

function openWorkspace(payload, tab) {
  charts.destroyAll();
  Object.assign(state, {
    id: payload.dataset.id,
    dataset: payload.dataset,
    profile: payload.profile,
    insights: payload.insights,
    summary: payload.summary,
    rendered: new Set(),
    preview: { offset: 0, limit: 25, sort: null, order: "asc" },
    column: null,
    stats: null,
    model: null,
    sqlResult: null,
    lastTrainOptions: null,
  });
  const d = payload.dataset;
  $("#view-home").hidden = true;
  $("#view-workspace").hidden = false;
  $("#topbar-context").hidden = false;
  $("#new-dataset-btn").hidden = false;
  $("#topbar-dataset").textContent = d.name;
  $("#ws-source").textContent = d.source === "sample" ? "Sample dataset" : "Uploaded dataset";
  $("#ws-title").textContent = d.name;
  $("#ws-meta").textContent = `${pluralize(d.rows, "row")} · ${pluralize(d.columns, "column")} · `
    + `${formatBytes(payload.profile.overview.memory_bytes)} in memory`;
  $("#export-csv").href = api.exportUrl(d.id);
  $("#report-download").href = api.reportUrl(d.id, true);
  document.title = `${d.name} · DataLens`;
  setupTargetSelects();
  $("#stats-output").innerHTML = `<div class="empty">Choose a target and run the tests to see which features are associated with it.</div>`;
  $("#model-output").innerHTML = `<div class="empty">Configure the run above. Every model is cross-validated and compared against a naive baseline.</div>`;
  $("#sql-result").innerHTML = `<div class="empty">Run a query to see results.</div>`;
  $("#sql-meta").textContent = "";
  $("#sql-download").disabled = true;
  window.scrollTo({ top: 0 });
  setTab(tab);
}

function setTab(tab) {
  if (!state.profile) return;
  if (!$(`[data-panel="${tab}"]`)) tab = "overview";
  state.tab = tab;
  $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === tab)));
  $$(".panel").forEach((p) => { p.hidden = p.dataset.panel !== tab; });
  setHash(tab);
  const renderers = {
    overview: renderOverview,
    columns: renderColumns,
    correlations: renderCorrelations,
    statistics: () => {},
    model: restoreModel,
    sql: initSql,
    report: renderReport,
  };
  // Charts need a visible canvas to size correctly, so panels render on first view.
  if (!state.rendered.has(tab) || tab === "report") {
    state.rendered.add(tab);
    renderers[tab]();
  }
}

// ------------------------------------------------------------------ overview

function renderOverview() {
  const { overview: ov, quality: q, alerts, missing } = state.profile;
  $("#overview-kpis").innerHTML = [
    kpi("Rows", formatNumber(ov.rows)),
    kpi("Columns", formatNumber(ov.columns), Object.entries(ov.type_counts).map(([t, n]) => `${n} ${t}`).join(" · ")),
    kpi("Missing cells", formatPercent(ov.missing_pct), `${formatNumber(ov.missing_cells)} cells`),
    kpi("Duplicate rows", formatNumber(ov.duplicate_rows), formatPercent(ov.duplicate_pct)),
    kpi("Quality score", `${q.score.toFixed(0)} / 100`, `Grade ${q.grade}`, true),
  ].join("");
  $("#overview-summary").textContent = state.summary;

  $("#quality-breakdown").innerHTML = `
    <div class="quality-score"><span class="big">${q.score.toFixed(1)}</span><span class="grade">${esc(q.grade)}</span></div>
    ${Object.entries(q.components).map(([k, v]) => `
      <div class="quality-row">
        <span>${esc(k[0].toUpperCase() + k.slice(1))} <span class="muted small">${Math.round(q.weights[k] * 100)}%</span></span>
        ${bar(v)}
        <span class="num">${v.toFixed(1)}</span>
      </div>`).join("")}
    <p class="muted small" style="margin-top:10px">Completeness: non-missing cells. Uniqueness: non-duplicate rows.
      Validity: numeric values inside Tukey fences. Consistency: columns that are not empty, constant or mostly missing.</p>`;

  $("#overview-insights").innerHTML = findingsHtml(state.insights);
  $("#alert-count").textContent = pluralize(alerts.length, "alert");
  $("#overview-alerts").innerHTML = alerts.length
    ? alerts.map((a) => `<div class="alert"><span class="tag ${a.severity}">${a.severity}</span><span>${esc(a.message)}</span></div>`).join("")
    : `<p class="muted">No data-quality issues detected.</p>`;

  const top = missing.slice(0, 12);
  if (top.length) {
    charts.horizontalBars("chart-missing", top.map((m) => m.column), top.map((m) => m.pct), {
      title: "% missing", highlight: top.length, format: (v) => formatPercent(v),
    });
  } else {
    $("#chart-missing").parentElement.innerHTML = `<p class="muted">Every column is complete.</p>`;
  }
  loadPreview();
}

async function loadPreview() {
  const p = state.preview;
  const target = $("#preview-table");
  try {
    const data = await withRecovery(() => api.rows(state.id, p));
    const types = Object.fromEntries(state.profile.columns.map((c) => [c.name, c.semantic_type]));
    const head = data.columns.map((c) => {
      const arrow = p.sort === c ? (p.order === "asc" ? " ↑" : " ↓") : "";
      const numeric = types[c] === "numeric" || types[c] === "identifier";
      return `<th class="sortable${numeric ? " num" : ""}" data-sort="${esc(c)}" title="Sort by ${esc(c)}">${esc(c)}${arrow}</th>`;
    }).join("");
    const body = data.rows.map((row) => `<tr>${row.map((v) => {
      if (v === null) return `<td class="null">null</td>`;
      return `<td class="${typeof v === "number" ? "num" : ""}">${esc(formatCell(v))}</td>`;
    }).join("")}</tr>`).join("");
    target.innerHTML = `<table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    const end = Math.min(p.offset + p.limit, data.total);
    $("#preview-pager").innerHTML = `
      <span>${formatNumber(p.offset + 1)}–${formatNumber(end)} of ${formatNumber(data.total)}</span>
      <button class="btn btn-ghost btn-sm" data-page="-1" ${p.offset === 0 ? "disabled" : ""}>Previous</button>
      <button class="btn btn-ghost btn-sm" data-page="1" ${end >= data.total ? "disabled" : ""}>Next</button>`;
  } catch (err) {
    target.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  }
}

// ------------------------------------------------------------------ columns

function renderColumns() {
  const list = $("#column-list");
  const query = $("#column-search").value.trim().toLowerCase();
  const cols = state.profile.columns.filter((c) => c.name.toLowerCase().includes(query));
  list.innerHTML = cols.map((c) => `
    <li><button data-column="${esc(c.name)}" class="${c.name === state.column ? "active" : ""}">
      <span class="col-name">${esc(c.name)}</span>${typeTag(c.semantic_type)}
    </button></li>`).join("") || `<li class="muted small" style="padding:8px">No matching columns.</li>`;
  if (!state.column) selectColumn(state.profile.columns[0].name);
}

function selectColumn(name) {
  state.column = name;
  $$("#column-list button").forEach((b) => b.classList.toggle("active", b.dataset.column === name));
  const c = state.profile.columns.find((col) => col.name === name);
  const stats = [
    ["Non-null", formatNumber(c.count)],
    ["Missing", `${formatNumber(c.missing)} (${formatPercent(c.missing_pct)})`],
    ["Distinct", `${formatNumber(c.unique)} (${formatPercent(c.unique_pct)})`],
    ["Storage type", c.dtype],
  ];
  const n = c.numeric;
  if (n) {
    stats.push(
      ["Mean", formatNumber(n.mean, { digits: 5 })], ["Std. deviation", formatNumber(n.std, { digits: 5 })],
      ["Minimum", formatNumber(n.min, { digits: 6 })], ["5th percentile", formatNumber(n.p5, { digits: 5 })],
      ["Q1 (25%)", formatNumber(n.q1, { digits: 5 })], ["Median", formatNumber(n.median, { digits: 5 })],
      ["Q3 (75%)", formatNumber(n.q3, { digits: 5 })], ["95th percentile", formatNumber(n.p95, { digits: 5 })],
      ["Maximum", formatNumber(n.max, { digits: 6 })], ["IQR", formatNumber(n.iqr, { digits: 5 })],
      ["Skewness", formatNumber(n.skewness)], ["Excess kurtosis", formatNumber(n.kurtosis)],
      ["Coeff. of variation", formatNumber(n.cv)], ["Zeros", `${formatNumber(n.zeros)} (${formatPercent(n.zeros_pct)})`],
    );
  }
  if (c.outliers) {
    stats.push(["IQR outliers", `${formatNumber(c.outliers.count)} (${formatPercent(c.outliers.pct)})`],
      ["Tukey fences", `${formatNumber(c.outliers.lower, { digits: 4 })} – ${formatNumber(c.outliers.upper, { digits: 4 })}`]);
  }
  if (c.text) stats.push(["Avg. length", formatNumber(c.text.avg_length)], ["Length range", `${c.text.min_length} – ${c.text.max_length}`]);
  if (c.datetime) {
    stats.push(["Earliest", c.datetime.min.slice(0, 10)], ["Latest", c.datetime.max.slice(0, 10)],
      ["Span", pluralize(c.datetime.range_days, "day")]);
  }

  let chartTitle = "";
  if (c.histogram) chartTitle = "Distribution";
  else if (c.timeline) chartTitle = "Rows per month";
  else if (c.top_values) chartTitle = "Most frequent values";

  $("#column-detail").innerHTML = `
    <div class="card-header" style="margin-bottom:0">
      <div><h2>${esc(c.name)}</h2></div>${typeTag(c.semantic_type)}
    </div>
    <div class="stat-grid">${stats.map(([k, v]) => `<div class="stat"><div class="stat-label">${esc(k)}</div><div class="stat-value" title="${esc(v)}">${esc(v)}</div></div>`).join("")}</div>
    ${chartTitle ? `<h3 class="card-title" style="margin-bottom:10px">${chartTitle}</h3><div class="chart-box tall"><canvas id="chart-column"></canvas></div>` : ""}
    ${c.semantic_type === "identifier" ? `<p class="muted">Identifiers are unique per row, so they are excluded from statistics and modelling.</p>` : ""}`;

  if (c.histogram) charts.histogram("chart-column", c.histogram.edges, c.histogram.counts);
  else if (c.timeline) charts.timeline("chart-column", c.timeline);
  else if (c.top_values) {
    const values = c.top_values;
    charts.horizontalBars("chart-column", values.map((v) => v.value), values.map((v) => v.count), {
      title: "Rows", highlight: 1, format: (x) => `${formatNumber(x)} rows`,
    });
  }
}

// ------------------------------------------------------------------ correlations

function renderCorrelations() {
  const corr = state.profile.correlations;
  if (corr.columns.length < 2) {
    $("#heatmap").innerHTML = `<div class="empty">At least two numeric columns are needed for a correlation matrix.</div>`;
    $("#corr-pairs").innerHTML = "";
    return;
  }
  const matrix = corr[state.corrMethod];
  const head = corr.columns.map((c) => `<th title="${esc(c)}">${esc(c)}</th>`).join("");
  const body = corr.columns.map((row, i) => `<tr><th title="${esc(row)}">${esc(row)}</th>${matrix[i].map((v, j) => {
    const bg = correlationColor(v);
    const fg = textOn(Math.abs(v ?? 0));
    return `<td style="background:${bg};color:${fg}" title="${esc(row)} × ${esc(corr.columns[j])}: ${v === null ? "n/a" : v.toFixed(3)}">${v === null ? "" : v.toFixed(2)}</td>`;
  }).join("")}</tr>`).join("");
  $("#heatmap").innerHTML = `<table class="heatmap"><thead><tr><th></th>${head}</tr></thead><tbody>${body}</tbody></table>`;

  const rows = corr.top_pairs.map((p) => {
    const r = state.corrMethod === "pearson" ? p.pearson : p.spearman;
    const strength = Math.abs(r) >= 0.7 ? "strong" : Math.abs(r) >= 0.4 ? "moderate" : "weak";
    return [p.a, p.b, formatNumber(p.pearson, { digits: 3 }), formatNumber(p.spearman, { digits: 3 }),
      raw(`<div class="leader-bar">${bar(Math.abs(r) * 100, { neg: r < 0 })}<span class="tag">${strength}${r < 0 ? " −" : " +"}</span></div>`)];
  });
  $("#corr-pairs").innerHTML = tableHtml(["Variable A", "Variable B", "Pearson r", "Spearman ρ", "Strength"], rows, { numeric: [2, 3] });
}

// ------------------------------------------------------------------ target selects

function setupTargetSelects() {
  const candidates = columnsOfType("numeric", "categorical", "boolean");
  const suggested = state.dataset.suggested_target?.column;
  const options = candidates.map((c) => `<option value="${esc(c.name)}" ${c.name === suggested ? "selected" : ""}>${esc(c.name)} (${esc(c.semantic_type)})</option>`).join("");
  $("#stats-target").innerHTML = options;
  $("#model-target").innerHTML = options;
  $("#model-task").value = "";
  $("#model-tune").checked = false;
  $("#model-choices").innerHTML = (state.config?.model_choices ?? []).map((m) => `
    <label class="check"><input type="checkbox" value="${esc(m.key)}" checked> ${esc(m.name)}</label>`).join("");
  $("#stats-run").disabled = !candidates.length;
  $("#model-run").disabled = !candidates.length;
}

// ------------------------------------------------------------------ statistics

async function runStatistics() {
  const button = $("#stats-run");
  const target = $("#stats-target").value;
  setBusy(button, true, "Running…");
  $("#stats-output").innerHTML = `<div class="card loading-panel"><div class="spinner lg"></div><span class="muted">Running hypothesis tests…</span></div>`;
  try {
    const alpha = $("#stats-alpha").value;
    state.stats = await withRecovery(() => api.statistics(state.id, { target, alpha }));
    renderStatistics();
  } catch (err) {
    $("#stats-output").innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  } finally {
    setBusy(button, false);
  }
}

function renderStatistics() {
  const s = state.stats;
  const effectMax = { r: 1, cramers_v: 1, eta_squared: 0.3 };
  const rows = s.tests.map((t) => [
    t.feature,
    t.test,
    formatNumber(t.statistic, { digits: 4 }),
    formatPValue(t.p_value),
    formatPValue(t.p_adjusted),
    raw(`<div class="leader-bar">${bar(100 * Math.abs(t.effect_size ?? 0) / effectMax[t.effect_measure], { neg: (t.effect_size ?? 0) < 0 })}
      <span class="num" style="min-width:48px">${formatNumber(t.effect_size, { digits: 3 })}</span></div>`),
    raw(`<span class="tag">${esc(t.effect)}</span>`),
    raw(t.significant ? `<span class="tag solid">yes</span>` : `<span class="tag">no</span>`),
  ]);
  const normalRows = s.normality.map((n) => [n.column, n.test, formatNumber(n.statistic, { digits: 4 }), formatPValue(n.p_value),
    formatNumber(n.skewness, { digits: 3 }), raw(n.normal ? `<span class="tag">normal</span>` : `<span class="tag warning">non-normal</span>`)]);

  $("#stats-output").innerHTML = `
    <div class="kpis">
      ${kpi("Features tested", formatNumber(s.tests.length))}
      ${kpi("Significant", formatNumber(s.significant_count), `at α = ${s.alpha}`, true)}
      ${kpi("Target", s.target, s.task)}
      ${kpi("Correction", "BH-FDR", "controls false discoveries")}
    </div>
    <div class="grid-2" style="margin-top:18px">
      <div class="card"><div class="card-header"><h2 class="card-title">Findings</h2></div><div class="findings">${findingsHtml(s.insights)}</div></div>
      ${s.class_balance ? `<div class="card"><div class="card-header"><h2 class="card-title">Class balance</h2></div><div class="chart-box"><canvas id="chart-balance"></canvas></div></div>`
        : `<div class="card"><div class="card-header"><h2 class="card-title">How to read this</h2></div>
           <p class="muted">Numeric features use Pearson correlation (with Spearman ρ as a rank-based check); categorical features use one-way ANOVA with η² as the effect size. Significance shows that an association is unlikely to be chance; the effect size shows whether it matters.</p></div>`}
    </div>
    <div class="card">
      <div class="card-header"><h2 class="card-title">Association with ${esc(s.target)}</h2><span class="muted small">Sorted by significance, then effect size</span></div>
      <div class="table-wrap">${rows.length ? tableHtml(["Feature", "Test", "Statistic", "p-value", "Adjusted p", "Effect size", "Magnitude", "Significant"], rows, { numeric: [2, 3, 4] }) : `<div class="empty">No testable features.</div>`}</div>
    </div>
    <div class="card">
      <div class="card-header"><h2 class="card-title">Normality of numeric columns</h2><span class="muted small">H₀: the column is normally distributed</span></div>
      <div class="table-wrap">${normalRows.length ? tableHtml(["Column", "Test", "Statistic", "p-value", "Skewness", "Verdict"], normalRows, { numeric: [2, 3, 4] }) : `<div class="empty">No numeric columns to test.</div>`}</div>
    </div>`;
  if (s.class_balance) {
    charts.horizontalBars("chart-balance", s.class_balance.map((c) => c.label), s.class_balance.map((c) => c.pct), {
      title: "% of rows", highlight: s.class_balance.length, format: (v) => formatPercent(v),
    });
  }
}

// ------------------------------------------------------------------ model

const TRAIN_STEPS = [
  "Preparing features and encoders",
  "Cross-validating candidate models",
  "Tuning the best model",
  "Evaluating on the hold-out set",
  "Computing permutation importance",
];

// A model trained earlier in this server session survives a page reload.
async function restoreModel() {
  if (!state.dataset.has_model || state.model) return;
  try {
    state.model = await withRecovery(() => api.modelSummary(state.id));
    state.lastTrainOptions = optionsFromSummary(state.model);
    $("#model-target").value = state.model.target;
    renderModel();
  } catch {
    /* the empty state is already shown */
  }
}

async function runModel() {
  const button = $("#model-run");
  const models = $$("#model-choices input:checked").map((i) => i.value);
  if (!models.length) {
    toast("Select at least one algorithm.", { error: true });
    return;
  }
  const options = {
    target: $("#model-target").value,
    task: $("#model-task").value || null,
    models,
    test_size: Number($("#model-test-size").value),
    cv_folds: Number($("#model-folds").value),
    tune: $("#model-tune").checked,
  };
  const steps = TRAIN_STEPS.filter((s, i) => i !== 2 || options.tune);
  $("#model-output").innerHTML = `<div class="card loading-panel"><div class="spinner lg"></div>
    <ul class="steps">${steps.map((s, i) => `<li class="${i === 0 ? "active" : ""}">${s}</li>`).join("")}</ul></div>`;
  let step = 0;
  const timer = setInterval(() => {
    const items = $$(".steps li");
    if (step >= items.length - 1) return;
    items[step].className = "done";
    step += 1;
    items[step].className = "active";
  }, options.tune ? 2600 : 1600);

  setBusy(button, true, "Training…");
  try {
    state.lastTrainOptions = options;
    state.model = await withRecovery(() => api.train(state.id, options));
    renderModel();
  } catch (err) {
    $("#model-output").innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  } finally {
    clearInterval(timer);
    setBusy(button, false);
  }
}

function renderModel() {
  const m = state.model;
  const ev = m.evaluation.metrics;
  const isClf = m.task === "classification";
  const primary = m.primary_metric;
  const best = m.leaderboard.find((r) => r.key === m.best_model.key && !r.baseline);
  const metricKeys = Object.keys(m.leaderboard[0].cv);

  const headline = isClf
    ? [kpi("Hold-out accuracy", formatMetric("accuracy", ev.accuracy), `baseline ${formatMetric("accuracy", ev.baseline_accuracy)}`, true),
      kpi("Balanced accuracy", formatMetric("balanced_accuracy", ev.balanced_accuracy)),
      kpi("F1 (weighted)", formatMetric("f1_weighted", ev.f1_weighted)),
      kpi("ROC AUC", formatMetric("roc_auc", ev.roc_auc))]
    : [kpi("Hold-out R²", formatMetric("r2", ev.r2), `baseline ${formatMetric("r2", ev.baseline_r2)}`, true),
      kpi("MAE", formatMetric("mae", ev.mae), `baseline ${formatMetric("mae", ev.baseline_mae)}`),
      kpi("RMSE", formatMetric("rmse", ev.rmse)),
      kpi("MAPE", formatMetric("mape", ev.mape))];

  // Leaderboard bars are scaled between the baseline and the best score.
  const scores = m.leaderboard.map((r) => r.cv[primary].mean).filter((v) => v !== null);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  const leaderRows = m.leaderboard.map((r) => {
    const v = r.cv[primary].mean;
    const width = v === null || hi === lo ? 0 : 100 * (v - lo) / (hi - lo);
    return [
      r.rank ?? "–",
      raw(`${esc(r.name)}${r.key === m.best_model.key && !r.baseline ? ' <span class="tag solid">best</span>' : ""}`),
      ...metricKeys.map((k) => `${formatMetric(k, r.cv[k].mean)} ± ${formatSpread(k, r.cv[k].std)}`),
      raw(`<div class="leader-bar">${bar(r.baseline ? 0 : Math.max(width, 2))}</div>`),
      formatMetric(primary, r.train_score),
      `${r.fit_seconds.toFixed(2)}s`,
    ];
  });

  const importance = m.feature_importance.slice(0, 15);
  const diagnostics = isClf ? classificationDiagnostics(m) : regressionDiagnostics();

  $("#model-output").innerHTML = `
    <div class="kpis">
      ${kpi("Best model", m.best_model.name, `${metricLabel(primary)} ${formatMetric(primary, best.cv[primary].mean)} (CV)`)}
      ${headline.join("")}
    </div>
    <div class="grid-2" style="margin-top:18px">
      <div class="card"><div class="card-header"><h2 class="card-title">Findings</h2><span class="muted small">${m.duration_seconds}s · ${formatNumber(m.train_rows)} train / ${formatNumber(m.test_rows)} hold-out rows</span></div>
        <div class="findings">${findingsHtml(m.insights)}</div></div>
      <div class="card"><div class="card-header"><h2 class="card-title">Feature importance</h2><span class="muted small">Permutation, on hold-out data</span></div>
        <div class="chart-box tall"><canvas id="chart-importance"></canvas></div></div>
    </div>
    <div class="card">
      <div class="card-header"><h2 class="card-title">Model leaderboard</h2><span class="muted small">${m.cv_folds}-fold cross-validation on the training set · ranked by ${esc(metricLabel(primary))}</span></div>
      <div class="table-wrap">${tableHtml(["Rank", "Model", ...metricKeys.map(metricLabel), "Relative", `Train ${metricLabel(primary)}`, "Time"], leaderRows,
        { numeric: [0, ...metricKeys.map((_, i) => i + 2), metricKeys.length + 3, metricKeys.length + 4], rowClass: (i) => (m.leaderboard[i].key === m.best_model.key && !m.leaderboard[i].baseline ? "highlight" : "") })}</div>
    </div>
    ${diagnostics}
    ${m.tuning ? tuningCard(m.tuning) : ""}
    <div class="grid-2">
      <div class="card"><div class="card-header"><h2 class="card-title">What-if prediction</h2><span class="muted small">Score a single record with the trained pipeline</span></div>
        <form id="predict-form" class="predict-form">${predictFields(m.input_schema)}</form>
        <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn btn-primary" id="predict-run" type="button">Predict</button>
          <a class="btn btn-ghost" data-ensure href="${api.modelUrl(state.id)}">Download model (.joblib)</a>
        </div>
        <div id="predict-result"></div>
      </div>
      <div class="card"><div class="card-header"><h2 class="card-title">Features used</h2></div>${featuresCard(m)}</div>
    </div>`;

  charts.horizontalBars("chart-importance", importance.map((f) => f.feature), importance.map((f) => f.importance), {
    title: `Drop in ${metricLabel(primary)} when shuffled`, highlight: 3, format: (v) => formatNumber(v, { digits: 4 }),
  });
  if (isClf && m.evaluation.roc_curve) charts.rocCurve("chart-roc", m.evaluation.roc_curve, ev.roc_auc);
  if (!isClf) {
    charts.actualVsPredicted("chart-scatter", m.evaluation.scatter);
    charts.histogram("chart-residuals", m.evaluation.residual_histogram.edges, m.evaluation.residual_histogram.counts, { title: "Rows" });
  }
}

function classificationDiagnostics(m) {
  const { labels, matrix } = m.evaluation.confusion_matrix;
  const max = Math.max(...matrix.flat(), 1);
  const cm = `<table class="confusion"><thead><tr><th></th>${labels.map((l) => `<th>${esc(l)}</th>`).join("")}</tr></thead>
    <tbody>${matrix.map((row, i) => `<tr><th style="text-align:right">${esc(labels[i])}</th>${row.map((v, j) => {
      const t = v / max;
      const bg = i === j ? mixColor("#ffffff", "#d97757", t) : mixColor("#ffffff", "#6f7f8c", t);
      return `<td style="background:${bg};color:${textOn(t)}">${formatNumber(v)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
  const perClass = m.evaluation.per_class.map((c) => [c.label, formatPercent(c.precision, { fraction: true }),
    formatPercent(c.recall, { fraction: true }), formatPercent(c.f1, { fraction: true }), formatNumber(c.support)]);
  const second = m.evaluation.roc_curve
    ? `<div class="card"><div class="card-header"><h2 class="card-title">ROC curve</h2><span class="muted small">Positive class: ${esc(m.evaluation.roc_curve.positive_label)}</span></div><div class="chart-box tall"><canvas id="chart-roc"></canvas></div></div>`
    : `<div class="card"><div class="card-header"><h2 class="card-title">Per-class performance</h2></div><div class="table-wrap">${tableHtml(["Class", "Precision", "Recall", "F1", "Support"], perClass, { numeric: [1, 2, 3, 4] })}</div></div>`;
  return `<div class="grid-2">
      <div class="card"><div class="card-header"><h2 class="card-title">Confusion matrix</h2><span class="muted small">Rows: actual · Columns: predicted</span></div>
        <div style="overflow:auto">${cm}</div>
        ${m.evaluation.roc_curve ? `<div class="table-wrap" style="margin-top:16px">${tableHtml(["Class", "Precision", "Recall", "F1", "Support"], perClass, { numeric: [1, 2, 3, 4] })}</div>` : ""}
      </div>
      ${second}
    </div>`;
}

function regressionDiagnostics() {
  return `<div class="grid-2">
      <div class="card"><div class="card-header"><h2 class="card-title">Actual vs predicted</h2><span class="muted small">Hold-out sample</span></div><div class="chart-box tall"><canvas id="chart-scatter"></canvas></div></div>
      <div class="card"><div class="card-header"><h2 class="card-title">Residual distribution</h2><span class="muted small">Actual − predicted</span></div><div class="chart-box tall"><canvas id="chart-residuals"></canvas></div></div>
    </div>`;
}

function tuningCard(t) {
  const params = Object.entries(t.best_params).map(([k, v]) => [k, String(v)]);
  return `<div class="card"><div class="card-header"><h2 class="card-title">Hyper-parameter search</h2>
      <span class="muted small">${t.iterations} randomised configurations · ${t.improved ? "tuned model kept" : "defaults kept"}</span></div>
      <div class="grid-2">
        <div class="kpis">${kpi("CV score before", formatNumber(t.score_before, { digits: 4 }))}${kpi("CV score after", formatNumber(t.score_after, { digits: 4 }), "", t.improved)}</div>
        <div class="table-wrap">${tableHtml(["Parameter", "Best value"], params)}</div>
      </div></div>`;
}

function featuresCard(m) {
  const f = m.features;
  const group = (label, items) => (items.length ? `<p style="margin-bottom:8px"><span class="muted small">${label}</span><br>${items.map((i) => `<span class="tag" style="margin:3px 4px 0 0">${esc(i)}</span>`).join("")}</p>` : "");
  const excluded = Object.entries(f.excluded).map(([c, why]) => `${c} (${why})`);
  return `${group("Numeric · median imputation + standard scaling", f.numeric)}
    ${group("Categorical · one-hot encoding, max 20 levels", f.categorical)}
    ${group("Date · year, month, weekday, day of year", f.datetime)}
    ${group("Excluded", excluded)}
    ${m.notes.length ? `<ul class="notes">${m.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}`;
}

function predictFields(schema) {
  return schema.map((f) => {
    const label = `<span class="field-label">${esc(f.name)}</span>`;
    if (f.kind === "categorical") {
      return `<label class="field">${label}<select class="input" name="${esc(f.name)}">${f.options.map((o) => `<option ${o === f.default ? "selected" : ""}>${esc(o)}</option>`).join("")}</select></label>`;
    }
    if (f.kind === "datetime") {
      return `<label class="field">${label}<input class="input" type="date" name="${esc(f.name)}" value="${esc(f.default ?? "")}"></label>`;
    }
    const step = f.integer ? 1 : "any";
    const value = f.integer ? Math.round(f.default ?? 0) : f.default ?? "";
    return `<label class="field">${label}<input class="input" type="number" step="${step}" name="${esc(f.name)}" value="${esc(value)}" placeholder="${esc(formatNumber(f.min))} – ${esc(formatNumber(f.max))}"></label>`;
  }).join("");
}

async function runPrediction() {
  const button = $("#predict-run");
  const record = Object.fromEntries(new FormData($("#predict-form")));
  setBusy(button, true, "Scoring…");
  try {
    const { predictions } = await withRecovery(() => api.predict(state.id, record));
    const p = predictions[0];
    const value = typeof p.prediction === "number" ? formatNumber(p.prediction, { digits: 5 }) : p.prediction;
    let probs = "";
    if (p.probabilities) {
      probs = Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).map(([label, prob]) => `
        <div class="prob-row"><span>${esc(label)}</span>${bar(prob * 100)}<span class="num">${formatPercent(prob, { fraction: true })}</span></div>`).join("");
    }
    $("#predict-result").innerHTML = `<div class="predict-result">
      <div class="muted small">Predicted ${esc(state.model.target)}</div>
      <div class="value">${esc(value)}</div>${probs ? `<div style="margin-top:10px">${probs}</div>` : ""}</div>`;
  } catch (err) {
    $("#predict-result").innerHTML = `<div class="error-box" style="margin-top:14px">${esc(err.message)}</div>`;
  } finally {
    setBusy(button, false);
  }
}

// ------------------------------------------------------------------ SQL

async function initSql() {
  try {
    const schema = await withRecovery(() => api.sqlSchema(state.id));
    $("#sql-table").textContent = schema.table;
    $("#sql-columns").innerHTML = schema.columns.map((c) => `<li data-insert="${esc(c.name)}" title="Insert into query"><code>${esc(c.name)}</code><span>${esc(c.type)}</span></li>`).join("");
    $("#sql-examples").innerHTML = schema.examples.map((e, i) => `<button data-example="${i}">${esc(e.title)}</button>`).join("");
    state.sqlExamples = schema.examples;
    if (!$("#sql-editor").value.trim() || state.sqlEditorDataset !== state.id) {
      $("#sql-editor").value = schema.examples[1]?.sql ?? schema.examples[0].sql;
      state.sqlEditorDataset = state.id;
    }
  } catch (err) {
    toast(err.message, { error: true });
  }
}

async function runSql() {
  const button = $("#sql-run");
  setBusy(button, true, "Running…");
  try {
    const query = $("#sql-editor").value;
    const result = await withRecovery(() => api.sql(state.id, query));
    state.sqlResult = result;
    $("#sql-meta").textContent = `${pluralize(result.row_count, "row")}${result.truncated ? " (truncated)" : ""} · ${result.elapsed_ms} ms`;
    const numeric = result.columns.map((_, i) => i).filter((i) => result.rows.some((r) => typeof r[i] === "number"));
    const rows = result.rows.map((r) => r.map((v) => (v === null ? null : formatCell(v))));
    $("#sql-result").innerHTML = result.rows.length ? tableHtml(result.columns, rows, { numeric }) : `<div class="empty">The query returned no rows.</div>`;
    $("#sql-download").disabled = !result.rows.length;
  } catch (err) {
    state.sqlResult = null;
    $("#sql-meta").textContent = "";
    $("#sql-download").disabled = true;
    $("#sql-result").innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  } finally {
    setBusy(button, false);
  }
}

function insertAtCursor(textarea, text) {
  const { selectionStart: s, selectionEnd: e, value } = textarea;
  textarea.value = value.slice(0, s) + text + value.slice(e);
  textarea.selectionStart = textarea.selectionEnd = s + text.length;
  textarea.focus();
}

// ------------------------------------------------------------------ report

async function renderReport() {
  try {
    await ensureServerState();
  } catch (err) {
    toast(err.message, { error: true });
  }
  $("#report-frame").src = `${api.reportUrl(state.id)}?t=${Date.now()}`;
}

// ------------------------------------------------------------------ events

function bindEvents() {
  window.addEventListener("hashchange", route);
  // Downloads are plain links; make sure this instance has the dataset (and model) before following them.
  document.addEventListener("click", async (event) => {
    const link = event.target.closest("a[data-ensure]");
    if (!link) return;
    event.preventDefault();
    try {
      await ensureServerState();
      window.location.href = link.href;
    } catch (err) {
      toast(err.message, { error: true });
    }
  });
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-action], [data-tab], [data-tab-link], [data-sample], [data-column], [data-sort], [data-page], [data-method], [data-example], [data-insert]");
    if (!target) return;
    if (target.dataset.action === "home") {
      event.preventDefault();
      showHome();
    } else if (target.dataset.tab) setTab(target.dataset.tab);
    else if (target.dataset.tabLink) setTab(target.dataset.tabLink);
    else if (target.dataset.sample) ingest(api.loadSample(target.dataset.sample), "Loading sample dataset…");
    else if (target.dataset.column) selectColumn(target.dataset.column);
    else if (target.dataset.sort) {
      const p = state.preview;
      p.order = p.sort === target.dataset.sort && p.order === "asc" ? "desc" : "asc";
      p.sort = target.dataset.sort;
      p.offset = 0;
      loadPreview();
    } else if (target.dataset.page) {
      state.preview.offset = Math.max(0, state.preview.offset + Number(target.dataset.page) * state.preview.limit);
      loadPreview();
    } else if (target.dataset.method) {
      state.corrMethod = target.dataset.method;
      $$("#corr-method button").forEach((b) => b.classList.toggle("active", b === target));
      renderCorrelations();
    } else if (target.dataset.example) {
      $("#sql-editor").value = state.sqlExamples[Number(target.dataset.example)].sql;
      runSql();
    } else if (target.dataset.insert) {
      insertAtCursor($("#sql-editor"), `"${target.dataset.insert}"`);
    }
  });

  const dropzone = $("#dropzone");
  const input = $("#file-input");
  input.addEventListener("change", () => { handleFile(input.files[0]); input.value = ""; });
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

  $("#column-search").addEventListener("input", renderColumns);
  $("#stats-run").addEventListener("click", runStatistics);
  $("#model-run").addEventListener("click", runModel);
  $("#model-output").addEventListener("click", (e) => { if (e.target.id === "predict-run") runPrediction(); });
  $("#sql-run").addEventListener("click", runSql);
  $("#sql-editor").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); runSql(); }
    if (e.key === "Tab") { e.preventDefault(); insertAtCursor(e.target, "  "); }
  });
  $("#sql-download").addEventListener("click", () => {
    if (state.sqlResult) download("query_result.csv", toCsv(state.sqlResult.columns, state.sqlResult.rows));
  });
  $("#report-refresh").addEventListener("click", renderReport);
  $("#report-print").addEventListener("click", () => $("#report-frame").contentWindow?.print());
  $$(".tabs").forEach((tabs) => tabs.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const all = $$(".tab");
    const i = all.findIndex((t) => t.dataset.tab === state.tab);
    const next = all[(i + (e.key === "ArrowRight" ? 1 : all.length - 1)) % all.length];
    setTab(next.dataset.tab);
    next.focus();
  }));
}

async function init() {
  bindEvents();
  try {
    state.config = await api.config();
    $("#app-version").textContent = `v${state.config.version}`;
    $("#dropzone-sub").textContent = `Up to ${state.config.max_upload_mb} MB · analysed in memory, never written to disk.`;
  } catch (err) {
    toast(err.message, { error: true });
  }
  loadSamples();
  route();
}

init();
