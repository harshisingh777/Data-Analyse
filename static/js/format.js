// Pure formatting helpers. No DOM access, so they are unit-tested with `node --test`.

export const METRIC_LABELS = {
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  f1_weighted: "F1 (weighted)",
  f1_macro: "F1 (macro)",
  roc_auc: "ROC AUC",
  baseline_accuracy: "Baseline accuracy",
  train_accuracy: "Train accuracy",
  r2: "R²",
  mae: "MAE",
  rmse: "RMSE",
  median_ae: "Median AE",
  mape: "MAPE",
  baseline_mae: "Baseline MAE",
  baseline_r2: "Baseline R²",
  train_r2: "Train R²",
};

// Metrics that are proportions in [0, 1] and read best as percentages.
export const PROPORTION_METRICS = new Set([
  "accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "baseline_accuracy", "train_accuracy", "mape",
]);

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function isMissing(value) {
  return value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value));
}

/** Human-friendly number: thousands separators, sensible precision, compact for huge values. */
export function formatNumber(value, { digits = 3, compact = false } = {}) {
  if (isMissing(value)) return "–";
  if (typeof value !== "number") return String(value);
  const abs = Math.abs(value);
  if (compact && abs >= 1e6) {
    return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  }
  if (Number.isInteger(value)) return value.toLocaleString("en-US");
  if (abs >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (abs >= 1) return value.toLocaleString("en-US", { maximumFractionDigits: Math.max(digits - 1, 0) });
  if (abs === 0) return "0";
  return Number(value.toPrecision(digits)).toString();
}

/** Format a percentage. `value` is already 0–100 unless `fraction` is true. */
export function formatPercent(value, { digits = 1, fraction = false } = {}) {
  if (isMissing(value)) return "–";
  const pct = fraction ? value * 100 : value;
  return `${pct.toFixed(digits)}%`;
}

export function formatBytes(bytes) {
  if (isMissing(bytes)) return "–";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatPValue(p) {
  if (isMissing(p)) return "–";
  if (p < 0.0001) return "< 0.0001";
  return p.toFixed(4);
}

export function formatMetric(key, value) {
  if (isMissing(value)) return "–";
  if (PROPORTION_METRICS.has(key)) return formatPercent(value, { fraction: true });
  if (key === "r2" || key === "roc_auc" || key === "baseline_r2" || key === "train_r2") {
    return (Math.abs(value) < 0.0005 ? 0 : value).toFixed(3); // avoid "-0.000"
  }
  return formatNumber(value, { digits: 4 });
}

/** Standard deviation in the same unit as `formatMetric` (percentage points for proportions). */
export function formatSpread(key, std) {
  if (isMissing(std)) return "–";
  if (PROPORTION_METRICS.has(key)) return (std * 100).toFixed(1);
  return formatNumber(std, { digits: 2 });
}

export function metricLabel(key) {
  return METRIC_LABELS[key] ?? key.replaceAll("_", " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** Blend two hex colours; t = 0 returns `from`, t = 1 returns `to`. */
export function mixColor(from, to, t) {
  const parse = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const a = parse(from);
  const b = parse(to);
  const clamped = Math.min(Math.max(t, 0), 1);
  const rgb = a.map((v, i) => Math.round(v + (b[i] - v) * clamped));
  return `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/** Diverging colour for a correlation coefficient in [-1, 1]. */
export function correlationColor(r, { positive = "#d97757", negative = "#6f7f8c", neutral = "#ffffff" } = {}) {
  if (isMissing(r)) return "#f5f3ed";
  return r >= 0 ? mixColor(neutral, positive, r) : mixColor(neutral, negative, -r);
}

/** Readable text colour for a background intensity (0–1). */
export function textOn(intensity) {
  return intensity > 0.55 ? "#ffffff" : "#1f1e1c";
}

/** Serialise a result grid to RFC 4180 CSV. */
export function toCsv(columns, rows) {
  const cell = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  return [columns.map(cell).join(","), ...rows.map((r) => r.map(cell).join(","))].join("\r\n");
}

/** Short x-axis labels for histogram bins. */
export function binLabels(edges) {
  const labels = [];
  for (let i = 0; i < edges.length - 1; i += 1) {
    labels.push(`${formatNumber(edges[i], { compact: true })} – ${formatNumber(edges[i + 1], { compact: true })}`);
  }
  return labels;
}

/** Display a table cell: numbers get separators, midnight ISO timestamps lose their time part. */
export function formatCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return formatNumber(value, { digits: 6 });
  const s = String(value);
  const midnight = s.match(/^(\d{4}-\d{2}-\d{2})T00:00:00$/);
  if (midnight) return midnight[1];
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(s) ? s.replace("T", " ") : s;
}

export function pluralize(count, word, plural = `${word}s`) {
  return `${count.toLocaleString("en-US")} ${count === 1 ? word : plural}`;
}
