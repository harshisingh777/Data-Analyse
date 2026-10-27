// Chart.js helpers with a single, restrained visual language.
import { binLabels, formatNumber } from "./format.js";

const ACCENT = "#d97757";
const ACCENT_SOFT = "rgba(217, 119, 87, 0.18)";
const MUTED = "#6b6860";
const GRID = "#efece4";
const NEUTRAL = "#c9c3b5";

const registry = new Map();

const shorten = (label, max = 22) => (label.length > max ? `${label.slice(0, max - 1)}…` : label);

function ready() {
  if (!window.Chart) return false;
  if (!ready.configured) {
    const { defaults } = window.Chart;
    defaults.font.family = '"Inter", -apple-system, "Segoe UI", sans-serif';
    defaults.font.size = 11.5;
    defaults.color = MUTED;
    defaults.borderColor = GRID;
    defaults.plugins.legend.display = false;
    defaults.plugins.tooltip.backgroundColor = "#1f1e1c";
    defaults.plugins.tooltip.padding = 10;
    defaults.plugins.tooltip.cornerRadius = 6;
    defaults.plugins.tooltip.displayColors = false;
    defaults.animation.duration = 250;
    ready.configured = true;
  }
  return true;
}

function draw(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !ready()) return null;
  // Chart.js measures and caches axis-label widths on first draw; measuring with the fallback font
  // clips labels once Inter arrives, so wait for web fonts before the first draw.
  if (document.fonts && document.fonts.status !== "loaded") {
    document.fonts.ready.then(() => draw(canvasId, config));
    return null;
  }
  registry.get(canvasId)?.destroy();
  const chart = new window.Chart(canvas, {
    ...config,
    options: { responsive: true, maintainAspectRatio: false, ...config.options },
  });
  registry.set(canvasId, chart);
  return chart;
}

export function destroyAll() {
  registry.forEach((chart) => chart.destroy());
  registry.clear();
}

const valueAxis = (title, extra = {}) => ({
  grid: { color: GRID },
  border: { display: false },
  title: title ? { display: true, text: title, color: MUTED } : undefined,
  ticks: { callback: (v) => formatNumber(v, { compact: true }) },
  ...extra,
});
const categoryAxis = (extra = {}) => ({ grid: { display: false }, border: { color: GRID }, ...extra });

export function horizontalBars(canvasId, labels, values, { title, highlight = 0, format = formatNumber, colors } = {}) {
  return draw(canvasId, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors ?? values.map((_, i) => (i < highlight ? ACCENT : NEUTRAL)),
        borderRadius: 3,
        maxBarThickness: 18,
      }],
    },
    options: {
      indexAxis: "y",
      scales: {
        x: valueAxis(title, { beginAtZero: true }),
        y: categoryAxis({ ticks: { autoSkip: false, callback: (_, i) => shorten(String(labels[i])) } }),
      },
      plugins: { tooltip: { callbacks: { label: (ctx) => format(ctx.parsed.x) } } },
    },
  });
}

export function histogram(canvasId, edges, counts, { title = "Count" } = {}) {
  return draw(canvasId, {
    type: "bar",
    data: {
      labels: binLabels(edges),
      datasets: [{ data: counts, backgroundColor: ACCENT, borderRadius: 2, barPercentage: 1, categoryPercentage: 0.96 }],
    },
    options: {
      scales: {
        x: categoryAxis({ ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 } }),
        y: valueAxis(title, { beginAtZero: true }),
      },
      plugins: { tooltip: { callbacks: { label: (ctx) => `${formatNumber(ctx.parsed.y)} rows` } } },
    },
  });
}

export function verticalBars(canvasId, labels, values, { title = "Count" } = {}) {
  return draw(canvasId, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: ACCENT, borderRadius: 3, maxBarThickness: 36 }] },
    options: {
      scales: { x: categoryAxis({ ticks: { maxRotation: 45, autoSkip: true } }), y: valueAxis(title, { beginAtZero: true }) },
    },
  });
}

export function timeline(canvasId, points) {
  return draw(canvasId, {
    type: "line",
    data: {
      labels: points.map((p) => p.period),
      datasets: [{
        data: points.map((p) => p.count),
        borderColor: ACCENT,
        backgroundColor: ACCENT_SOFT,
        fill: true,
        tension: 0.25,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2,
      }],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: { x: categoryAxis({ ticks: { maxTicksLimit: 8 } }), y: valueAxis("Rows", { beginAtZero: true }) },
    },
  });
}

export function rocCurve(canvasId, roc, auc) {
  const points = roc.fpr.map((x, i) => ({ x, y: roc.tpr[i] }));
  return draw(canvasId, {
    type: "line",
    data: {
      datasets: [
        { label: `Model (AUC ${auc?.toFixed(3) ?? "–"})`, data: points, borderColor: ACCENT, backgroundColor: ACCENT_SOFT,
          fill: true, pointRadius: 0, borderWidth: 2, tension: 0 },
        { label: "Random guess", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: NEUTRAL, borderDash: [5, 5],
          pointRadius: 0, borderWidth: 1.5, fill: false },
      ],
    },
    options: {
      parsing: false,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate" }, grid: { color: GRID } },
        y: { min: 0, max: 1, title: { display: true, text: "True positive rate" }, grid: { color: GRID } },
      },
      plugins: { legend: { display: true, position: "bottom", labels: { boxWidth: 12, boxHeight: 2 } } },
    },
  });
}

export function actualVsPredicted(canvasId, points) {
  const values = points.flatMap((p) => [p.actual, p.predicted]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  return draw(canvasId, {
    type: "scatter",
    data: {
      datasets: [
        { label: "Hold-out rows", data: points.map((p) => ({ x: p.actual, y: p.predicted })),
          backgroundColor: "rgba(217, 119, 87, 0.45)", pointRadius: 3, pointHoverRadius: 5 },
        { label: "Perfect prediction", type: "line", data: [{ x: min, y: min }, { x: max, y: max }],
          borderColor: NEUTRAL, borderDash: [5, 5], pointRadius: 0, borderWidth: 1.5 },
      ],
    },
    options: {
      scales: {
        x: valueAxis("Actual", { type: "linear" }),
        y: valueAxis("Predicted"),
      },
      plugins: {
        legend: { display: true, position: "bottom", labels: { boxWidth: 10 } },
        tooltip: { callbacks: { label: (ctx) => `actual ${formatNumber(ctx.parsed.x)} · predicted ${formatNumber(ctx.parsed.y)}` } },
      },
    },
  });
}
