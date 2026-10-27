// Run with: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  binLabels,
  correlationColor,
  escapeHtml,
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
} from "../../static/js/format.js";

test("escapeHtml neutralises markup", () => {
  assert.equal(escapeHtml(`<img src=x onerror="a('b')">&`), "&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;");
  assert.equal(escapeHtml(null), "");
});

test("formatNumber picks sensible precision", () => {
  assert.equal(formatNumber(1234567), "1,234,567");
  assert.equal(formatNumber(1234.56), "1,235");
  assert.equal(formatNumber(12.3456), "12.35");
  assert.equal(formatNumber(0.000123456), "0.000123");
  assert.equal(formatNumber(0), "0");
  assert.equal(formatNumber(2500000, { compact: true }), "2.5M");
  assert.equal(formatNumber(null), "–");
  assert.equal(formatNumber(Number.NaN), "–");
  assert.equal(formatNumber("text"), "text");
});

test("percentages, bytes and p-values", () => {
  assert.equal(formatPercent(12.345), "12.3%");
  assert.equal(formatPercent(0.5, { fraction: true }), "50.0%");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(5 * 1024 * 1024), "5.0 MB");
  assert.equal(formatPValue(0.00001), "< 0.0001");
  assert.equal(formatPValue(0.04321), "0.0432");
});

test("formatMetric and formatSpread respect metric units", () => {
  assert.equal(formatMetric("accuracy", 0.8613), "86.1%");
  assert.equal(formatMetric("r2", 0.91234), "0.912");
  assert.equal(formatMetric("r2", -0.0001), "0.000");
  assert.equal(formatMetric("mae", 33856.6), "33,857");
  assert.equal(formatMetric("roc_auc", null), "–");
  assert.equal(formatSpread("accuracy", 0.018), "1.8");
  assert.equal(formatSpread("mae", 1777.4), "1,777");
  assert.equal(metricLabel("f1_weighted"), "F1 (weighted)");
  assert.equal(metricLabel("custom_metric"), "Custom metric");
});

test("formatCell trims midnight timestamps", () => {
  assert.equal(formatCell("2024-03-01T00:00:00"), "2024-03-01");
  assert.equal(formatCell("2024-03-01T13:45:00"), "2024-03-01 13:45:00");
  assert.equal(formatCell(1500.25), "1,500");
  assert.equal(formatCell(null), "");
});

test("colour helpers", () => {
  assert.equal(mixColor("#000000", "#ffffff", 0.5), "#808080");
  assert.equal(mixColor("#000000", "#ffffff", 2), "#ffffff");
  assert.equal(correlationColor(1), "#d97757");
  assert.equal(correlationColor(-1), "#6f7f8c");
  assert.equal(correlationColor(0), "#ffffff");
  assert.equal(correlationColor(null), "#f5f3ed");
  assert.equal(textOn(0.9), "#ffffff");
  assert.equal(textOn(0.1), "#1f1e1c");
});

test("toCsv quotes fields that need it", () => {
  const csv = toCsv(["name", "note"], [["a", 'say "hi"'], ["b,c", null], ["d", "line\nbreak"]]);
  assert.equal(csv, 'name,note\r\na,"say ""hi"""\r\n"b,c",\r\nd,"line\nbreak"');
});

test("binLabels and pluralize", () => {
  assert.deepEqual(binLabels([0, 5, 10]), ["0 – 5", "5 – 10"]);
  assert.equal(pluralize(1, "row"), "1 row");
  assert.equal(pluralize(2412, "row"), "2,412 rows");
});
