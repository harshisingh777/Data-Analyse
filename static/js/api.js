// Thin wrapper over the DataLens JSON API.

async function request(path, { method = "GET", body, json } = {}) {
  const init = { method, headers: {} };
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("Cannot reach the DataLens server. Is it running?");
  }
  const isJson = (response.headers.get("Content-Type") || "").includes("application/json");
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) {
    const message = isJson && payload.error ? payload.error : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload;
}

export const api = {
  config: () => request("/api/config"),
  samples: () => request("/api/samples"),
  upload(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/api/datasets", { method: "POST", body: form });
  },
  loadSample: (key) => request(`/api/samples/${encodeURIComponent(key)}`, { method: "POST" }),
  dataset: (id) => request(`/api/datasets/${id}`),
  rows: (id, { offset = 0, limit = 50, sort, order } = {}) => {
    const params = new URLSearchParams({ offset, limit });
    if (sort) params.set("sort", sort);
    if (order) params.set("order", order);
    return request(`/api/datasets/${id}/rows?${params}`);
  },
  statistics: (id, { target, alpha }) =>
    request(`/api/datasets/${id}/statistics?${new URLSearchParams({ target, alpha })}`),
  train: (id, options) => request(`/api/datasets/${id}/train`, { method: "POST", json: options }),
  modelSummary: (id) => request(`/api/datasets/${id}/model`),
  predict: (id, record) => request(`/api/datasets/${id}/predict`, { method: "POST", json: { record } }),
  sqlSchema: (id) => request(`/api/datasets/${id}/sql/schema`),
  sql: (id, query) => request(`/api/datasets/${id}/sql`, { method: "POST", json: { query } }),
  reportUrl: (id, download = false) => `/api/datasets/${id}/report${download ? "?download=1" : ""}`,
  modelUrl: (id) => `/api/datasets/${id}/model/download`,
  exportUrl: (id) => `/api/datasets/${id}/export`,
};
