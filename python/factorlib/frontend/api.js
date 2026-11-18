// 市场隔离 REST API 客户端与全局通知/弹窗辅助

export function toast(msg, kind = "ok", ttlMs = 3200) {
  const box = document.getElementById("toastBox");
  if (!box) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => {
    el.remove();
  }, ttlMs);
}

export function showDialog(title, bodyText) {
  const dlg = document.getElementById("dlg");
  const t = document.getElementById("dlgT");
  const b = document.getElementById("dlgB");
  if (!dlg || !t || !b) return;
  t.textContent = title;
  b.textContent = bodyText;
  if (typeof dlg.showModal === "function") {
    dlg.showModal();
  } else {
    dlg.setAttribute("open", "");
  }
}

async function request(method, url, body = null) {
  const opts = {
    method,
    headers: {},
  };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { error: text || res.statusText };
  }
  if (!res.ok) {
    const errMsg = (data && (data.error || data.detail)) || res.statusText || `HTTP ${res.status}`;
    throw new Error(typeof errMsg === "string" ? errMsg : JSON.stringify(errMsg));
  }
  return data;
}

export function apiGet(market, path, params = {}) {
  const url = new URL(`/api/${market}${path}`, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") {
      url.searchParams.set(k, String(v));
    }
  });
  return request("GET", url.toString());
}

export function apiPost(market, path, body = {}) {
  return request("POST", `/api/${market}${path}`, body);
}

export function apiPatch(market, path, body = {}) {
  return request("PATCH", `/api/${market}${path}`, body);
}

export function apiDelete(market, path) {
  return request("DELETE", `/api/${market}${path}`);
}
