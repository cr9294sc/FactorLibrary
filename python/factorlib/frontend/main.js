// FactorLibrary 前端 ES 模块总调度入口

import { apiGet, apiPost, toast } from "./api.js";
import { parseHash, setHash } from "./router.js";
import { ms, store } from "./state.js";
import { renderDataPage } from "./pages/data.js";
import { renderLabPage } from "./pages/lab.js";
import { renderComparePage } from "./pages/compare.js";
import { refreshRuns, renderMiningPage } from "./pages/mining.js";
import { renderRegistryPage } from "./pages/registry.js";

function updateChromeFromContext() {
  const m = store.market;
  const st = ms(m);
  const ctx = st.context;
  const appEl = document.getElementById("app");
  if (appEl) appEl.setAttribute("data-market", m);

  document.querySelectorAll("#marketSwitch button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.getAttribute("data-m") === m));
  });
  document.querySelectorAll(".nav-btn").forEach((b) => {
    if (b.getAttribute("data-p") === store.page) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.hidden = p.id !== `p-${store.page}`;
  });

  const routeEl = document.getElementById("routeText");
  if (routeEl) routeEl.textContent = `#/${m}/${store.page}`;
  const mkName = document.getElementById("mkName");
  if (mkName) mkName.textContent = m === "futures" ? "期货" : "A 股";

  if (!ctx) return;
  const pInfo = ctx.panel_info || {};
  const regSum = ctx.registry_summary || {};
  const bySt = regSum.by_status || {};

  const panelLabel = document.getElementById("panelLabel");
  if (panelLabel) panelLabel.textContent = `panel ${pInfo.data_version || "v1"} (${pInfo.source || "synthetic"})`;
  const extCount = document.getElementById("extCount");
  if (extCount) extCount.textContent = `+${(pInfo.mounted_extensions || []).length} 扩展包`;
  const runCount = document.getElementById("runCount");
  if (runCount) runCount.textContent = String(ctx.active_runs_count || 0);
  const actCount = document.getElementById("actCount");
  if (actCount) actCount.textContent = String(bySt.active || 0);
  const candCount = document.getElementById("candCount");
  if (candCount) candCount.textContent = String(bySt.candidate || 0);

  const srcBadge = document.getElementById("panelSourceBadge");
  if (srcBadge) srcBadge.textContent = pInfo.source === "real" ? "真实面板" : "合成演示";
  const nEl = document.getElementById("nCount");
  if (nEl) nEl.textContent = String(pInfo.n_symbols || "-");
  const tEl = document.getElementById("tCount");
  if (tEl) tEl.textContent = String(pInfo.n_days || "-");
}

export async function navigate(market = store.market, page = store.page) {
  store.market = market;
  store.page = page;
  setHash(market, page);
  updateChromeFromContext();

  const st = ms(market);
  if (!st.context) {
    try {
      st.context = await apiGet(market, "/context");
      updateChromeFromContext();
    } catch (err) {
      toast(`加载 [${market}] 上下文失败: ${err.message}`, "bad");
      return;
    }
  }

  if (page === "data") await renderDataPage(navigate);
  else if (page === "lab") await renderLabPage(navigate);
  else if (page === "compare") await renderComparePage(navigate);
  else if (page === "mining") await renderMiningPage(navigate);
  else if (page === "registry") await renderRegistryPage(navigate);

  updateChromeFromContext();
}

async function togglePanelSource() {
  const m = store.market;
  const st = ms(m);
  const curSrc = st.context && st.context.panel_info ? st.context.panel_info.source : "synthetic";
  const nextMode = curSrc === "real" ? "synthetic" : "real";
  toast(`正在切换为${nextMode === "real" ? "真实" : "合成演示"}面板…`, "warn");
  try {
    st.context = await apiPost(m, "/panel/switch", { mode: nextMode });
    st.klineState.data = null;
    st.compare.workflowResult = null;
    st.compare.abResult = null;
    toast(`已切换至 ${st.context.panel_info.source} 面板 (${st.context.panel_info.n_days}d × ${st.context.panel_info.n_symbols})`, "ok");
    await navigate(m, store.page);
  } catch (err) {
    toast(`切换面板失败: ${err.message}`, "bad");
  }
}

function startBackgroundPolling() {
  if (store.pollTimer) clearInterval(store.pollTimer);
  store.pollTimer = setInterval(async () => {
    const m = store.market;
    const st = ms(m);
    const hasRunning = (st.mining.runs || []).some((r) => r.status === "running" || r.status === "queued");
    if (store.page === "mining" || hasRunning) {
      await refreshRuns(m, st, navigate);
      if (store.page === "mining" && st.mining.openLogRunId) {
        const drawer = document.getElementById("logDrawer");
        if (drawer && !drawer.hidden) {
          const d = await apiGet(m, `/runs/${encodeURIComponent(st.mining.openLogRunId)}`).catch(() => null);
          if (d && d.log_tail) drawer.textContent = d.log_tail;
        }
      }
    }
  }, 1500);
}

function boot() {
  const initial = parseHash();
  store.market = initial.market;
  store.page = initial.page;

  document.querySelectorAll("#marketSwitch button").forEach((btn) => {
    btn.onclick = () => navigate(btn.getAttribute("data-m"), store.page);
  });
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.onclick = () => navigate(store.market, btn.getAttribute("data-p"));
  });

  const runsChip = document.getElementById("runsChip");
  if (runsChip) runsChip.onclick = () => navigate(store.market, "mining");
  const regChip = document.getElementById("regChip");
  if (regChip) regChip.onclick = () => navigate(store.market, "registry");
  const panelChip = document.getElementById("panelChip");
  if (panelChip) panelChip.onclick = togglePanelSource;
  const toggleBtn = document.getElementById("btnTogglePanel");
  if (toggleBtn) toggleBtn.onclick = togglePanelSource;

  window.addEventListener("hashchange", () => {
    const parsed = parseHash();
    navigate(parsed.market, parsed.page);
  });

  startBackgroundPolling();
  navigate(store.market, store.page);
}

boot();
