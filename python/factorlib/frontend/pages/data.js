// Page 1: 数据与特征池中心 (面板体检、1d K线与组合、1m扩展包挂载、FeatureSet编辑器)

import { apiGet, apiPost, toast } from "../api.js";
import { drawKlineChart, escapeHtml, fmtPct } from "../components/charts.js";
import { ms, store } from "../state.js";

const BASE_OPS_CANDIDATES = [
  "ts_mean",
  "ts_std",
  "ts_zscore",
  "ts_rank",
  "ts_corr",
  "ts_delta",
  "ts_decay_linear",
  "cs_zscore",
  "cs_rank",
  "abs",
  "sign",
  "slog",
  "sub",
  "div",
];
const WIN_CANDIDATES = [3, 5, 10, 20, 40, 60];

function chipHtml(label, active, attrName) {
  return `<button type="button" class="chip ${active ? "active" : ""}" aria-pressed="${active}" ${attrName}="${escapeHtml(label)}">${escapeHtml(label)}</button>`;
}

function bindChipToggle(container, onChange) {
  if (!container) return;
  container.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      const next = btn.getAttribute("aria-pressed") !== "true";
      btn.setAttribute("aria-pressed", String(next));
      btn.classList.toggle("active", next);
      if (onChange) onChange();
    };
  });
}

function activeChips(container, attrName) {
  if (!container) return [];
  return [...container.querySelectorAll("button[aria-pressed='true']")].map((b) =>
    b.getAttribute(attrName)
  );
}

export async function renderDataPage(navigate) {
  const m = store.market;
  const st = ms(m);
  const ctx = st.context;
  if (!ctx) return;

  const pInfo = ctx.panel_info || {};
  const kpiEl = document.getElementById("p1KpiGrid");
  if (kpiEl) {
    const extCount = (pInfo.mounted_extensions || []).length;
    kpiEl.innerHTML = `
      <div class="card p-3 kpi"><span class="lbl">data_version</span><b>${escapeHtml(pInfo.data_version || "v1")}</b></div>
      <div class="card p-3 kpi"><span class="lbl">日期跨度 (${pInfo.n_days || 0}d)</span><b style="font-size:14px;">${escapeHtml(pInfo.start_date || "-")} ~ ${escapeHtml(pInfo.end_date || "-")}</b></div>
      <div class="card p-3 kpi"><span class="lbl">有效标的 N</span><b>${pInfo.n_symbols || 0}</b></div>
      <div class="card p-3 kpi"><span class="lbl">平均缺失率</span><b>${fmtPct(pInfo.avg_missing_rate || 0, 1)}</b></div>
      <div class="card p-3 kpi"><span class="lbl">字段数 / 已挂扩展</span><b>${pInfo.n_fields || 0} / ${extCount}</b></div>
    `;
  }

  // 市场专属覆盖率卡片
  const mkSpec = document.getElementById("mkSpecific");
  if (mkSpec) {
    const bars = (pInfo.coverage_bars || [])
      .map(
        (b) => `
      <div class="mt-2">
        <div class="flex justify-between lbl"><span>${escapeHtml(b.label)}</span><span>${b.pct}%</span></div>
        <div class="bar"><i style="width:${Math.min(100, Math.max(0, b.pct))}%"></i></div>
      </div>`
      )
      .join("");
    const subNote =
      m === "futures"
        ? `主力映射 rule=${pInfo.dominant_rule || 3} · alt_data_lag_days=${pInfo.alt_data_lag_days ?? 1}`
        : `基准池：${(pInfo.benchmarks || ["中证1000"]).join(" / ")}`;
    mkSpec.innerHTML = `<b>${m === "futures" ? "期货专属基线" : "A 股专属基线"}</b><div class="lbl mt-1">${escapeHtml(subNote)}</div>${bars}`;
  }

  // 加载扩展包列表与 FeatureSets
  const [exts, fsets] = await Promise.all([
    apiGet(m, "/extensions").catch(() => pInfo.available_extensions || []),
    apiGet(m, "/feature-sets").catch(() => []),
  ]);
  st.extensions = exts || [];
  st.featureSets = fsets || [];
  if (!st.activeFeatureSetId && st.featureSets.length) {
    st.activeFeatureSetId = st.featureSets[0].id;
  }

  renderExtensionsBox(m, st);
  renderFeatureSetEditor(m, st, navigate);
  await loadAndRenderKline(m, st);
}

function renderExtensionsBox(market, st) {
  const el = document.getElementById("extList");
  if (!el) return;
  const exts = st.extensions || [];
  el.innerHTML =
    exts
      .map((e) => {
        const cov = e.coverage || {};
        const covStr = cov.start_date
          ? `${cov.start_date}${cov.end_date ? ` ~ ${cov.end_date}` : ""}`
          : "可用";
        const disabled = !e.built || e.kind === "intraday_tensor";
        return `
      <div class="card p-2">
        <div class="flex justify-between items-center">
          <b>${escapeHtml(e.pack_id)} <span class="lbl">${escapeHtml(e.version || "v1")}</span></b>
          <label class="flex gap-1 items-center lbl">
            <input type="checkbox" data-ext-pack="${escapeHtml(e.pack_id)}" ${e.mounted ? "checked" : ""} ${disabled ? "disabled" : ""}>
            挂载
          </label>
        </div>
        <div class="lbl mt-1">${escapeHtml((e.fields || []).join(", "))}</div>
        <div class="lbl">覆盖 ${escapeHtml(covStr)} · lag=${e.lag_days ?? 0}d (${escapeHtml(e.asof_rule || "T_close")})</div>
      </div>`;
      })
      .join("") || '<div class="lbl">暂无扩展包</div>';

  el.querySelectorAll("input[data-ext-pack]").forEach((cb) => {
    cb.onchange = async () => {
      const packId = cb.getAttribute("data-ext-pack");
      const action = cb.checked ? "mount" : "unmount";
      try {
        await apiPost(market, "/extensions/mount", { pack_id: packId, action });
        st.context = await apiGet(market, "/context");
        toast(`${action === "mount" ? "已挂载" : "已卸载"}扩展包 ${packId}`, "ok");
        await renderDataPage();
      } catch (err) {
        cb.checked = !cb.checked;
        toast(`扩展包操作失败: ${err.message}`, "bad");
      }
    };
  });
}

function renderFeatureSetEditor(market, st, navigate) {
  const pInfo = (st.context && st.context.panel_info) || {};
  const presets = (st.context && st.context.postprocess_presets) || { standard: [], minimal: [] };
  const fsets = st.featureSets || [];
  const activeFs = fsets.find((f) => f.id === st.activeFeatureSetId) || fsets[0] || null;

  const fsSel = document.getElementById("fsSelect");
  const diffSel = document.getElementById("fsDiffSelect");
  const diffHint = document.getElementById("fsDiffHint");
  if (fsSel) {
    fsSel.innerHTML = fsets
      .map((f) => `<option value="${f.id}" ${activeFs && f.id === activeFs.id ? "selected" : ""}>${escapeHtml(f.name)} (v${f.version})</option>`)
      .join("");
    fsSel.onchange = () => {
      st.activeFeatureSetId = Number(fsSel.value);
      renderFeatureSetEditor(market, st, navigate);
    };
  }

  if (diffSel && diffHint) {
    diffSel.innerHTML =
      `<option value="">版本/池对比…</option>` +
      fsets
        .filter((f) => !activeFs || f.id !== activeFs.id)
        .map((f) => `<option value="${f.id}">对比: ${escapeHtml(f.name)}</option>`)
        .join("");
    diffSel.onchange = () => {
      const other = fsets.find((f) => f.id === Number(diffSel.value));
      if (!other || !activeFs) {
        diffHint.style.display = "none";
        return;
      }
      const aSet = new Set(activeFs.base_fields || []);
      const bSet = new Set(other.base_fields || []);
      const added = [...aSet].filter((x) => !bSet.has(x));
      const removed = [...bSet].filter((x) => !aSet.has(x));
      diffHint.style.display = "block";
      diffHint.textContent = `对比 ${other.name} → ${activeFs.name}：新增字段 [${added.join(", ") || "无"}] · 移除 [${removed.join(", ") || "无"}]`;
    };
  }

  const nameInp = document.getElementById("fsName");
  const descInp = document.getElementById("fsDesc");
  const presetSel = document.getElementById("fsPreset");
  if (nameInp && activeFs) nameInp.value = activeFs.name || "";
  if (descInp && activeFs) descInp.value = activeFs.description || "";
  if (presetSel) {
    presetSel.innerHTML = Object.keys(presets)
      .map((k) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`)
      .join("");
  }

  // 基础字段：来自 panel_info.fields（排除元掩码字段）
  const mountedExtFields = new Set();
  (st.extensions || [])
    .filter((e) => e.mounted)
    .forEach((e) => (e.fields || []).forEach((f) => mountedExtFields.add(f)));

  const ignoreFields = new Set(["sector", "industry_code", "listed_days", "in_universe_mask", "limit_mask"]);
  const allBase = (pInfo.fields || []).filter((f) => !ignoreFields.has(f) && !mountedExtFields.has(f));
  const selectedBase = new Set((activeFs && activeFs.base_fields) || allBase.slice(0, 8));

  const baseEl = document.getElementById("baseFields");
  if (baseEl) {
    baseEl.innerHTML = allBase.map((f) => chipHtml(f, selectedBase.has(f), "data-bf")).join("");
    bindChipToggle(baseEl);
  }

  const extEl = document.getElementById("extFields");
  if (extEl) {
    const extList = [...mountedExtFields];
    extEl.innerHTML = extList.length
      ? extList.map((f) => chipHtml(f, true, "data-ef")).join("")
      : '<span class="lbl">暂无已挂载扩展特征（可在左侧挂载 1m 扩展包）</span>';
    if (extList.length) bindChipToggle(extEl);
  }

  const selectedOps = new Set((activeFs && activeFs.operators) || BASE_OPS_CANDIDATES.slice(0, 9));
  const opsEl = document.getElementById("opsWl");
  if (opsEl) {
    opsEl.innerHTML = BASE_OPS_CANDIDATES.map((o) => chipHtml(o, selectedOps.has(o), "data-op")).join("");
    bindChipToggle(opsEl);
  }

  const selectedWins = new Set((activeFs && activeFs.windows) || [5, 10, 20, 40]);
  const winEl = document.getElementById("winWl");
  if (winEl) {
    winEl.innerHTML = WIN_CANDIDATES.map((w) => chipHtml(String(w), selectedWins.has(w), "data-win")).join("");
    bindChipToggle(winEl);
  }

  const buildPayload = (customName = null) => {
    const presetKey = presetSel ? presetSel.value : "standard";
    const mountedIds = (st.extensions || []).filter((e) => e.mounted).map((e) => e.qualified_id || e.pack_id);
    const chosenExtFields = activeChips(extEl, "data-ef");
    return {
      name: customName || (nameInp ? nameInp.value.trim() : "fs_custom"),
      description: descInp ? descInp.value.trim() : "",
      base_fields: [...activeChips(baseEl, "data-bf"), ...chosenExtFields],
      extensions: mountedIds,
      operators: activeChips(opsEl, "data-op"),
      windows: activeChips(winEl, "data-win").map(Number),
      default_postprocess: presets[presetKey] || presets.standard || [],
      upsert: true,
    };
  };

  const saveBtn = document.getElementById("fsSaveBtn");
  if (saveBtn) {
    saveBtn.onclick = async () => {
      try {
        const res = await apiPost(market, "/feature-sets", buildPayload());
        st.activeFeatureSetId = res.feature_set.id;
        toast(`FeatureSet ${res.feature_set.name} (v${res.feature_set.version}) 已保存`, "ok");
        await renderDataPage(navigate);
      } catch (err) {
        toast(`保存 FeatureSet 失败: ${err.message}`, "bad");
      }
    };
  }

  const copyBtn = document.getElementById("fsCopyBtn");
  if (copyBtn) {
    copyBtn.onclick = async () => {
      const baseName = (nameInp ? nameInp.value.trim() : "fs") || "fs";
      const copyName = `${baseName}_copy_${Date.now() % 1000}`;
      try {
        const res = await apiPost(market, "/feature-sets", buildPayload(copyName));
        st.activeFeatureSetId = res.feature_set.id;
        toast(`已复制为新 FeatureSet: ${copyName}`, "ok");
        await renderDataPage(navigate);
      } catch (err) {
        toast(`复制失败: ${err.message}`, "bad");
      }
    };
  }

  const toRecipeBtn = document.getElementById("fsToRecipeBtn");
  if (toRecipeBtn && navigate) {
    toRecipeBtn.onclick = () => navigate(market, "mining");
  }
}

async function loadAndRenderKline(market, st) {
  const ks = st.klineState;
  const basketParam = (ks.basket || []).join(",");
  try {
    const data = await apiGet(market, "/panel/kline", {
      symbol: ks.symbol || "",
      range_bars: ks.range,
      adj: ks.adj,
      basket: basketParam,
    });
    ks.data = data;
    if (!ks.symbol || !(data.symbols || []).some((s) => s.symbol === ks.symbol)) {
      ks.symbol = data.symbol || (data.symbols[0] && data.symbols[0].symbol) || null;
    }
    if (!ks.basket.length && data.symbols && data.symbols.length) {
      ks.basket = data.symbols.slice(0, Math.min(2, data.symbols.length)).map((s) => s.symbol);
    }
    renderKlineControls(market, st);
  } catch (err) {
    toast(`加载 K 线数据失败: ${err.message}`, "warn");
  }
}

function renderKlineControls(market, st) {
  const ks = st.klineState;
  const data = ks.data;
  if (!data) return;
  const list = data.symbols || [];
  const cur = list.find((s) => s.symbol === ks.symbol) || list[0];

  const symChip = document.getElementById("klSym");
  if (symChip && cur) {
    symChip.textContent = `${cur.symbol} · ${cur.sector} · 1d`;
  }

  const secs = ["全部", ...new Set(list.map((x) => x.sector))];
  const secEl = document.getElementById("klSec");
  if (secEl) {
    secEl.innerHTML = secs
      .map((s) => `<button type="button" class="chip ${s === ks.sector ? "active" : ""}" data-sec="${escapeHtml(s)}">${escapeHtml(s)}</button>`)
      .join("");
    secEl.querySelectorAll("[data-sec]").forEach((b) => {
      b.onclick = () => {
        ks.sector = b.getAttribute("data-sec");
        renderKlineControls(market, st);
      };
    });
  }

  const q = (ks.query || "").toLowerCase();
  const filtered = list.filter(
    (x) =>
      (ks.sector === "全部" || x.sector === ks.sector) &&
      `${x.symbol} ${x.sector}`.toLowerCase().includes(q)
  );
  const listEl = document.getElementById("klList");
  if (listEl) {
    listEl.innerHTML =
      filtered
        .map(
          (x) => `
      <li class="list-item flex items-center ${x.symbol === ks.symbol ? "mk-soft" : ""}">
        <button type="button" class="flex-1 px-2 py-1.5" style="text-align:left;" data-pick="${escapeHtml(x.symbol)}">
          <b>${escapeHtml(x.symbol)}</b>
          <span class="lbl"> ${escapeHtml(x.sector)} · <span class="${x.last_change >= 0 ? "pos" : "neg"}">${(x.last_change * 100).toFixed(2)}%</span></span>
        </button>
        <button type="button" class="px-2 mk-text" title="加入组合" data-add="${escapeHtml(x.symbol)}">＋</button>
      </li>`
        )
        .join("") || '<li class="p-2 lbl">无匹配标的</li>';

    listEl.querySelectorAll("[data-pick]").forEach((b) => {
      b.onclick = async () => {
        ks.symbol = b.getAttribute("data-pick");
        await loadAndRenderKline(market, st);
      };
    });
    listEl.querySelectorAll("[data-add]").forEach((b) => {
      b.onclick = async () => {
        const sym = b.getAttribute("data-add");
        if (!ks.basket.includes(sym) && ks.basket.length < 6) {
          ks.basket.push(sym);
          await loadAndRenderKline(market, st);
        }
      };
    });
  }

  const basketEl = document.getElementById("klBasket");
  if (basketEl) {
    basketEl.innerHTML =
      ks.basket
        .map(
          (sym) => `
      <span class="chip ${sym === ks.symbol ? "active" : ""}">
        <button type="button" data-bpick="${escapeHtml(sym)}">${escapeHtml(sym)}</button>
        <button type="button" data-brm="${escapeHtml(sym)}" aria-label="移除 ${escapeHtml(sym)}">×</button>
      </span>`
        )
        .join("") || '<span class="lbl">空</span>';
    basketEl.querySelectorAll("[data-bpick]").forEach((b) => {
      b.onclick = async () => {
        ks.symbol = b.getAttribute("data-bpick");
        await loadAndRenderKline(market, st);
      };
    });
    basketEl.querySelectorAll("[data-brm]").forEach((b) => {
      b.onclick = async () => {
        const sym = b.getAttribute("data-brm");
        ks.basket = ks.basket.filter((x) => x !== sym);
        await loadAndRenderKline(market, st);
      };
    });
  }

  const indEl = document.getElementById("klInd");
  if (indEl) {
    indEl.innerHTML = Object.keys(ks.ind)
      .map(
        (k) => `<label class="flex items-center gap-1.5 lbl"><input type="checkbox" data-ind="${k}" ${ks.ind[k] ? "checked" : ""}>${k}</label>`
      )
      .join("");
    indEl.querySelectorAll("[data-ind]").forEach((c) => {
      c.onchange = () => {
        ks.ind[c.getAttribute("data-ind")] = c.checked;
        drawKlineChart(
          document.getElementById("klSvg"),
          ks.data,
          ks,
          document.getElementById("klOhlc")
        );
      };
    });
  }

  const adjSel = document.getElementById("klAdj");
  if (adjSel) {
    adjSel.innerHTML =
      market === "futures"
        ? `<option value="hfq" ${ks.adj === "hfq" ? "selected" : ""}>主力连续(rule=3)·后复权</option><option value="raw" ${ks.adj === "raw" ? "selected" : ""}>不复权</option>`
        : `<option value="hfq" ${ks.adj === "hfq" ? "selected" : ""}>后复权</option><option value="raw" ${ks.adj === "raw" ? "selected" : ""}>不复权</option>`;
    adjSel.onchange = async () => {
      ks.adj = adjSel.value;
      await loadAndRenderKline(market, st);
    };
  }

  const subSel = document.getElementById("klSub");
  if (subSel) {
    subSel.value = ks.sub;
    subSel.onchange = () => {
      ks.sub = subSel.value;
      drawKlineChart(
        document.getElementById("klSvg"),
        ks.data,
        ks,
        document.getElementById("klOhlc")
      );
    };
  }

  const qInp = document.getElementById("klQ");
  if (qInp) {
    qInp.value = ks.query;
    qInp.oninput = () => {
      ks.query = qInp.value;
      renderKlineControls(market, st);
    };
  }

  document.querySelectorAll("#klMode button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.getAttribute("data-km") === ks.mode));
    b.onclick = () => {
      ks.mode = b.getAttribute("data-km");
      renderKlineControls(market, st);
    };
  });

  document.querySelectorAll("#klRange button").forEach((b) => {
    b.setAttribute("aria-pressed", String(Number(b.getAttribute("data-r")) === ks.range));
    b.onclick = async () => {
      ks.range = Number(b.getAttribute("data-r"));
      await loadAndRenderKline(market, st);
    };
  });

  drawKlineChart(
    document.getElementById("klSvg"),
    ks.data,
    ks,
    document.getElementById("klOhlc")
  );
}
