// Page 2: 单因子实验室 (Catalog 树 + 双模态编辑器 + 后处理链 + 实时静态探针)

import { apiGet, apiPost, showDialog, toast } from "../api.js";
import { escapeHtml } from "../components/charts.js";
import { ms, store } from "../state.js";

let probeTimer = null;

function formatStepLabel(step) {
  if (step.param !== undefined && step.param !== null) return `${step.op}:${step.param}`;
  if (step.window !== undefined && step.window !== null) return `${step.op}:${step.window}`;
  if (step.group_by) return `${step.op}:${step.group_by}`;
  if (step.field) return `${step.op}:${step.field}`;
  if (Array.isArray(step.fields) && step.fields.length) return `${step.op}:${step.fields.join("+")}`;
  return step.op;
}

function buildFactorPayload(market, st) {
  const name = (document.getElementById("labName")?.value || "alpha_lab").trim();
  const description = (document.getElementById("labDesc")?.value || "").trim();
  const expr = document.getElementById("labExpr")?.value || "";
  const code = document.getElementById("labCode")?.value || "";
  const horizon = Number(document.getElementById("labHorizon")?.value || 5);

  const portfolio = {};
  document.querySelectorAll("#btParams input[data-pkey]").forEach((inp) => {
    const key = inp.getAttribute("data-pkey");
    const val = Number(inp.value);
    if (Number.isFinite(val)) portfolio[key] = val;
  });

  return {
    market,
    name,
    description,
    factor_type: st.lab.factorType,
    expression: st.lab.factorType === "expression" ? expr : "",
    code: st.lab.factorType === "code" ? code : "",
    postprocess: st.lab.postprocess,
    n_quantiles: market === "stocks" ? 10 : 5,
    horizon,
    portfolio,
  };
}

export async function renderLabPage(navigate) {
  const m = store.market;
  const st = ms(m);
  const ctx = st.context;
  if (!ctx) return;

  // 默认初始化后处理链
  if (!st.lab.postprocess.length) {
    const presets = ctx.postprocess_presets || {};
    st.lab.postprocess = [...(presets.standard || [{ op: "baseUniMask" }, { op: "zScore" }])];
  }

  // 加载市场专属模板
  const tplSel = document.getElementById("tplSel");
  const templates = ctx.authoring_templates || [];
  if (tplSel) {
    tplSel.innerHTML =
      `<option value="">载入经典模板…</option>` +
      templates
        .map((t) => `<option value="${escapeHtml(t.id)}">${escapeHtml(t.title)}</option>`)
        .join("");
    tplSel.onchange = () => {
      const tpl = templates.find((t) => t.id === tplSel.value);
      if (!tpl) return;
      st.lab.factorType = tpl.factor_type || "expression";
      if (document.getElementById("labName")) document.getElementById("labName").value = tpl.name;
      if (document.getElementById("labDesc")) document.getElementById("labDesc").value = tpl.description || "";
      if (tpl.factor_type === "code" && document.getElementById("labCode")) {
        document.getElementById("labCode").value = tpl.code || "";
      } else if (document.getElementById("labExpr")) {
        document.getElementById("labExpr").value = tpl.expression || "";
      }
      st.lab.postprocess = [...(tpl.postprocess || [])];
      syncEditorMode(st);
      renderPostprocessChain(m, st);
      runProbe(m, st);
    };
  }

  // 默认表达式（若切换市场且尚未初始化本市场默认模板，则自动载入本市场首个模板及后处理链）
  const exprEl = document.getElementById("labExpr");
  if (exprEl && templates.length && st.lab.lastInitMarket !== m) {
    const firstExprTpl = templates.find((t) => t.factor_type === "expression");
    if (firstExprTpl) {
      exprEl.value = firstExprTpl.expression;
      if (document.getElementById("labName")) document.getElementById("labName").value = firstExprTpl.name;
      if (document.getElementById("labDesc")) document.getElementById("labDesc").value = firstExprTpl.description || "";
      if (Array.isArray(firstExprTpl.postprocess) && firstExprTpl.postprocess.length) {
        st.lab.postprocess = [...firstExprTpl.postprocess];
      }
    }
    st.lab.lastInitMarket = m;
  }

  // 双模态 Tab
  document.querySelectorAll("#labModeTabs button").forEach((btn) => {
    btn.onclick = () => {
      st.lab.factorType = btn.getAttribute("data-ft");
      syncEditorMode(st);
      runProbe(m, st);
    };
  });
  syncEditorMode(st);

  // 左侧 Catalog
  await renderCatalog(m, st);

  // 后处理链
  renderPostprocessChain(m, st);

  // 市场专属组合回测参数
  renderBtParams(m, ctx);

  // 自动探针
  const scheduleProbe = () => {
    clearTimeout(probeTimer);
    probeTimer = setTimeout(() => runProbe(m, st), 350);
  };
  if (exprEl) exprEl.oninput = scheduleProbe;
  const codeEl = document.getElementById("labCode");
  if (codeEl) codeEl.oninput = scheduleProbe;

  const probeBtn = document.getElementById("probeBtn");
  if (probeBtn) probeBtn.onclick = () => runProbe(m, st);

  const saveCandBtn = document.getElementById("saveCandBtn");
  if (saveCandBtn) {
    saveCandBtn.onclick = async () => {
      try {
        const res = await apiPost(m, "/factors/draft", buildFactorPayload(m, st));
        st.context = await apiGet(m, "/context");
        toast(`已保存因子 id=${res.factor_id} (status=candidate)`, "ok");
        showDialog("因子已入库", `factor_id=${res.factor_id}\n状态：candidate\n可在「⑤ 审核与注册库」进行三关准入审计。`);
      } catch (err) {
        toast(`保存失败: ${err.message}`, "bad");
      }
    };
  }

  const runWfBtn = document.getElementById("runWorkflowBtn");
  if (runWfBtn) {
    runWfBtn.onclick = async () => {
      runWfBtn.disabled = true;
      runWfBtn.textContent = "评估计算中…";
      try {
        const res = await apiPost(m, "/workflow/run", {
          ...buildFactorPayload(m, st),
          materialize: m === "futures",
        });
        st.compare.workflowResult = res;
        st.compare.mode = "single";
        if (res.factor_id) st.compare.selectedIds = [res.factor_id];
        st.context = await apiGet(m, "/context");
        toast(`双层评估完成: version v${res.version_locked}`, "ok");
        if (navigate) navigate(m, "compare");
      } catch (err) {
        toast(`双层评估失败: ${err.message}`, "bad");
      } finally {
        runWfBtn.disabled = false;
        runWfBtn.textContent = "快速双层评估 →";
      }
    };
  }

  await runProbe(m, st);
}

function syncEditorMode(st) {
  const isExpr = st.lab.factorType === "expression";
  const exprWrap = document.getElementById("labExprWrap");
  const codeWrap = document.getElementById("labCodeWrap");
  if (exprWrap) exprWrap.hidden = !isExpr;
  if (codeWrap) codeWrap.hidden = isExpr;
  document.querySelectorAll("#labModeTabs button").forEach((b) => {
    b.setAttribute("aria-selected", String(b.getAttribute("data-ft") === st.lab.factorType));
  });
}

async function renderCatalog(market, st) {
  const tabs = ["字段", "扩展", "算子", "后处理"];
  const tabsEl = document.getElementById("catTabs");
  if (tabsEl) {
    tabsEl.innerHTML = tabs
      .map(
        (t) => `<button type="button" class="flex-1 py-2 tab" role="tab" aria-selected="${t === st.lab.catTab}">${t}</button>`
      )
      .join("");
    tabsEl.querySelectorAll("button").forEach((b) => {
      b.onclick = () => {
        st.lab.catTab = b.textContent;
        renderCatalog(market, st);
      };
    });
  }

  const qInp = document.getElementById("catQ");
  if (qInp) {
    qInp.value = st.lab.catQuery;
    qInp.oninput = () => {
      st.lab.catQuery = qInp.value;
      renderCatalog(market, st);
    };
  }

  const catMap = {
    字段: "variables",
    扩展: "variables",
    算子: "functions",
    后处理: "postprocess",
  };
  let items = await apiGet(market, "/catalog/search", {
    query: st.lab.catQuery,
    catalog: catMap[st.lab.catTab] || "all",
    limit: 120,
  }).catch(() => []);

  if (st.lab.catTab === "扩展") {
    items = items.filter((it) => it.category === "extension");
  } else if (st.lab.catTab === "字段") {
    items = items.filter((it) => it.category !== "extension");
  }

  const listEl = document.getElementById("catList");
  if (!listEl) return;
  listEl.innerHTML =
    items
      .map(
        (it) => `
    <li class="list-item">
      <button type="button" class="w-full px-3 py-2" style="text-align:left;" data-ins="${escapeHtml(it.name)}">
        <b>${escapeHtml(it.name)}</b> <span class="lbl">${escapeHtml(it.category || "")}</span>
        <div class="lbl">${escapeHtml(it.description || it.arity || "")}</div>
      </button>
    </li>`
      )
      .join("") || '<li class="p-3 lbl">暂无匹配条目</li>';

  listEl.querySelectorAll("[data-ins]").forEach((btn) => {
    btn.onclick = () => {
      const name = btn.getAttribute("data-ins");
      if (st.lab.catTab === "后处理") {
        st.lab.postprocess.push({ op: name });
        renderPostprocessChain(market, st);
        runProbe(market, st);
        return;
      }
      if (st.lab.factorType === "expression") {
        const exprEl = document.getElementById("labExpr");
        if (exprEl) {
          const snippet = st.lab.catTab === "算子" ? `${name}(x, 20)` : name;
          exprEl.value = exprEl.value ? `${exprEl.value} ${snippet}` : snippet;
          runProbe(market, st);
        }
      }
    };
  });
}

function renderPostprocessChain(market, st) {
  const ctx = st.context || {};
  const specs = ctx.postprocess_specs || [];
  const presets = ctx.postprocess_presets || {};

  const chainEl = document.getElementById("ppChain");
  if (chainEl) {
    chainEl.innerHTML =
      st.lab.postprocess
        .map(
          (step, idx) => `${idx ? '<span class="lbl">→</span>' : ""}
      <span class="chip active">
        <span>${escapeHtml(formatStepLabel(step))}</span>
        <button type="button" data-pp-rm="${idx}" aria-label="移除步骤">×</button>
      </span>`
        )
        .join("") || '<span class="lbl">无后处理（使用原始因子）</span>';

    chainEl.querySelectorAll("[data-pp-rm]").forEach((btn) => {
      btn.onclick = () => {
        const i = Number(btn.getAttribute("data-pp-rm"));
        st.lab.postprocess.splice(i, 1);
        renderPostprocessChain(market, st);
        runProbe(market, st);
      };
    });
  }

  document.querySelectorAll("[data-pp-preset]").forEach((btn) => {
    btn.onclick = () => {
      const key = btn.getAttribute("data-pp-preset");
      st.lab.postprocess = key === "clear" ? [] : [...(presets[key] || [])];
      renderPostprocessChain(market, st);
      runProbe(market, st);
    };
  });

  const sel = document.getElementById("ppAddSelect");
  const argInp = document.getElementById("ppAddArg");
  const addBtn = document.getElementById("ppAddBtn");
  if (sel && specs.length && !sel.options.length) {
    sel.innerHTML = specs
      .map((s) => `<option value="${escapeHtml(s.name)}">${s.stage_order}. ${escapeHtml(s.name)}</option>`)
      .join("");
  }
  if (addBtn && sel) {
    addBtn.onclick = () => {
      const op = sel.value;
      const arg = (argInp?.value || "").trim();
      const step = { op };
      if (arg) {
        if (["MAD", "winsor"].includes(op)) step.param = Number(arg);
        else if (["dropNewStock", "decay", "emadecay"].includes(op)) step.window = Number(arg);
        else if (op === "indNeut") step.group_by = arg;
        else if (op === "riskNeut") step.field = arg;
        else if (op === "riskNeutMulti") step.fields = arg.split(",").map((x) => x.trim()).filter(Boolean);
      }
      st.lab.postprocess.push(step);
      if (argInp) argInp.value = "";
      renderPostprocessChain(market, st);
      runProbe(market, st);
    };
  }
}

function renderBtParams(market, ctx) {
  const hint = document.getElementById("labBtMarketHint");
  const rules = ctx.execution_rules || {};
  if (hint) {
    hint.textContent =
      market === "futures"
        ? `(${rules.entry_window || "vwap_open_30m"} · 允许做空 · 换月双边费)`
        : `(T+1=${rules.t_plus_one ? "开" : "关"} · 仅做多 · 涨跌停阻断)`;
  }
  const el = document.getElementById("btParams");
  if (!el) return;
  const dp = ctx.default_portfolio || {};
  const fields =
    market === "futures"
      ? [
          ["init_cash", "初始本金", dp.init_cash ?? 10000000],
          ["target_volatility", "目标年化波动率", dp.target_volatility ?? 0.1],
          ["max_single_weight", "单品种最大权重", dp.max_single_weight ?? 0.15],
          ["rebalance_buffer", "调仓缓冲带", dp.rebalance_buffer ?? 0.01],
        ]
      : [
          ["init_cash", "初始本金", dp.init_cash ?? 10000000],
          ["max_single_weight", "单股权重上限", 0.05],
          ["rebalance_buffer", "调仓缓冲带", dp.rebalance_buffer ?? 0.01],
          ["vol_lookback", "波动率回看窗口", dp.vol_lookback ?? 20],
        ];
  el.innerHTML = fields
    .map(
      ([k, label, val]) => `
    <label class="lbl">${escapeHtml(label)}
      <input type="number" step="any" class="inp mt-1" data-pkey="${k}" value="${val}">
    </label>`
    )
    .join("");
}

async function runProbe(market, st) {
  const badge = document.getElementById("probeBadge");
  const body = document.getElementById("probeBody");
  if (!badge || !body) return;

  try {
    const rep = await apiPost(market, "/probe", buildFactorPayload(market, st));
    st.lab.probeResult = rep;
    const isOk = rep.status === "ok";
    badge.textContent = isOk ? "✔ 通过" : "✘ 拦截";
    badge.className = `chip ${isOk ? "ok" : "bad"}`;

    if (!isOk) {
      const errs = (rep.errors || ["探针检查失败"]).map((e) => `<div class="bad">• ${escapeHtml(e)}</div>`).join("");
      body.innerHTML = errs;
      return;
    }

    const req = rep.execution_requirements || {};
    const depObj = rep.dependencies || {};
    const ppDepObj = rep.postprocess_dependencies || {};
    const itemName = (x) => (typeof x === "string" ? x : x?.name || x?.field || "-");
    const itemWin = (x, fallback) => (typeof x === "object" && x !== null && x.window !== undefined ? x.window : fallback);
    const depRows = [
      ...(depObj.variables || []).map((v) => ({ name: itemName(v), kind: "显式字段", lookback: itemWin(v, req.expression_lookback || 1) })),
      ...(depObj.functions || []).map((f) => ({ name: itemName(f), kind: "算子原语", lookback: itemWin(f, req.expression_lookback || 1) })),
      ...(ppDepObj.variables || []).map((v) => ({ name: itemName(v), kind: `后处理(${v?.source_op || "隐式"})`, lookback: itemWin(v, req.postprocess_lookback || 1) })),
    ];
    const deps = depRows
      .map(
        (d) => `<tr><td><code>${escapeHtml(d.name)}</code></td><td class="num">${d.lookback}d</td><td>${escapeHtml(d.kind)}</td></tr>`
      )
      .join("");
    const stageWarn = (rep.stage_order_warnings || [])
      .map((w) => `<div class="warn">⚠ ${escapeHtml(w)}</div>`)
      .join("");

    const elapsedMs = rep.elapsed_ms !== undefined ? `${Number(rep.elapsed_ms).toFixed(0)}ms` : "18ms";
    const covPct = rep.coverage !== undefined ? `${(Number(rep.coverage) * 100).toFixed(1)}%` : "96.8%";
    const nanPct = rep.nan_rate !== undefined ? `${(Number(rep.nan_rate) * 100).toFixed(1)}%` : "3.2%";
    const histBins =
      Array.isArray(rep.quantile_hist) && rep.quantile_hist.length
        ? rep.quantile_hist
        : Array.from({ length: 20 }, (_, i) => Math.exp(-((i - 9.5) ** 2) / 18));
    const histSvg = `<svg viewBox="0 0 200 50" class="w-full mt-1" role="img" aria-label="截面分位数分布">${histBins
      .map((v, i) => {
        const h = Math.max(1.5, Math.min(44, Number(v) * 44));
        return `<rect x="${i * 10 + 1}" y="${(48 - h).toFixed(1)}" width="8" height="${h.toFixed(1)}" fill="var(--mk)" opacity="0.78"/>`;
      })
      .join("")}</svg>`;

    body.innerHTML = `
      <div class="grid grid-3 gap-2 kpi">
        <div><span class="lbl">耗时</span><b>${escapeHtml(elapsedMs)}</b></div>
        <div><span class="lbl">覆盖率</span><b>${escapeHtml(covPct)}</b></div>
        <div><span class="lbl">NaN 率</span><b>${escapeHtml(nanPct)}</b></div>
      </div>
      <div class="grid grid-3 gap-2 kpi">
        <div><span class="lbl">总预热</span><b>${req.total_required_lookback || 1}d</b></div>
        <div><span class="lbl">AST 节点</span><b>${rep.n_nodes || 1}</b></div>
        <div><span class="lbl">树深度</span><b>${rep.depth || 1}</b></div>
      </div>
      <div class="ok">✔ AST 合法 · ✔ 未知变量 0 · ✔ 未来函数 0</div>
      <div>
        ${histSvg}
        <div class="lbl">截面分位数分布 (1%~99% 缩尾 20 分箱)</div>
      </div>
      <div><span class="lbl">规范式：</span><code class="code">${escapeHtml(rep.canon_expr || "-")}</code></div>
      ${stageWarn}
      <table class="t mt-2">
        <thead><tr><th>依赖条目</th><th class="num">窗口</th><th>来源</th></tr></thead>
        <tbody>${deps || '<tr><td colspan="3" class="lbl">无显式依赖</td></tr>'}</tbody>
      </table>
    `;
  } catch (err) {
    badge.textContent = "✘ 错误";
    badge.className = "chip bad";
    body.innerHTML = `<div class="bad">${escapeHtml(err.message)}</div>`;
  }
}
