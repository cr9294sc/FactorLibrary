// FactorLibrary 现代量化终端前端交互控制器
(function () {
  "use strict";

  const API_BASE = "";

  const state = {
    context: null,
    catalogItems: [],
    catalogCategory: "all",
    catalogQuery: "",
    factorType: "expression",
    postprocessSteps: [],
    currentProbe: null,
    lastWorkflowResult: null,
    factors: [],
    runs: [],
    probeTimer: null,
  };

  // =========================================================================
  // DOM 元素查询
  // =========================================================================
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const els = {
    panelBadge: $("#panel-badge"),
    btnSwitchPanel: $("#btn-switch-panel"),
    regBadge: $("#reg-badge"),
    toastContainer: $("#toast-container"),
    catalogCount: $("#catalog-count-badge"),
    catalogSearch: $("#catalog-search-input"),
    catalogPills: $("#catalog-pills"),
    catalogList: $("#catalog-list"),
    selectTemplate: $("#select-template"),
    factorName: $("#factor-name-input"),
    factorDesc: $("#factor-desc-input"),
    editorExprWrap: $("#editor-expr-wrap"),
    factorExpr: $("#factor-expr-input"),
    editorCodeWrap: $("#editor-code-wrap"),
    factorCode: $("#factor-code-input"),
    selectAddOp: $("#select-add-op"),
    opArgWrap: $("#op-arg-input-wrap"),
    btnAddStep: $("#btn-add-step"),
    stepsContainer: $("#steps-container"),
    probeStatus: $("#probe-status-pill"),
    probeCanonExpr: $("#probe-canon-expr"),
    probeCanonHash: $("#probe-canon-hash"),
    probeNodes: $("#probe-n-nodes"),
    probeDepth: $("#probe-depth"),
    lbExpr: $("#lb-expr"),
    lbPp: $("#lb-pp"),
    lbTotal: $("#lb-total"),
    lbWarmup: $("#lb-warmup"),
    stageWarning: $("#stage-warning-box"),
    depTbody: $("#dep-tbody"),
    btnSaveDraft: $("#btn-save-draft"),
    btnRunWorkflow: $("#btn-run-workflow"),
    btnOpenReport: $("#btn-open-report"),
    resultTitle: $("#result-title"),
    resultMeta: $("#result-meta"),
    chartQuantileNav: $("#chart-quantile-nav"),
    chartIcCum: $("#chart-ic-cum"),
    chartBtEquity: $("#chart-bt-equity"),
    styleTbody: $("#style-exposure-tbody"),
    annualTbody: $("#annual-attribution-tbody"),
    waterfallTbody: $("#waterfall-tbody"),
    costDragBadge: $("#cost-drag-badge"),
    monthlyHost: $("#monthly-table-host"),
    extremeTbody: $("#extreme-tbody"),
    regStatusFilter: $("#reg-status-filter"),
    regSourceFilter: $("#reg-source-filter"),
    btnInitSeeds: $("#btn-init-seeds"),
    btnRefreshFactors: $("#btn-refresh-factors"),
    factorsTbody: $("#factors-tbody"),
    factorModal: $("#factor-modal"),
    modalTitle: $("#modal-factor-title"),
    modalBody: $("#modal-factor-body"),
    btnCloseModal: $("#btn-close-modal"),
    pipeFactorInput: $("#pipe-factor-input"),
    pipeRunGp: $("#pipe-run-gp"),
    pipeGpOptions: $("#pipe-gp-options"),
    pipeGpGens: $("#pipe-gp-gens"),
    pipeGpPop: $("#pipe-gp-pop"),
    pipeMlModel: $("#pipe-ml-model"),
    btnRunPipeline: $("#btn-run-pipeline"),
    btnRefreshRuns: $("#btn-refresh-runs"),
    runsTbody: $("#runs-tbody"),
  };

  // =========================================================================
  // API 请求辅助函数
  // =========================================================================
  async function apiGet(endpoint, params = {}) {
    const url = new URL(API_BASE + endpoint, window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.append(k, v);
    });
    const res = await fetch(url.toString());
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  }

  async function apiPost(endpoint, body = {}) {
    const res = await fetch(API_BASE + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  }

  async function apiPatch(endpoint, body = {}) {
    const res = await fetch(API_BASE + endpoint, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  }

  async function apiDelete(endpoint) {
    const res = await fetch(API_BASE + endpoint, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  }

  function showToast(msg, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    els.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4000);
  }

  // =========================================================================
  // 初始化与上下文加载
  // =========================================================================
  async function initApp() {
    bindEvents();
    renderOpArgInput(els.selectAddOp.value);
    try {
      await loadContext();
      await loadCatalog();
      await loadFactors();
      await loadRuns();
    } catch (err) {
      showToast(`初始化失败: ${err.message}`, "error");
    }
  }

  async function loadContext() {
    const ctx = await apiGet("/api/context");
    state.context = ctx;

    // 更新 Header 面板状态指示
    const pInfo = ctx.panel_info || {};
    const isReal = pInfo.source === "real";
    els.panelBadge.textContent = `${isReal ? "真实研究面板" : "合成演示面板"} (${pInfo.n_days}天 x ${pInfo.n_symbols}品种)`;
    els.panelBadge.className = `status-pill ${isReal ? "pill-green" : "pill-blue"}`;

    const rSum = ctx.registry_summary || {};
    els.regBadge.textContent = `因子库: ${rSum.total || 0} 因子`;

    // 填充模板下拉列表
    els.selectTemplate.innerHTML = '<option value="">快速载入经典模板...</option>';
    (ctx.authoring_templates || []).forEach((tpl) => {
      const opt = document.createElement("option");
      opt.value = tpl.id;
      opt.textContent = tpl.title;
      els.selectTemplate.appendChild(opt);
    });

    // 载入首个推荐模板
    if (ctx.authoring_templates && ctx.authoring_templates.length > 0) {
      applyTemplate(ctx.authoring_templates[0]);
    }
  }

  function applyTemplate(tpl) {
    els.factorName.value = tpl.name || "alpha_factor";
    els.factorDesc.value = tpl.description || "";
    state.factorType = tpl.factor_type || "expression";

    const radio = document.querySelector(`input[name="factor_type"][value="${state.factorType}"]`);
    if (radio) radio.checked = true;
    updateEditorMode();

    if (state.factorType === "expression") {
      els.factorExpr.value = tpl.expression || "";
    } else {
      els.factorCode.value = tpl.code || "";
    }

    state.postprocessSteps = JSON.parse(JSON.stringify(tpl.postprocess || []));
    renderPostprocessSteps();
    triggerProbe();
  }

  // =========================================================================
  // 因子研究目录检索 (Catalog)
  // =========================================================================
  async function loadCatalog() {
    try {
      const items = await apiGet("/api/catalog/search", {
        query: state.catalogQuery,
        catalog: state.catalogCategory,
        limit: 100,
      });
      state.catalogItems = items;
      renderCatalogList();
    } catch (err) {
      els.catalogList.innerHTML = `<div class="empty-tip">目录加载失败: ${err.message}</div>`;
    }
  }

  function renderCatalogList() {
    els.catalogCount.textContent = state.catalogItems.length;
    if (state.catalogItems.length === 0) {
      els.catalogList.innerHTML = '<div class="empty-tip">未找到匹配的目录条目</div>';
      return;
    }
    const html = state.catalogItems
      .map((it) => {
        const formula = it.formula ? `<div class="ci-formula">${escapeHtml(it.formula)}</div>` : "";
        return `
        <div class="catalog-item" data-name="${escapeHtml(it.name)}" data-catalog="${it.catalog}">
          <div class="ci-head">
            <span class="ci-name">${escapeHtml(it.name)}</span>
            <span class="ci-tag">${it.category || it.catalog}</span>
          </div>
          <div class="ci-desc">${escapeHtml(it.description || "")}</div>
          ${formula}
        </div>
      `;
      })
      .join("");
    els.catalogList.innerHTML = html;

    // 点击目录条目快捷插入
    $$(".catalog-item").forEach((el) => {
      el.addEventListener("click", () => {
        const name = el.getAttribute("data-name");
        const cat = el.getAttribute("data-catalog");
        handleCatalogItemClick(name, cat);
      });
    });
  }

  function handleCatalogItemClick(name, catalog) {
    if (catalog === "postprocess") {
      // 切换并追加至后处理算子链
      state.postprocessSteps.push({ op: name });
      renderPostprocessSteps();
      triggerProbe();
      showToast(`已追加后处理步骤: ${name}()`, "info");
    } else {
      // 插入到当前激活的代码编辑器
      if (state.factorType === "expression") {
        insertTextToEditor(els.factorExpr, name);
      } else {
        insertTextToEditor(els.factorCode, `context.history("${name}", 20)`);
      }
      triggerProbe();
    }
  }

  function insertTextToEditor(textarea, text) {
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const val = textarea.value;
    textarea.value = val.substring(0, start) + text + val.substring(end);
    textarea.selectionStart = textarea.selectionEnd = start + text.length;
    textarea.focus();
  }

  // =========================================================================
  // 12 算子声明式后处理编排器
  // =========================================================================
  function renderOpArgInput(op) {
    let html = "";
    if (op === "dropNewStock") {
      html = '<input type="number" id="op-arg-val" class="form-input input-sm" style="width:80px;" value="60" title="上市交易日天数">';
    } else if (op === "MAD") {
      html = '<input type="number" id="op-arg-val" class="form-input input-sm" style="width:80px;" value="3.0" step="0.5" title="MAD 倍数">';
    } else if (op === "winsor") {
      html = '<input type="number" id="op-arg-val" class="form-input input-sm" style="width:80px;" value="0.025" step="0.005" title="双侧截尾比例">';
    } else if (op === "indNeut") {
      html = '<input type="text" id="op-arg-val" class="form-input input-sm" style="width:90px;" value="sector" title="板块分类列名">';
    } else if (op === "riskNeut") {
      html = `
        <select id="op-arg-val" class="form-select select-sm" style="width:110px;">
          <option value="SIZE">SIZE</option>
          <option value="VOLATILITY">VOLATILITY</option>
          <option value="MOMENTUM">MOMENTUM</option>
          <option value="CARRY">CARRY</option>
          <option value="INVENTORY">INVENTORY</option>
          <option value="POSITIONING">POSITIONING</option>
        </select>
      `;
    } else if (op === "riskNeutMulti") {
      html = '<input type="text" id="op-arg-val" class="form-input input-sm" style="width:150px;" value="SIZE,VOLATILITY,CARRY" title="风格暴露字段，逗号分隔">';
    } else if (op === "decay" || op === "emadecay") {
      html = '<input type="number" id="op-arg-val" class="form-input input-sm" style="width:70px;" value="5" min="1" title="滑动平滑窗口">';
    }
    els.opArgWrap.innerHTML = html;
  }

  function getOpStageOrder(op) {
    if (["baseUniMask", "trdMask", "dropNewStock"].includes(op)) return 1;
    if (["MAD", "winsor"].includes(op)) return 2;
    if (["indNeut", "riskNeut", "riskNeutMulti"].includes(op)) return 3;
    if (["zScore", "rank"].includes(op)) return 4;
    if (["decay", "emadecay"].includes(op)) return 5;
    return 1;
  }

  function renderPostprocessSteps() {
    if (state.postprocessSteps.length === 0) {
      els.stepsContainer.innerHTML = '<div class="empty-tip">未配置后处理步骤，将使用原始因子值</div>';
      return;
    }
    const html = state.postprocessSteps
      .map((step, idx) => {
        const stage = getOpStageOrder(step.op);
        let argText = "";
        if (step.param !== undefined && step.param !== null) argText = `(${step.param})`;
        else if (step.window !== undefined && step.window !== null) argText = `(${step.window}d)`;
        else if (step.group_by) argText = `(${step.group_by})`;
        else if (step.field) argText = `(${step.field})`;
        else if (step.fields) argText = `([${step.fields.join(",")}])`;

        return `
        <div class="step-chip stage-${stage}" data-idx="${idx}">
          <div class="flex-align-gap">
            <span class="text-xs text-muted">#${idx + 1}</span>
            <span class="step-op-title">${escapeHtml(step.op)}</span>
            <span class="text-xs text-dim">${argText}</span>
          </div>
          <div class="step-actions">
            <button class="btn btn-xs btn-ghost btn-step-up" data-idx="${idx}" ${idx === 0 ? "disabled" : ""}>↑</button>
            <button class="btn btn-xs btn-ghost btn-step-down" data-idx="${idx}" ${idx === state.postprocessSteps.length - 1 ? "disabled" : ""}>↓</button>
            <button class="btn btn-xs btn-ghost text-red btn-step-del" data-idx="${idx}">×</button>
          </div>
        </div>
      `;
      })
      .join("");
    els.stepsContainer.innerHTML = html;

    // 绑定步骤微调按钮
    $$(".btn-step-up").forEach((b) =>
      b.addEventListener("click", (e) => {
        const idx = parseInt(e.currentTarget.getAttribute("data-idx"));
        if (idx > 0) {
          const tmp = state.postprocessSteps[idx];
          state.postprocessSteps[idx] = state.postprocessSteps[idx - 1];
          state.postprocessSteps[idx - 1] = tmp;
          renderPostprocessSteps();
          triggerProbe();
        }
      })
    );
    $$(".btn-step-down").forEach((b) =>
      b.addEventListener("click", (e) => {
        const idx = parseInt(e.currentTarget.getAttribute("data-idx"));
        if (idx < state.postprocessSteps.length - 1) {
          const tmp = state.postprocessSteps[idx];
          state.postprocessSteps[idx] = state.postprocessSteps[idx + 1];
          state.postprocessSteps[idx + 1] = tmp;
          renderPostprocessSteps();
          triggerProbe();
        }
      })
    );
    $$(".btn-step-del").forEach((b) =>
      b.addEventListener("click", (e) => {
        const idx = parseInt(e.currentTarget.getAttribute("data-idx"));
        state.postprocessSteps.splice(idx, 1);
        renderPostprocessSteps();
        triggerProbe();
      })
    );
  }

  // =========================================================================
  // Stage 3 毫秒级静态探针实时诊断 (Probe)
  // =========================================================================
  function getFactorPayload() {
    return {
      name: els.factorName.value.trim() || "unnamed_factor",
      factor_type: state.factorType,
      expression: els.factorExpr.value.trim(),
      code: els.factorCode.value,
      postprocess: state.postprocessSteps,
      description: els.factorDesc.value.trim(),
      portfolio: {
        init_cash: parseFloat(els.cfgInitCash ? els.cfgInitCash.value : 10000000),
        target_volatility: parseFloat($("#cfg-target-vol").value || 0.1),
        rebalance_buffer: parseFloat($("#cfg-buffer").value || 0.01),
        max_single_weight: parseFloat($("#cfg-max-weight").value || 0.15),
      },
    };
  }

  function triggerProbe() {
    clearTimeout(state.probeTimer);
    state.probeTimer = setTimeout(async () => {
      await runProbe();
    }, 150);
  }

  async function runProbe() {
    const payload = getFactorPayload();
    try {
      const rep = await apiPost("/api/probe", payload);
      state.currentProbe = rep;
      renderProbeReport(rep);
    } catch (err) {
      renderProbeError(err.message);
    }
  }

  function renderProbeReport(rep) {
    const isOk = rep.status === "ok";
    els.probeStatus.textContent = isOk ? "PASS" : "FAIL";
    els.probeStatus.className = `status-pill ${isOk ? "pill-green" : "pill-red"}`;

    els.probeCanonExpr.textContent = rep.canon_expr || "-";
    els.probeCanonHash.textContent = rep.canon_hash || "-";
    els.probeNodes.textContent = rep.n_nodes !== undefined ? rep.n_nodes : "-";
    els.probeDepth.textContent = rep.depth !== undefined ? rep.depth : "-";

    const req = rep.execution_requirements || {};
    els.lbExpr.textContent = `${req.expression_lookback || 1} 日`;
    els.lbPp.textContent = `${req.postprocess_lookback || 0} 日`;
    els.lbTotal.textContent = `${req.total_required_lookback || 1} 日`;
    els.lbWarmup.textContent = req.warmup_bars !== undefined ? req.warmup_bars : 0;

    // 阶段顺序告警
    if (rep.stage_order_warnings && rep.stage_order_warnings.length > 0) {
      els.stageWarning.style.display = "block";
      els.stageWarning.innerHTML = "<b>⚠️ 后处理阶段顺序提示:</b><br>" + rep.stage_order_warnings.join("<br>");
    } else {
      els.stageWarning.style.display = "none";
    }

    // 依赖清单表格
    const explicitVars = (rep.dependencies && rep.dependencies.variables) || [];
    const ppVars = (rep.postprocess_dependencies && rep.postprocess_dependencies.variables) || [];
    const allDeps = [
      ...explicitVars.map((v) => ({ name: v.name, window: v.window, source: "主表达式" })),
      ...ppVars.map((v) => ({ name: v.name, window: v.window, source: v.source_op || "后处理" })),
    ];

    if (allDeps.length === 0) {
      els.depTbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">无特殊外部字段依赖</td></tr>';
    } else {
      els.depTbody.innerHTML = allDeps
        .map(
          (d) => `
        <tr>
          <td><b>${escapeHtml(d.name)}</b></td>
          <td><span class="font-mono text-green">${d.window}d</span></td>
          <td><span class="text-xs text-muted">${escapeHtml(d.source)}</span></td>
        </tr>
      `
        )
        .join("");
    }
  }

  function renderProbeError(errMsg) {
    els.probeStatus.textContent = "ERROR";
    els.probeStatus.className = "status-pill pill-red";
    els.stageWarning.style.display = "block";
    els.stageWarning.innerHTML = `<b>语法校验错误:</b><br>${escapeHtml(errMsg)}`;
  }

  // =========================================================================
  // Stage 4 & 5: 执行双层回测与版本锁定
  // =========================================================================
  async function runDualLayerWorkflow() {
    const payload = getFactorPayload();
    els.btnRunWorkflow.disabled = true;
    els.btnRunWorkflow.textContent = "正在执行双层回测...";
    showToast("正在调用 Rust 内核执行逐日盯市与 Layer 1 归因...", "info");

    try {
      const res = await apiPost("/api/workflow/run", payload);
      state.lastWorkflowResult = res;
      showToast(`回测完成！不可变版本已锁定: v${res.version_locked || 1}`, "success");
      renderAnalyticsView(res);
      switchTab("tab-analytics");
      await loadFactors();
    } catch (err) {
      showToast(`回测执行失败: ${err.message}`, "error");
    } finally {
      els.btnRunWorkflow.disabled = false;
      els.btnRunWorkflow.textContent = "执行双层回测与版本锁定 🚀";
    }
  }

  async function saveDraft() {
    const payload = getFactorPayload();
    try {
      const res = await apiPost("/api/factors/draft", payload);
      showToast(`草稿已保存 (Factor ID: ${res.factor_id})`, "success");
      await loadFactors();
    } catch (err) {
      showToast(`草稿保存失败: ${err.message}`, "error");
    }
  }

  // =========================================================================
  // 视图 2：双层回测与归因深度看板渲染
  // =========================================================================
  function renderAnalyticsView(data) {
    els.resultTitle.textContent = `回测与归因诊断: ${data.definition.name} (v${data.version_locked || 1})`;
    els.resultMeta.textContent = `代数指纹: ${data.probe_report.canon_hash} | 运行批次: ${data.run_id}`;
    if (data.report_url) {
      els.btnOpenReport.href = data.report_url;
      els.btnOpenReport.style.display = "inline-flex";
    }

    const v4 = data.v4_stats || {};
    const bt = data.backtest || {};

    // 1. Layer 1 KPI 卡片
    const icSum = (v4.ic_analysis && v4.ic_analysis.summary) || {};
    const longSum = (v4.long_analysis && v4.long_analysis.summary) || {};

    $("#kpi-ric-mean").textContent = formatSigned(icSum.rank_ic_mean, 4);
    $("#kpi-ric-ir").textContent = `IR: ${formatNum(icSum.rank_ic_ir, 2)}`;
    $("#kpi-pic-mean").textContent = formatSigned(icSum.perc_ic_mean, 4);
    $("#kpi-pic-ir").textContent = `IR: ${formatNum(icSum.perc_ic_ir, 2)}`;
    $("#kpi-ic5-mean").textContent = formatSigned(icSum.ic_5d_mean, 4);
    const rankIc5dText =
      icSum.rank_ic_5d_mean !== undefined
        ? ` | 5d RankIC: ${formatSigned(icSum.rank_ic_5d_mean, 4)}`
        : "";
    $("#kpi-ic5-ir").textContent = `IR: ${formatNum(icSum.ic_5d_ir, 2)}${rankIc5dText}`;
    $("#kpi-winrate").textContent = formatPct(icSum.win_rate);
    $("#kpi-excess-ret").textContent = formatPctSigned(longSum.annual_excess_return);
    $("#kpi-excess-sharpe").textContent = `超额夏普: ${formatNum(longSum.excess_sharpe, 2)}`;
    $("#kpi-excess-dd").textContent = formatPct(longSum.additive_max_drawdown);
    $("#kpi-excess-turnover").textContent = `日均换手: ${formatPct(longSum.daily_turnover)}`;

    // 2. Layer 1 SVG 图表 (五分组净值 + 累计 IC)
    renderQuantileNavChart(v4.group_analysis, v4.dates);
    renderIcCdfChart(v4.ic_analysis, v4.long_analysis, v4.dates);

    // 3. 6 大风格暴露归因表
    const styleExp = v4.style_exposure || {};
    const lsExp = styleExp.long_short_exposure || {};
    const styles = styleExp.styles || [];
    if (styles.length === 0) {
      els.styleTbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">暂无风格归因数据</td></tr>';
    } else {
      els.styleTbody.innerHTML = styles
        .map((s) => {
          const lVal = styleExp.long_exposure ? styleExp.long_exposure[s] : 0;
          const sVal = styleExp.short_exposure ? styleExp.short_exposure[s] : 0;
          const lsVal = lsExp[s] || 0;
          const rCorr = styleExp.rank_corr ? styleExp.rank_corr[s] : 0;
          const barW = Math.min(100, Math.abs(lsVal) * 80);
          const barBg = lsVal >= 0 ? "#10b981" : "#ef4444";
          return `
          <tr>
            <td><b>${s}</b></td>
            <td>${formatSigned(lVal, 3)}</td>
            <td>${formatSigned(sVal, 3)}</td>
            <td>
              ${formatSigned(lsVal, 3)}
              <span style="display:inline-block;vertical-align:middle;margin-left:6px;width:${barW}px;height:7px;background:${barBg};border-radius:2px;"></span>
            </td>
            <td>${formatSigned(rCorr, 3)}</td>
          </tr>
        `;
        })
        .join("");
    }

    // 4. 分年度归因表
    const icAnn = (v4.ic_analysis && v4.ic_analysis.annual) || {};
    const longAnn = (v4.long_analysis && v4.long_analysis.annual) || {};
    const years = Array.from(new Set([...Object.keys(icAnn), ...Object.keys(longAnn)])).sort();
    if (years.length === 0) {
      els.annualTbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">无分年度数据</td></tr>';
    } else {
      els.annualTbody.innerHTML = years
        .map((y) => {
          const ia = icAnn[y] || {};
          const la = longAnn[y] || {};
          return `
          <tr>
            <td><b>${y}</b></td>
            <td>${formatSigned(ia.rank_ic_mean, 4)}</td>
            <td>${formatNum(ia.rank_ic_ir, 2)}</td>
            <td>${formatSigned(ia.perc_ic_mean, 4)}</td>
            <td>${formatSigned(ia.ic_5d_mean, 4)}</td>
            <td class="text-green">${formatPctSigned(la.annual_excess_return)}</td>
            <td>${formatNum(la.excess_sharpe, 2)}</td>
            <td class="text-red">${formatPct(la.additive_max_drawdown)}</td>
          </tr>
        `;
        })
        .join("");
    }

    // 5. Layer 2 实盘 MTM KPI 卡片
    const m = bt.metrics || {};
    $("#kpi-tot-ret").textContent = formatPctSigned(m.total_return);
    $("#kpi-ann-ret").textContent = `年化: ${formatPct(m.annual_return)}`;
    $("#kpi-sharpe").textContent = formatNum(m.sharpe, 2);
    $("#kpi-vol").textContent = `波动: ${formatPct(m.annual_volatility)}`;
    $("#kpi-max-dd").textContent = formatPct(m.max_drawdown);
    $("#kpi-calmar").textContent = `卡尔玛: ${formatNum(m.calmar, 2)}`;
    $("#kpi-bt-winrate").textContent = formatPct(m.win_rate);
    $("#kpi-plr").textContent = `盈亏比: ${formatNum(m.profit_loss_ratio, 2)}`;
    $("#kpi-turnover").textContent = `${formatNum(m.annual_turnover, 1)}x`;
    $("#kpi-daily-turnover").textContent = `日均: ${formatPct(m.daily_turnover_mean)}`;
    $("#kpi-max-margin").textContent = formatPct(m.max_margin_utilization);
    $("#kpi-max-leverage").textContent = `最大杠杆: ${formatNum(m.max_leverage, 2)}x`;

    // 6. Layer 2 实盘净值折线图
    renderBtEquityChart(bt.series, bt.dates);

    // 7. 摩擦损耗瀑布
    const cb = bt.cost_breakdown || {};
    els.costDragBadge.textContent = formatPct(cb.cost_drag);
    els.costDragBadge.className = `badge ${cb.cost_drag > 0.3 ? "badge-red" : "badge-green"}`;
    els.waterfallTbody.innerHTML = `
      <tr><td><b>盯市毛收益 (Gross PnL)</b></td><td><b>¥${formatMoney(cb.gross_pnl)}</b></td><td>100%</td><td>理论无摩擦收益</td></tr>
      <tr><td>- 手续费 (Commissions)</td><td>-¥${formatMoney(cb.fees)}</td><td>${formatPct(cb.gross_pnl ? cb.fees / cb.gross_pnl : 0)}</td><td>开平仓双向按比例或按手扣除</td></tr>
      <tr><td>- 冲击滑点 (Slippage)</td><td>-¥${formatMoney(cb.slippage)}</td><td>${formatPct(cb.gross_pnl ? cb.slippage / cb.gross_pnl : 0)}</td><td>品种跳数滑点成本</td></tr>
      <tr><td>* 换月展期摩擦 (Roll)</td><td>¥${formatMoney(cb.roll_cost)}</td><td>${formatPct(cb.gross_pnl ? cb.roll_cost / cb.gross_pnl : 0)}</td><td>主力平旧开新交易费与滑点</td></tr>
      <tr><td><b>实际净盈亏 (Net PnL)</b></td><td><b>¥${formatMoney(cb.net_pnl)}</b></td><td><b>${formatPct(cb.gross_pnl ? cb.net_pnl / cb.gross_pnl : 0)}</b></td><td>投资者最终到手权益变动</td></tr>
    `;

    // 8. 月度收益热力表
    renderMonthlyTable(bt.monthly_returns);

    // 9. 极端交易日
    const ext = bt.extreme_days || {};
    const best = ext.best || [];
    const worst = ext.worst || [];
    const rows = [];
    for (let i = 0; i < Math.max(best.length, worst.length); i++) {
      const b = best[i] || { date: "-", return: 0 };
      const w = worst[i] || { date: "-", return: 0 };
      rows.push(`
        <tr>
          <td>#${i + 1}</td>
          <td class="text-green">${b.date} (+${(b.return * 100).toFixed(2)}%)</td>
          <td class="text-red">${w.date} (${(w.return * 100).toFixed(2)}%)</td>
        </tr>
      `);
    }
    els.extremeTbody.innerHTML = rows.join("");
  }

  // =========================================================================
  // SVG 图表生成器 (零依赖)
  // =========================================================================
  function renderQuantileNavChart(groupAnalysis, dates) {
    if (!groupAnalysis || !groupAnalysis.nav_series) {
      els.chartQuantileNav.innerHTML = '<div class="empty-tip">暂无分组数据</div>';
      return;
    }
    const seriesMap = groupAnalysis.nav_series;
    const bench = groupAnalysis.benchmark_nav || [];
    const keys = Object.keys(seriesMap).sort();
    if (keys.length === 0) return;

    const T = seriesMap[keys[0]].length;
    if (T <= 1) return;

    let minV = 0.95;
    let maxV = 1.05;
    keys.forEach((k) => {
      seriesMap[k].forEach((v) => {
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      });
    });

    const w = 600;
    const h = 220;
    const padL = 45;
    const padR = 20;
    const padT = 24;
    const padB = 24;
    const pw = w - padL - padR;
    const ph = h - padT - padB;
    const span = Math.max(maxV - minV, 0.01);

    const colors = ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#10b981"];
    const polyLines = keys
      .map((k, idx) => {
        const pts = seriesMap[k].map((v, i) => {
          const x = padL + (i / (T - 1)) * pw;
          const y = padT + (1.0 - (v - minV) / span) * ph;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        });
        const c = colors[idx % colors.length];
        return `<polyline fill="none" stroke="${c}" stroke-width="1.8" points="${pts.join(" ")}" />`;
      })
      .join("");

    let benchLine = "";
    if (bench.length === T) {
      const bPts = bench.map((v, i) => {
        const x = padL + (i / (T - 1)) * pw;
        const y = padT + (1.0 - (v - minV) / span) * ph;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      benchLine = `<polyline fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,3" points="${bPts.join(" ")}" />`;
    }

    const svg = `
      <svg viewBox="0 0 ${w} ${h}" class="chart-svg">
        <rect x="${padL}" y="${padT}" width="${pw}" height="${ph}" fill="#111827" stroke="#334155" />
        <text x="5" y="${padT + 10}" fill="#94a3b8" font-size="10">${maxV.toFixed(2)}</text>
        <text x="5" y="${padT + ph}" fill="#94a3b8" font-size="10">${minV.toFixed(2)}</text>
        ${benchLine}
        ${polyLines}
        <text x="${padL}" y="${h - 6}" fill="#64748b" font-size="10">${dates[0] || ""}</text>
        <text x="${padL + pw - 60}" y="${h - 6}" fill="#64748b" font-size="10">${dates[dates.length - 1] || ""}</text>
      </svg>
    `;
    els.chartQuantileNav.innerHTML = svg;
  }

  function renderIcCdfChart(icAnalysis, longAnalysis, dates) {
    const s = (icAnalysis && icAnalysis.series) || {};
    const ric = s.rank_ic || [];
    const T = ric.length;
    if (T <= 1) {
      els.chartIcCum.innerHTML = '<div class="empty-tip">暂无 IC 数据</div>';
      return;
    }

    const cumRic = [];
    let acc = 0;
    ric.forEach((v) => {
      acc += v || 0;
      cumRic.push(acc);
    });

    const w = 600;
    const h = 220;
    const padL = 45;
    const padR = 20;
    const padT = 24;
    const padB = 24;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    const minV = Math.min(...cumRic);
    const maxV = Math.max(...cumRic);
    const span = Math.max(maxV - minV, 0.1);

    const pts = cumRic.map((v, i) => {
      const x = padL + (i / (T - 1)) * pw;
      const y = padT + (1.0 - (v - minV) / span) * ph;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const svg = `
      <svg viewBox="0 0 ${w} ${h}" class="chart-svg">
        <rect x="${padL}" y="${padT}" width="${pw}" height="${ph}" fill="#111827" stroke="#334155" />
        <text x="5" y="${padT + 10}" fill="#94a3b8" font-size="10">${maxV.toFixed(1)}</text>
        <text x="5" y="${padT + ph}" fill="#94a3b8" font-size="10">${minV.toFixed(1)}</text>
        <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="${pts.join(" ")}" />
        <text x="${padL}" y="${h - 6}" fill="#64748b" font-size="10">${dates[0] || ""}</text>
        <text x="${padL + pw - 60}" y="${h - 6}" fill="#64748b" font-size="10">${dates[dates.length - 1] || ""}</text>
      </svg>
    `;
    els.chartIcCum.innerHTML = svg;
  }

  function renderBtEquityChart(series, dates) {
    const eq = (series && series.equity) || [];
    const dd = (series && series.drawdown) || [];
    const T = eq.length;
    if (T <= 1) {
      els.chartBtEquity.innerHTML = '<div class="empty-tip">暂无回测净值数据</div>';
      return;
    }

    const normEq = eq.map((v) => v / eq[0]);
    const minEq = Math.min(...normEq);
    const maxEq = Math.max(...normEq);
    const spanEq = Math.max(maxEq - minEq, 0.01);

    const minDd = Math.min(...dd);
    const spanDd = Math.max(Math.abs(minDd), 0.01);

    const w = 960;
    const h = 260;
    const hEq = 170;
    const hDd = 70;
    const padL = 50;
    const padR = 20;
    const pw = w - padL - padR;

    const ptsEq = normEq.map((v, i) => {
      const x = padL + (i / (T - 1)) * pw;
      const y = 15 + (1.0 - (v - minEq) / spanEq) * (hEq - 25);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const ptsDd = [
      `${padL},${hEq + 15}`,
      ...dd.map((v, i) => {
        const x = padL + (i / (T - 1)) * pw;
        const y = hEq + 15 + (Math.abs(v) / spanDd) * (hDd - 20);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }),
      `${padL + pw},${hEq + 15}`,
    ];

    const svg = `
      <svg viewBox="0 0 ${w} ${h}" class="chart-svg">
        <rect x="${padL}" y="10" width="${pw}" height="${hEq - 15}" fill="#111827" stroke="#334155" />
        <text x="5" y="22" fill="#3b82f6" font-size="10">${maxEq.toFixed(2)}</text>
        <text x="5" y="${hEq - 10}" fill="#3b82f6" font-size="10">${minEq.toFixed(2)}</text>
        <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="${ptsEq.join(" ")}" />

        <rect x="${padL}" y="${hEq + 15}" width="${pw}" height="${hDd}" fill="#1c1917" stroke="#44403c" />
        <text x="5" y="${hEq + 27}" fill="#ef4444" font-size="10">0%</text>
        <text x="5" y="${hEq + hDd + 10}" fill="#ef4444" font-size="10">${(minDd * 100).toFixed(1)}%</text>
        <polygon fill="rgba(239, 68, 68, 0.3)" stroke="#ef4444" stroke-width="1.2" points="${ptsDd.join(" ")}" />

        <text x="${padL}" y="${h - 5}" fill="#64748b" font-size="10">${dates[0] || ""}</text>
        <text x="${padL + pw - 60}" y="${h - 5}" fill="#64748b" font-size="10">${dates[dates.length - 1] || ""}</text>
      </svg>
    `;
    els.chartBtEquity.innerHTML = svg;
  }

  function renderMonthlyTable(monthlyList) {
    if (!monthlyList || monthlyList.length === 0) {
      els.monthlyHost.innerHTML = '<div class="empty-tip">暂无月度数据</div>';
      return;
    }
    let html = `
      <table class="data-table table-sm text-center">
        <thead>
          <tr>
            <th>年份</th>
            <th>01月</th><th>02月</th><th>03月</th><th>04月</th>
            <th>05月</th><th>06月</th><th>07月</th><th>08月</th>
            <th>09月</th><th>10月</th><th>11月</th><th>12月</th>
            <th>全年合计</th>
          </tr>
        </thead>
        <tbody>
    `;
    monthlyList.forEach((row) => {
      html += `<tr><td><b>${row.year}</b></td>`;
      for (let m = 1; m <= 12; m++) {
        const k = m < 10 ? `0${m}` : `${m}`;
        const val = row.months[k] || 0;
        const cls = val > 0 ? "text-green" : val < 0 ? "text-red" : "text-muted";
        html += `<td class="${cls}">${(val * 100).toFixed(2)}%</td>`;
      }
      const yVal = row.year_total || 0;
      const yCls = yVal >= 0 ? "text-green" : "text-red";
      html += `<td class="${yCls}"><b>${(yVal * 100).toFixed(2)}%</b></td></tr>`;
    });
    html += "</tbody></table>";
    els.monthlyHost.innerHTML = html;
  }

  // =========================================================================
  // 视图 3：SQLite 因子库资产与不可变版本中心
  // =========================================================================
  async function loadFactors() {
    const status = els.regStatusFilter.value;
    const source = els.regSourceFilter.value;
    try {
      const list = await apiGet("/api/factors", { status, source });
      state.factors = list;
      renderFactorsTable();
    } catch (err) {
      els.factorsTbody.innerHTML = `<tr><td colspan="10" class="text-center text-red">加载因子失败: ${err.message}</td></tr>`;
    }
  }

  function renderFactorsTable() {
    if (state.factors.length === 0) {
      els.factorsTbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">因子库为空，点击右上角一键初始化种子因子</td></tr>';
      return;
    }
    const html = state.factors
      .map((f) => {
        const statusPill = `<span class="status-pill ${
          f.status === "accepted" ? "pill-green" : f.status === "active" ? "pill-blue" : "pill-gray"
        }">${f.status}</span>`;

        const m = f.primary_metric || {};
        return `
        <tr>
          <td>${f.id}</td>
          <td><b>${escapeHtml(f.name || `factor_${f.id}`)}</b></td>
          <td><span class="ci-tag">${f.factor_type}</span></td>
          <td class="font-mono text-xs" style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
            ${escapeHtml(f.canon_expr)}
          </td>
          <td>${statusPill}</td>
          <td>${f.source}</td>
          <td>${formatSigned(m.ic_mean, 4)}</td>
          <td>${formatNum(m.ic_ir, 2)}</td>
          <td><span class="badge">v${f.latest_version || 1} (${f.version_count}版)</span></td>
          <td>
            <div class="flex-align-gap">
              <button class="btn btn-xs btn-outline btn-load-factor" data-id="${f.id}">载入</button>
              <button class="btn btn-xs btn-secondary btn-detail-factor" data-id="${f.id}">版本</button>
              <button class="btn btn-xs btn-ghost text-red btn-del-factor" data-id="${f.id}">删</button>
            </div>
          </td>
        </tr>
      `;
      })
      .join("");
    els.factorsTbody.innerHTML = html;

    $$(".btn-load-factor").forEach((b) =>
      b.addEventListener("click", (e) => {
        const fid = parseInt(e.currentTarget.getAttribute("data-id"));
        loadFactorToIde(fid);
      })
    );
    $$(".btn-detail-factor").forEach((b) =>
      b.addEventListener("click", (e) => {
        const fid = parseInt(e.currentTarget.getAttribute("data-id"));
        openFactorModal(fid);
      })
    );
    $$(".btn-del-factor").forEach((b) =>
      b.addEventListener("click", async (e) => {
        const fid = parseInt(e.currentTarget.getAttribute("data-id"));
        if (confirm(`确定删除因子 #${fid} 及其所有历史版本吗？`)) {
          await apiDelete(`/api/factors/${fid}`);
          showToast(`已删除因子 #${fid}`, "info");
          await loadFactors();
        }
      })
    );
  }

  async function loadFactorToIde(factorId) {
    const f = await apiGet(`/api/factors/${factorId}`);
    if (!f) return;
    els.factorName.value = f.name || `factor_${f.id}`;
    els.factorDesc.value = (f.definition && f.definition.description) || "";
    state.factorType = f.factor_type || "expression";
    const radio = document.querySelector(`input[name="factor_type"][value="${state.factorType}"]`);
    if (radio) radio.checked = true;
    updateEditorMode();

    if (state.factorType === "expression") {
      els.factorExpr.value = (f.definition && f.definition.expression) || f.canon_expr.split(" |> ")[0];
    } else {
      els.factorCode.value = (f.definition && f.definition.code) || "";
    }

    if (f.definition && f.definition.postprocess) {
      state.postprocessSteps = f.definition.postprocess;
    } else {
      state.postprocessSteps = [];
    }
    renderPostprocessSteps();
    triggerProbe();
    switchTab("tab-ide");
    showToast(`已将因子 #${factorId} 载入工作台编辑器`, "success");
  }

  async function openFactorModal(factorId) {
    els.modalBody.innerHTML = '<div class="loading-spinner">加载因子详情与版本快照...</div>';
    els.factorModal.style.display = "flex";
    try {
      const f = await apiGet(`/api/factors/${factorId}`);
      els.modalTitle.textContent = `因子资产详情: ${f.name || `factor_${f.id}`} (ID: ${f.id})`;
      const versions = f.versions || [];
      const versionsHtml = versions
        .map(
          (v) => `
        <div class="card" style="margin-bottom:8px;">
          <div class="flex-between">
            <div><b>版本 v${v.version}</b> <span class="text-xs text-muted">锁定时间: ${v.created_at || "-"}</span></div>
            <button class="btn btn-xs btn-outline btn-restore-version" data-fid="${f.id}" data-v="${v.version}">载入此版本快照</button>
          </div>
          <div class="font-mono text-xs" style="margin-top:6px;word-break:break-all;color:#38bdf8;">${escapeHtml(v.canon_expr)}</div>
        </div>
      `
        )
        .join("");

      els.modalBody.innerHTML = `
        <div style="margin-bottom:12px;">
          <p><b>生命周期状态:</b> <span class="badge">${f.status}</span> 
             <b>推荐方向 sign:</b> ${f.sign === 1 ? "+1 (正向)" : "-1 (反向)"} 
             <b>代数指纹:</b> <code class="font-mono">${f.canon_hash}</code>
          </p>
          <div class="flex-align-gap" style="margin-top:8px;">
            <span>状态流转:</span>
            <button class="btn btn-xs btn-outline btn-set-status" data-fid="${f.id}" data-status="accepted">设为 Accepted</button>
            <button class="btn btn-xs btn-outline btn-set-status" data-fid="${f.id}" data-status="active">设为 Active (入生产池)</button>
            <button class="btn btn-xs btn-outline btn-set-status" data-fid="${f.id}" data-status="deprecated">设为 Deprecated</button>
          </div>
        </div>
        <h4>不可变历史版本快照 (Locked Versions)</h4>
        <div>${versionsHtml || '<div class="empty-tip">暂无锁定版本快照</div>'}</div>
      `;

      $$(".btn-set-status").forEach((b) =>
        b.addEventListener("click", async (e) => {
          const fid = e.currentTarget.getAttribute("data-fid");
          const st = e.currentTarget.getAttribute("data-status");
          await apiPatch(`/api/factors/${fid}/status`, { status: st });
          showToast(`因子 #${fid} 状态已更新为 ${st}`, "success");
          await openFactorModal(fid);
          await loadFactors();
        })
      );
      $$(".btn-restore-version").forEach((b) =>
        b.addEventListener("click", (e) => {
          const vNum = parseInt(e.currentTarget.getAttribute("data-v"));
          const targetV = versions.find((x) => x.version === vNum);
          if (targetV && targetV.definition) {
            els.factorName.value = targetV.definition.name || f.name;
            state.factorType = targetV.definition.factor_type || "expression";
            const radio = document.querySelector(`input[name="factor_type"][value="${state.factorType}"]`);
            if (radio) radio.checked = true;
            updateEditorMode();
            if (state.factorType === "expression") els.factorExpr.value = targetV.definition.expression || "";
            else els.factorCode.value = targetV.definition.code || "";
            state.postprocessSteps = targetV.definition.postprocess || [];
            renderPostprocessSteps();
            triggerProbe();
            els.factorModal.style.display = "none";
            switchTab("tab-ide");
            showToast(`已载入版本快照 v${vNum}`, "success");
          }
        })
      );
    } catch (err) {
      els.modalBody.innerHTML = `<div class="text-red">加载详情失败: ${err.message}</div>`;
    }
  }

  // =========================================================================
  // 视图 4：全流程流水线与实验中心
  // =========================================================================
  async function loadRuns() {
    try {
      const runs = await apiGet("/api/runs");
      state.runs = runs;
      renderRunsTable();
    } catch (err) {
      els.runsTbody.innerHTML = `<tr><td colspan="7" class="text-center text-red">加载实验失败: ${err.message}</td></tr>`;
    }
  }

  function renderRunsTable() {
    if (state.runs.length === 0) {
      els.runsTbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">暂无历史实验产物</td></tr>';
      return;
    }
    els.runsTbody.innerHTML = state.runs
      .map((r) => {
        const s = r.summary || {};
        const repLink = r.report_url
          ? `<a href="${r.report_url}" target="_blank" class="btn btn-xs btn-outline">报告 ↗</a>`
          : "-";
        return `
        <tr>
          <td><span class="font-mono text-xs">${r.run_id}</span></td>
          <td><span class="ci-tag">${r.kind}</span></td>
          <td><span class="status-pill ${r.status === "completed" ? "pill-green" : "pill-blue"}">${r.status}</span></td>
          <td class="${s.annual_return >= 0 ? "text-green" : "text-red"}">${formatPctSigned(s.annual_return)}</td>
          <td>${formatNum(s.sharpe, 2)}</td>
          <td class="text-red">${formatPct(s.max_drawdown)}</td>
          <td>${repLink}</td>
        </tr>
      `;
      })
      .join("");
  }

  async function triggerPipeline() {
    const factorExpr = els.pipeFactorInput.value.trim();
    const runGp = els.pipeRunGp.checked;
    const gpGens = parseInt(els.pipeGpGens.value || 3);
    const gpPop = parseInt(els.pipeGpPop.value || 40);
    const mlModel = els.pipeMlModel.value;

    els.btnRunPipeline.disabled = true;
    els.btnRunPipeline.textContent = "全流程流水线正在执行...";
    showToast("端到端流水线已启动 (数据加载 -> 种子入库 -> GP/ML -> 回测)...", "info");

    try {
      const res = await apiPost("/api/pipeline/run", {
        factor_expr: factorExpr,
        run_gp: runGp,
        gp_generations: gpGens,
        gp_population: gpPop,
        ml_model: mlModel,
      });
      showToast(`流水线执行成功！Run ID: ${res.run_id}`, "success");
      await loadRuns();
      await loadFactors();
      if (res.report_url) {
        window.open(res.report_url, "_blank");
      }
    } catch (err) {
      showToast(`流水线执行失败: ${err.message}`, "error");
    } finally {
      els.btnRunPipeline.disabled = false;
      els.btnRunPipeline.textContent = "启动全流程端到端研发流水线 🚀";
    }
  }

  // =========================================================================
  // 事件绑定与交互逻辑
  // =========================================================================
  function bindEvents() {
    // 选项卡切换
    $$(".nav-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const target = tab.getAttribute("data-tab");
        switchTab(target);
      });
    });

    // 目录搜索与分类
    els.catalogSearch.addEventListener("input", (e) => {
      state.catalogQuery = e.target.value.trim();
      loadCatalog();
    });
    $$("#catalog-pills .pill-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$("#catalog-pills .pill-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.catalogCategory = btn.getAttribute("data-cat");
        loadCatalog();
      });
    });

    // 编辑器模式切换
    $$('input[name="factor_type"]').forEach((r) => {
      r.addEventListener("change", (e) => {
        state.factorType = e.target.value;
        updateEditorMode();
        triggerProbe();
      });
    });

    // 代码输入触发探针
    els.factorExpr.addEventListener("input", triggerProbe);
    els.factorCode.addEventListener("input", triggerProbe);
    els.factorName.addEventListener("input", triggerProbe);

    // 模板选择
    els.selectTemplate.addEventListener("change", (e) => {
      const tplId = e.target.value;
      if (!tplId || !state.context) return;
      const tpl = (state.context.authoring_templates || []).find((t) => t.id === tplId);
      if (tpl) applyTemplate(tpl);
    });

    // 后处理预设按钮
    $$(".btn-preset").forEach((btn) => {
      btn.addEventListener("click", () => {
        const pKey = btn.getAttribute("data-preset");
        if (pKey === "clear") {
          state.postprocessSteps = [];
        } else if (state.context && state.context.postprocess_presets && state.context.postprocess_presets[pKey]) {
          state.postprocessSteps = JSON.parse(JSON.stringify(state.context.postprocess_presets[pKey]));
        }
        renderPostprocessSteps();
        triggerProbe();
      });
    });

    // 追加后处理步骤
    els.selectAddOp.addEventListener("change", (e) => {
      renderOpArgInput(e.target.value);
    });
    els.btnAddStep.addEventListener("click", () => {
      const op = els.selectAddOp.value;
      const step = { op };
      const argInput = $("#op-arg-val");
      if (argInput) {
        const val = argInput.value;
        if (op === "dropNewStock" || op === "decay" || op === "emadecay") {
          step.window = parseInt(val);
        } else if (op === "MAD" || op === "winsor") {
          step.param = parseFloat(val);
        } else if (op === "indNeut") {
          step.group_by = val;
        } else if (op === "riskNeut") {
          step.field = val;
        } else if (op === "riskNeutMulti") {
          step.fields = val.split(",").map((s) => s.trim()).filter(Boolean);
        }
      }
      state.postprocessSteps.push(step);
      renderPostprocessSteps();
      triggerProbe();
    });

    // 动作按钮
    els.btnSaveDraft.addEventListener("click", saveDraft);
    els.btnRunWorkflow.addEventListener("click", runDualLayerWorkflow);

    // 面板切换
    els.btnSwitchPanel.addEventListener("click", async () => {
      const isReal = state.context && state.context.panel_info && state.context.panel_info.source === "real";
      const targetMode = isReal ? "synthetic" : "auto";
      try {
        const newCtx = await apiPost("/api/panel/switch", { mode: targetMode });
        state.context = newCtx;
        const pInfo = newCtx.panel_info || {};
        const isNowReal = pInfo.source === "real";
        els.panelBadge.textContent = `${isNowReal ? "真实研究面板" : "合成演示面板"} (${pInfo.n_days}天 x ${pInfo.n_symbols}品种)`;
        els.panelBadge.className = `status-pill ${isNowReal ? "pill-green" : "pill-blue"}`;
        showToast(`已切换数据源: ${isNowReal ? "真实研究面板" : "合成演示面板"}`, "success");
        triggerProbe();
      } catch (err) {
        showToast(`切换面板失败: ${err.message}`, "error");
      }
    });

    // 因子库与流水线动作
    els.regStatusFilter.addEventListener("change", loadFactors);
    els.regSourceFilter.addEventListener("change", loadFactors);
    els.btnRefreshFactors.addEventListener("click", loadFactors);
    els.btnInitSeeds.addEventListener("click", async () => {
      try {
        await apiPost("/api/factors/init-seeds");
        showToast("4 大经典文献种子因子已成功注册入库！", "success");
        await loadFactors();
      } catch (err) {
        showToast(`种子因子入库失败: ${err.message}`, "error");
      }
    });

    els.pipeRunGp.addEventListener("change", (e) => {
      els.pipeGpOptions.style.display = e.target.checked ? "grid" : "none";
    });
    els.btnRunPipeline.addEventListener("click", triggerPipeline);
    els.btnRefreshRuns.addEventListener("click", loadRuns);
    els.btnCloseModal.addEventListener("click", () => {
      els.factorModal.style.display = "none";
    });
  }

  function switchTab(tabId) {
    $$(".nav-tab").forEach((t) => {
      t.classList.toggle("active", t.getAttribute("data-tab") === tabId);
    });
    $$(".tab-pane").forEach((p) => {
      p.classList.toggle("active", p.id === tabId);
    });
  }

  function updateEditorMode() {
    if (state.factorType === "expression") {
      els.editorExprWrap.style.display = "block";
      els.editorCodeWrap.style.display = "none";
    } else {
      els.editorExprWrap.style.display = "none";
      els.editorCodeWrap.style.display = "block";
    }
  }

  // =========================================================================
  // 数值格式化辅助
  // =========================================================================
  function formatNum(v, d = 2) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    return Number(v).toFixed(d);
  }

  function formatSigned(v, d = 4) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    const n = Number(v);
    return (n >= 0 ? "+" : "") + n.toFixed(d);
  }

  function formatPct(v) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    return (Number(v) * 100).toFixed(2) + "%";
  }

  function formatPctSigned(v) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    const n = Number(v) * 100;
    return (n >= 0 ? "+" : "") + n.toFixed(2) + "%";
  }

  function formatMoney(v) {
    if (v === null || v === undefined || isNaN(v)) return "0.00";
    return Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // 启动
  window.addEventListener("DOMContentLoaded", initApp);
})();
