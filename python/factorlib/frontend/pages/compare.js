// Page 3: 回测结果与多因子 A/B 同台对比看板

import { apiGet, apiPost, toast } from "../api.js";
import { CHART_COLORS, drawHeatmap, drawLines, escapeHtml, fmtNum, fmtPct } from "../components/charts.js";
import { ms, store } from "../state.js";

function tableHtml(headers, rows, numColsStart = 1) {
  const ths = headers
    .map((h, i) => `<th class="${i >= numColsStart ? "num" : ""}">${escapeHtml(h)}</th>`)
    .join("");
  const trs = rows
    .map(
      (r) =>
        `<tr>${r
          .map((c, i) => `<td class="${i >= numColsStart ? "num" : ""}">${c}</td>`)
          .join("")}</tr>`
    )
    .join("");
  return `<table class="t"><thead><tr>${ths}</tr></thead><tbody>${trs || `<tr><td colspan="${headers.length}" class="lbl">暂无数据</td></tr>`}</tbody></table>`;
}

function signedSpan(val, digits = 4, isPct = false) {
  if (val === null || val === undefined || !Number.isFinite(Number(val))) return "-";
  const v = Number(val);
  const text = isPct ? fmtPct(v, 2) : v.toFixed(digits);
  return `<span class="${v >= 0 ? "pos" : "neg"}">${text}</span>`;
}

export async function renderComparePage() {
  const m = store.market;
  const st = ms(m);

  const factors = await apiGet(m, "/factors").catch(() => []);
  st.registry.factors = factors;

  const cmp = st.compare;
  if (!cmp.selectedIds.length && factors.length) {
    cmp.selectedIds = cmp.mode === "ab" ? factors.slice(0, Math.min(3, factors.length)).map((f) => f.id) : [factors[0].id];
  } else {
    // 过滤掉已不存在的因子 ID
    const validSet = new Set(factors.map((f) => f.id));
    cmp.selectedIds = cmp.selectedIds.filter((id) => validSet.has(id));
    if (!cmp.selectedIds.length && factors.length) {
      cmp.selectedIds = [factors[0].id];
    }
  }

  // 模式切换
  document.querySelectorAll("#cmpModeSeg button").forEach((btn) => {
    const mode = btn.getAttribute("data-cm");
    btn.setAttribute("aria-pressed", String(mode === cmp.mode));
    btn.onclick = async () => {
      cmp.mode = mode;
      if (mode === "single" && cmp.selectedIds.length > 1) {
        cmp.selectedIds = [cmp.selectedIds[0]];
      } else if (mode === "ab" && cmp.selectedIds.length < 2 && factors.length >= 2) {
        cmp.selectedIds = factors.slice(0, Math.min(3, factors.length)).map((f) => f.id);
      }
      await renderComparePage();
    };
  });

  // 因子选择 chips
  const chipsEl = document.getElementById("cmpChips");
  if (chipsEl) {
    chipsEl.innerHTML =
      factors
        .slice(0, 18)
        .map((f) => {
          const on = cmp.selectedIds.includes(f.id);
          const name = f.name || f.canon_expr.slice(0, 16);
          return `<button type="button" class="chip ${on ? "active" : ""}" data-cfid="${f.id}">${escapeHtml(name)} <span class="lbl">${escapeHtml(f.source)}</span></button>`;
        })
        .join("") || '<span class="lbl">当前市场因子库为空，可在「② 单因子实验室」或「⑤ 审核与注册库」初始化种子因子</span>';

    chipsEl.querySelectorAll("[data-cfid]").forEach((btn) => {
      btn.onclick = async () => {
        const fid = Number(btn.getAttribute("data-cfid"));
        if (cmp.mode === "single") {
          cmp.selectedIds = [fid];
          cmp.workflowResult = null;
        } else {
          if (cmp.selectedIds.includes(fid)) {
            if (cmp.selectedIds.length > 1) cmp.selectedIds = cmp.selectedIds.filter((x) => x !== fid);
          } else if (cmp.selectedIds.length < 6) {
            cmp.selectedIds = [...cmp.selectedIds, fid];
          } else {
            toast("多因子 A/B 对比最多选择 6 个因子", "warn");
          }
        }
        await renderComparePage();
      };
    });
  }

  const runBtn = document.getElementById("cmpRunBtn");
  if (runBtn) {
    runBtn.onclick = () => executeCompareOrSingle(m, st, factors, true);
  }

  const abBox = document.getElementById("abBox");
  if (abBox) abBox.hidden = cmp.mode !== "ab";

  if (cmp.mode === "ab") {
    if (cmp.selectedIds.length >= 2 && !cmp.abResult) {
      await executeCompareOrSingle(m, st, factors, false);
    }
    renderABView(m, st);
  } else {
    if (!cmp.workflowResult && cmp.selectedIds.length) {
      await executeCompareOrSingle(m, st, factors, false);
    }
    renderSingleView(m, st);
  }
}

async function executeCompareOrSingle(market, st, factors, forceRun) {
  const cmp = st.compare;
  const runBtn = document.getElementById("cmpRunBtn");
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.textContent = "计算中…";
  }
  try {
    if (cmp.mode === "ab") {
      if (cmp.selectedIds.length < 2) {
        if (forceRun) toast("请先勾选 2~6 个因子进行 A/B 对比", "warn");
        return;
      }
      cmp.abResult = await apiPost(market, "/compare", {
        factor_ids: cmp.selectedIds,
        horizon: 5,
      });
      renderABView(market, st);
    } else {
      const target = factors.find((f) => f.id === cmp.selectedIds[0]);
      if (!target && !cmp.workflowResult) return;
      if (target) {
        const detail = await apiGet(market, `/factors/${target.id}`);
        const defn = detail.definition || {
          name: detail.name || `factor_${detail.id}`,
          factor_type: detail.factor_type || "expression",
          expression: detail.canon_expr,
          materialization_path: detail.materialization_path,
          recipe_id: detail.recipe_id,
          postprocess: (st.context && st.context.postprocess_presets && st.context.postprocess_presets.standard) || "standard",
        };
        cmp.workflowResult = await apiPost(market, "/workflow/run", {
          ...defn,
          materialize: false,
          lock_version: false,
          n_quantiles: 5,
        });
      }
      renderSingleView(market, st);
    }
  } catch (err) {
    toast(`评估计算失败: ${err.message}`, "bad");
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = "执行评估 / 对比";
    }
  }
}

function renderSingleView(market, st) {
  const wf = st.compare.workflowResult;
  const pInfo = (st.context && st.context.panel_info) || {};
  const reportLink = document.getElementById("openReportLink");
  if (reportLink) {
    if (wf && wf.report_url) {
      reportLink.href = wf.report_url;
      reportLink.hidden = false;
    } else {
      reportLink.hidden = true;
    }
  }

  if (!wf) {
    document.getElementById("runTime").textContent = "";
    document.getElementById("btMeta").innerHTML = '<div class="card p-3 lbl" style="grid-column:span 4;">请选择因子或从「② 单因子实验室」发起双层评估。</div>';
    document.getElementById("btKpi").innerHTML = "";
    ["tAll", "tGrp", "tLS", "tYear", "tMonth", "mkDiag", "grpLeg", "icSum", "lsSum", "exSum"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = "";
    });
    ["icChart", "grpChart", "lsChart", "exChart"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = "";
    });
    return;
  }

  const v4 = wf.v4_stats || {};
  const icSum = (v4.ic_analysis && v4.ic_analysis.summary) || {};
  const longSum = (v4.long_analysis && v4.long_analysis.summary) || {};
  const bt = wf.backtest || {};
  const btMet = bt.metrics || {};
  const cost = bt.cost_breakdown || {};
  const dates = v4.dates || bt.dates || [];

  document.getElementById("runTime").textContent = `${dates[0] || "-"} ~ ${dates[dates.length - 1] || "-"} · run ${wf.run_id || "-"}`;

  const ppStr = ((wf.definition && wf.definition.postprocess) || []).map((s) => s.op || s).join(" → ") || "默认后处理";
  const metaCards = [
    ["回测区间", `${dates[0] || "-"} ~ ${dates[dates.length - 1] || "-"}`],
    ["市场与样本", `1d · ${market === "futures" ? "期货" : "A 股"} · N=${pInfo.n_symbols || "-"}`],
    ["因子名称", `${(wf.definition && wf.definition.name) || "-"} (sign=${wf.sign > 0 ? "+1" : "-1"})`],
    ["后处理链", ppStr],
  ];
  document.getElementById("btMeta").innerHTML = metaCards
    .map(([k, v]) => `<div class="card p-3"><span class="lbl">${escapeHtml(k)}</span><div style="font-weight:600;margin-top:4px;">${escapeHtml(v)}</div></div>`)
    .join("");

  const kpis = [
    ["IC 均值", fmtNum(icSum.ic_mean, 4)],
    ["ICIR", fmtNum(icSum.ic_ir, 4)],
    ["RankIC", fmtNum(icSum.rank_ic_mean, 4)],
    ["RankICIR", fmtNum(icSum.rank_ic_ir, 4)],
    ["多头超额年化", fmtPct(longSum.annual_excess_return, 2)],
    ["实盘夏普", fmtNum(btMet.sharpe, 3)],
    ["最大回撤", fmtPct(btMet.max_drawdown, 2)],
    ["日均换手率", fmtPct(btMet.daily_turnover_mean ?? longSum.daily_turnover, 2)],
    ["实盘年化收益", fmtPct(btMet.annual_return, 2)],
    ["IC > 0 胜率", fmtPct(icSum.win_rate, 1)],
    ["Calmar", fmtNum(btMet.calmar, 2)],
    ["有效交易日", `${dates.length} 天`],
  ];
  document.getElementById("btKpi").innerHTML = kpis
    .map(([k, v]) => `<div class="kcell"><span class="lbl">${escapeHtml(k)}</span><b>${escapeHtml(v)}</b></div>`)
    .join("");

  // 全市场指标
  document.getElementById("tAll").innerHTML = tableHtml(
    ["指标", "数值", "IR / 夏普"],
    [
      ["1d RankIC", signedSpan(icSum.rank_ic_mean, 4), fmtNum(icSum.rank_ic_ir, 3)],
      ["首尾20% PercIC", signedSpan(icSum.perc_ic_mean, 4), fmtNum(icSum.perc_ic_ir, 3)],
      ["5d Pearson IC", signedSpan(icSum.ic_5d_mean, 4), fmtNum(icSum.ic_5d_ir, 3)],
      ["5d RankIC", signedSpan(icSum.rank_ic_5d_mean, 4), fmtNum(icSum.rank_ic_5d_ir, 3)],
      ["多头超额 (vs 基准)", signedSpan(longSum.annual_excess_return, 2, true), fmtNum(longSum.excess_sharpe, 3)],
    ]
  );

  // 分层表现
  const gSum = (v4.group_analysis && v4.group_analysis.summary) || {};
  const gKeys = Object.keys(gSum);
  document.getElementById("tGrp").innerHTML = tableHtml(
    ["分组", "累计收益", "年化收益", "年化波动", "夏普"],
    gKeys.map((g) => [
      g,
      signedSpan(gSum[g].total_return, 2, true),
      signedSpan(gSum[g].annual_return, 2, true),
      fmtPct(gSum[g].annual_volatility, 2),
      fmtNum(gSum[g].sharpe, 2),
    ])
  );

  // 风格暴露归因
  const styleObj = v4.style_exposure || {};
  const styleNames = Array.isArray(styleObj.styles) ? styleObj.styles : [];
  const longExp = styleObj.long_exposure || {};
  const shortExp = styleObj.short_exposure || {};
  const lsExp = styleObj.long_short_exposure || {};
  const rankCorr = styleObj.rank_corr || {};
  document.getElementById("tLS").innerHTML = tableHtml(
    ["风险风格", "多头组暴露", "空头组暴露", "多空净暴露 (L-S)", "截面秩相关"],
    styleNames.map((s) => [
      escapeHtml(s),
      signedSpan(longExp[s], 3),
      signedSpan(shortExp[s], 3),
      signedSpan(lsExp[s], 3),
      signedSpan(rankCorr[s], 3),
    ])
  );

  // 分年表现
  const icAnn = (v4.ic_analysis && v4.ic_analysis.annual) || {};
  const longAnn = (v4.long_analysis && v4.long_analysis.annual) || {};
  const years = Object.keys(icAnn).sort();
  document.getElementById("tYear").innerHTML = tableHtml(
    ["年份", "IC", "RankIC", "RankICIR", "超额年化", "超额夏普", "加法回撤", "日换手"],
    years.map((y) => {
      const ia = icAnn[y] || {};
      const la = longAnn[y] || {};
      return [
        y,
        signedSpan(ia.ic_mean, 4),
        signedSpan(ia.rank_ic_mean, 4),
        fmtNum(ia.rank_ic_ir, 3),
        signedSpan(la.annual_excess_return, 2, true),
        fmtNum(la.excess_sharpe, 2),
        signedSpan(-(la.additive_max_drawdown || 0), 2, true),
        fmtPct(la.daily_turnover, 2),
      ];
    })
  );

  // 分月表现
  const mRows = bt.monthly_returns || [];
  const flatMonths = [];
  mRows.forEach((yr) => {
    Object.entries(yr.months || {}).forEach(([mo, val]) => {
      if (Math.abs(val) > 1e-10) flatMonths.push([`${yr.year}-${mo}`, signedSpan(val, 2, true)]);
    });
  });
  document.getElementById("tMonth").innerHTML = tableHtml(
    ["月份", "组合月收益"],
    flatMonths.slice(-12)
  );

  // 曲线渲染
  const ricSeries = (v4.ic_analysis && v4.ic_analysis.series && v4.ic_analysis.series.rank_ic) || [];
  let acc = 0;
  const cumRic = ricSeries.map((v) => {
    if (v !== null && Number.isFinite(v)) acc += v;
    return acc;
  });
  document.getElementById("icSum").textContent = `均值 ${fmtNum(icSum.rank_ic_mean, 4)} · 累计 ${fmtNum(acc, 2)}`;
  drawLines(document.getElementById("icChart"), [cumRic], 520, 190, 34, dates);

  const navObj = (v4.group_analysis && v4.group_analysis.nav_series) || {};
  const grpSeries = gKeys.map((g) => navObj[g] || []);
  drawLines(document.getElementById("grpChart"), grpSeries, 520, 190, 34, dates);
  const legEl = document.getElementById("grpLeg");
  if (legEl) {
    legEl.innerHTML = gKeys
      .map((g, i) => `<span><i style="display:inline-block;width:10px;height:3px;background:${CHART_COLORS[i % CHART_COLORS.length]};vertical-align:middle"></i> ${g}</span>`)
      .join("");
  }

  const eq = (bt.series && bt.series.equity) || [];
  const initCash = eq[0] || 1;
  const eqNorm = eq.map((v) => v / initCash);
  document.getElementById("lsSum").textContent = `总收益 ${fmtPct(btMet.total_return, 2)} · 回撤 ${fmtPct(btMet.max_drawdown, 2)}`;
  drawLines(document.getElementById("lsChart"), [eqNorm], 1000, 240, 38, bt.dates || dates, { colors: ["#dc2626"] });

  const exCum = (v4.long_analysis && v4.long_analysis.cum_excess_compound) || [];
  document.getElementById("exSum").textContent = `相对 ${market === "futures" ? "等权基准" : "市场基准"} 年化超额 ${fmtPct(longSum.annual_excess_return, 2)}`;
  drawLines(document.getElementById("exChart"), [exCum], 1000, 240, 38, dates, { colors: ["#0f766e"], percent: true });

  // 市场专属诊断卡片
  document.getElementById("mkDiagTitle").textContent =
    market === "futures" ? "期货专属：成本损耗 · 换月费 · 保证金" : "A 股专属：基准超额 · 交易摩擦 · 回撤统计";
  const marginRatio = (bt.series && bt.series.margin_ratio) || [0];
  const peakMargin = Math.max(...marginRatio, 0);
  const diagCells =
    market === "futures"
      ? [
          ["毛盈亏", `${((cost.gross_pnl || 0) / 1e4).toFixed(1)}万`],
          ["手续费", `${((cost.fees || 0) / 1e4).toFixed(1)}万`],
          ["滑点损耗", `${((cost.slippage || 0) / 1e4).toFixed(1)}万`],
          ["换月双边费", `${((cost.roll_cost || 0) / 1e4).toFixed(1)}万`],
          ["费用拖拽比", fmtPct(cost.cost_drag, 2)],
          ["峰值保证金率", fmtPct(peakMargin, 1)],
        ]
      : [
          ["超额年化", fmtPct(longSum.annual_excess_return, 2)],
          ["超额夏普", fmtNum(longSum.excess_sharpe, 3)],
          ["加法回撤", fmtPct(-(longSum.additive_max_drawdown || 0), 2)],
          ["手续费/印花税", `${((cost.fees || 0) / 1e4).toFixed(1)}万`],
          ["滑点损耗", `${((cost.slippage || 0) / 1e4).toFixed(1)}万`],
          ["费用拖拽比", fmtPct(cost.cost_drag, 2)],
        ];
  document.getElementById("mkDiag").innerHTML = diagCells
    .map(([k, v]) => `<div class="kpi"><span class="lbl">${escapeHtml(k)}</span><b>${escapeHtml(v)}</b></div>`)
    .join("");
}

function renderABView(market, st) {
  const ab = st.compare.abResult;
  if (!ab) return;

  const win = ab.intersection_window || {};
  const facs = ab.factors || [];
  const names = facs.map((f) => f.name);
  document.getElementById("runTime").textContent = `严格交集 OOS 窗口: ${win.start_date || "-"} ~ ${win.end_date || "-"} (${win.n_days || 0} 天)`;

  document.getElementById("btMeta").innerHTML = [
    ["对比模式", `多因子 A/B (${facs.length} 个因子)`],
    ["严格交集区间", `${win.start_date || "-"} ~ ${win.end_date || "-"}`],
    ["市场域", market === "futures" ? "期货 Futures" : "A 股 Stocks"],
    ["基准残差目标", `${(ab.orthogonal_residual && ab.orthogonal_residual.target_name) || "-"}`],
  ]
    .map(([k, v]) => `<div class="card p-3"><span class="lbl">${escapeHtml(k)}</span><div style="font-weight:600;margin-top:4px;">${escapeHtml(v)}</div></div>`)
    .join("");

  // 清理单因子视图专属区域并填入 A/B 概要指标
  const orth0 = ab.orthogonal_residual || {};
  const lead = facs[0] || {};
  document.getElementById("btKpi").innerHTML = [
    ["交集天数", `${win.n_days || 0} 天`],
    ["领衔因子 ICIR", fmtNum(lead.ic_ir, 3)],
    ["正交残差 ICIR", fmtNum(orth0.residual_icir, 3)],
    ["正交残差 t 值", fmtNum(orth0.residual_t_stat, 2)],
  ]
    .map(([k, v]) => `<div class="kcell"><span class="lbl">${escapeHtml(k)}</span><b>${escapeHtml(v)}</b></div>`)
    .join("");
  ["tGrp", "tLS", "tYear", "tMonth", "mkDiag", "grpLeg", "lsSum", "exSum"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = "";
  });
  ["grpChart", "lsChart", "exChart"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = "";
  });

  // 并排多因子对比表放到 tAll 中
  document.getElementById("tAll").innerHTML = tableHtml(
    ["因子", "来源", "IC", "ICIR", "多空年化", "夏普", "换手"],
    facs.map((f) => [
      `<b>${escapeHtml(f.name)}</b>`,
      `<span class="chip">${escapeHtml(f.source)}</span>`,
      signedSpan(f.ic_mean, 4),
      fmtNum(f.ic_ir, 3),
      signedSpan(f.ls_annual_ret, 2, true),
      fmtNum(f.ls_sharpe, 2),
      fmtPct(f.turnover, 1),
    ])
  );

  // 累计 RankIC 多因子同图曲线
  const cumSeries = (ab.cum_ic_series || []).map((s) => s.values);
  drawLines(document.getElementById("icChart"), cumSeries, 520, 190, 34, win.dates || [], {
    legend: names,
  });
  document.getElementById("icSum").textContent = "同交集窗口累计 RankIC";

  // 相关性热力图
  document.getElementById("abWindowBadge").textContent = `${win.n_days || 0} 个交集交易日`;
  drawHeatmap(document.getElementById("heat"), names, ab.corr_matrix || []);

  // 正交残差累积 IC
  const orth = ab.orthogonal_residual || {};
  document.getElementById("resIcSummary").textContent = `Resid ICIR=${fmtNum(orth.residual_icir, 3)} · t=${fmtNum(orth.residual_t_stat, 2)}`;
  drawLines(
    document.getElementById("resChart"),
    [orth.cum_raw_ic || [], orth.cum_residual_ic || []],
    460,
    170,
    30,
    orth.dates || [],
    { legend: ["原始 RankIC", "正交残差 IC"], colors: ["var(--mk)", "#16a34a"] }
  );
}
