// Page 5: 因子资产与审核晋升中心 (candidate 队列、台账、批量三关准入审计、晋升/驳回)

import { apiDelete, apiGet, apiPost, apiPatch, showDialog, toast } from "../api.js";
import { drawLines, escapeHtml, fmtNum, fmtPct } from "../components/charts.js";
import { ms, store } from "../state.js";

function gateBadge(val) {
  if (val === undefined || val === null) return '<span class="lbl">—</span>';
  return val ? '<span class="ok">✔</span>' : '<span class="bad">✘</span>';
}

export async function renderRegistryPage(navigate) {
  const m = store.market;
  const st = ms(m);
  const reg = st.registry;

  const fSrcEl = document.getElementById("fSrc");
  const fRunEl = document.getElementById("fRun");
  const fIcEl = document.getElementById("fIc");
  if (fSrcEl) fSrcEl.value = reg.sourceFilter;
  if (fRunEl) fRunEl.value = reg.runFilter;
  if (fIcEl) fIcEl.value = String(reg.minIcir);

  document.querySelectorAll("#regViewSeg button").forEach((btn) => {
    const rv = btn.getAttribute("data-rv");
    btn.setAttribute("aria-pressed", String(rv === reg.view));
    btn.onclick = () => {
      reg.view = rv;
      renderRegistryPage(navigate);
    };
  });

  if (fSrcEl) fSrcEl.onchange = () => { reg.sourceFilter = fSrcEl.value; renderRegistryPage(navigate); };
  if (fRunEl) fRunEl.oninput = () => { reg.runFilter = fRunEl.value.trim(); renderRegistryTable(m, st, navigate); };
  if (fIcEl) fIcEl.oninput = () => { reg.minIcir = Number(fIcEl.value || 0); renderRegistryTable(m, st, navigate); };

  const initSeedsBtn = document.getElementById("initSeedsBtn");
  if (initSeedsBtn) {
    initSeedsBtn.onclick = async () => {
      initSeedsBtn.disabled = true;
      try {
        const res = await apiPost(m, "/factors/init-seeds", {});
        toast(`已初始化 ${res.registered_ids.length} 个经典种子因子`, "ok");
        st.context = await apiGet(m, "/context");
        await renderRegistryPage(navigate);
      } catch (err) {
        toast(`初始化失败: ${err.message}`, "bad");
      } finally {
        initSeedsBtn.disabled = false;
      }
    };
  }

  const auditBtn = document.getElementById("auditBtn");
  if (auditBtn) {
    auditBtn.onclick = async () => {
      const ids = [...reg.selectedFactorIds];
      if (!ids.length) {
        showDialog("请先勾选因子", "勾选至少 1 个候选因子后点击「批量三关准入审计 ▶」。");
        return;
      }
      auditBtn.disabled = true;
      auditBtn.textContent = "审计中…";
      try {
        const res = await apiPost(m, "/factors/audit", { factor_ids: ids, horizon: 5 });
        (res.results || []).forEach((r) => reg.auditMap.set(r.factor_id, r));
        const passed = (res.results || []).filter((r) => r.all_passed).length;
        toast(`三关准入审计完成：${passed}/${ids.length} 个因子通过全部三关`, passed ? "ok" : "warn");
        renderRegistryTable(m, st, navigate);
        if (reg.detailFactor) await openFactorDetail(m, st, reg.detailFactor.id, navigate);
      } catch (err) {
        toast(`批量审计失败: ${err.message}`, "bad");
      } finally {
        auditBtn.disabled = false;
        auditBtn.textContent = "批量三关准入审计 ▶";
      }
    };
  }

  const batchPromBtn = document.getElementById("batchPromoteBtn");
  if (batchPromBtn) {
    batchPromBtn.onclick = async () => {
      const ids = [...reg.selectedFactorIds];
      if (!ids.length) return toast("请先勾选要晋升的因子", "warn");
      try {
        await apiPost(m, "/factors/batch-status", { factor_ids: ids, status: "active" });
        st.context = await apiGet(m, "/context");
        toast(`已批量晋升 ${ids.length} 个因子为 active`, "ok");
        reg.selectedFactorIds.clear();
        await renderRegistryPage(navigate);
      } catch (err) {
        toast(`批量晋升失败: ${err.message}`, "bad");
      }
    };
  }

  const batchCmpBtn = document.getElementById("batchCompareBtn");
  if (batchCmpBtn) {
    batchCmpBtn.onclick = () => {
      const ids = [...reg.selectedFactorIds];
      if (ids.length < 2 || ids.length > 6) {
        return toast("请在列表中勾选 2~6 个因子再进入 A/B 对比", "warn");
      }
      st.compare.mode = "ab";
      st.compare.selectedIds = ids;
      st.compare.abResult = null;
      if (navigate) navigate(m, "compare");
    };
  }

  const factors = await apiGet(m, "/factors", {
    source: reg.sourceFilter,
  }).catch(() => []);
  reg.factors = factors;

  renderRegistryTable(m, st, navigate);
}

function renderRegistryTable(market, st, navigate) {
  const reg = st.registry;
  const tbody = document.getElementById("regRows");
  if (!tbody) return;

  const rows = (reg.factors || []).filter((f) => {
    const isCand = f.status === "candidate";
    if (reg.view === "cand" && !isCand) return false;
    if (reg.view === "ledger" && isCand) return false;
    if (reg.runFilter && !(f.source_run || "").toLowerCase().includes(reg.runFilter.toLowerCase())) return false;
    const pm = f.primary_metric || {};
    if (Math.abs(Number(pm.ic_ir || 0)) < reg.minIcir) return false;
    return true;
  });

  tbody.innerHTML =
    rows
      .map((f) => {
        const pm = f.primary_metric || {};
        const aud = reg.auditMap.get(f.id) || f.audit_summary || {};
        const icir = aud.oos_icir !== undefined ? aud.oos_icir : pm.ic_ir;
        const turn = aud.turnover !== undefined ? aud.turnover : pm.turnover;
        const maxCorr = aud.max_corr;
        const checked = reg.selectedFactorIds.has(f.id);
        const name = f.name || f.canon_expr.slice(0, 26);
        const selCls = reg.detailFactor && reg.detailFactor.id === f.id ? "selected" : "";
        return `
      <tr class="${selCls}" style="cursor:pointer;" data-fid="${f.id}">
        <td><input type="checkbox" class="rowSel" data-cb-fid="${f.id}" ${checked ? "checked" : ""} aria-label="选择 ${escapeHtml(name)}"></td>
        <td><b>${escapeHtml(name)}</b><div class="lbl">${escapeHtml(f.source_run || "")}</div></td>
        <td><span class="chip">${escapeHtml(f.source)}</span></td>
        <td>${escapeHtml(f.factor_type || "expression")}</td>
        <td class="num">${fmtNum(icir, 2)}</td>
        <td class="num">${fmtPct(turn, 1)}</td>
        <td class="num ${maxCorr >= 0.6 ? "bad" : ""}">${maxCorr !== undefined ? fmtNum(maxCorr, 2) : "—"}</td>
        <td>${gateBadge(aud.gate1_pass)}</td>
        <td>${gateBadge(aud.gate2_pass)}</td>
        <td>${gateBadge(aud.gate3_pass)}</td>
        <td><span class="chip">${escapeHtml(f.status)}</span></td>
      </tr>`;
      })
      .join("") || `<tr><td colspan="11" class="lbl">暂无匹配因子记录</td></tr>`;

  const selAll = document.getElementById("selAll");
  if (selAll) {
    selAll.checked = rows.length > 0 && rows.every((r) => reg.selectedFactorIds.has(r.id));
    selAll.onchange = () => {
      rows.forEach((r) => {
        if (selAll.checked) reg.selectedFactorIds.add(r.id);
        else reg.selectedFactorIds.delete(r.id);
      });
      renderRegistryTable(market, st, navigate);
    };
  }

  tbody.querySelectorAll("input[data-cb-fid]").forEach((cb) => {
    cb.onclick = (e) => e.stopPropagation();
    cb.onchange = () => {
      const fid = Number(cb.getAttribute("data-cb-fid"));
      if (cb.checked) reg.selectedFactorIds.add(fid);
      else reg.selectedFactorIds.delete(fid);
    };
  });

  tbody.querySelectorAll("tr[data-fid]").forEach((tr) => {
    tr.onclick = () => openFactorDetail(market, st, Number(tr.getAttribute("data-fid")), navigate);
  });
}

async function openFactorDetail(market, st, factorId, navigate) {
  const panel = document.getElementById("regDetail");
  if (!panel) return;
  try {
    const detail = await apiGet(market, `/factors/${factorId}`);
    st.registry.detailFactor = detail;
    const aud = st.registry.auditMap.get(factorId) || null;
    const name = detail.name || `factor_${detail.id}`;
    const vers = detail.versions || [];
    const latestVer = vers.length ? vers[vers.length - 1].version : 1;

    panel.innerHTML = `
      <div class="flex justify-between items-center">
        <b>${escapeHtml(name)}</b>
        <span class="chip">${escapeHtml(detail.status)}</span>
      </div>
      <div class="lbl code" style="word-break:break-all;">${escapeHtml(detail.canon_expr)}</div>
      <svg id="dSpark" viewBox="0 0 320 90" class="w-full"></svg>
      <div class="lbl space-y-1">
        <div>source / type：<b>${escapeHtml(detail.source)}</b> / ${escapeHtml(detail.factor_type)}</div>
        <div>source_run：<code>${escapeHtml(detail.source_run || "-")}</code></div>
        <div>recipe_id：<code>${escapeHtml(detail.recipe_id || "-")}</code></div>
        <div>factor_version：v${latestVer} (共 ${vers.length} 版)</div>
        <div>.npz 物化：<code>${escapeHtml(detail.materialization_path || "未物化")}</code></div>
      </div>
      ${
        aud
          ? `<div class="card p-2 space-y-1 lbl">
              <b>三关准入审计摘要</b>
              <div>关1 (|ICIR|=${fmtNum(aud.oos_icir, 3)}, max|ρ|=${fmtNum(aud.max_corr, 2)}): ${gateBadge(aud.gate1_pass)}</div>
              <div>关2 (残差 ICIR=${fmtNum(aud.residual_icir, 3)}, t=${fmtNum(aud.residual_t_stat, 2)}): ${gateBadge(aud.gate2_pass)}</div>
              <div>关3 (消融 ΔSharpe=${fmtNum(aud.delta_sharpe, 3)}): ${gateBadge(aud.gate3_pass)}</div>
            </div>`
          : ""
      }
      <div class="flex gap-2">
        <button type="button" class="btn btn-p flex-1" id="dPromActive">晋升 active</button>
        <button type="button" class="btn flex-1" id="dPromAcc">accepted</button>
        <button type="button" class="btn" id="dDeprecate">驳回</button>
      </div>
      <div class="flex gap-2">
        <button type="button" class="btn flex-1" id="dGoEval">去诊断对比 →</button>
        <button type="button" class="btn" id="dDel">删除</button>
      </div>
    `;

    // 用各 split/horizon 指标绘制简单趋势
    const msList = (detail.metrics || []).map((x) => Number(x.ic_ir || 0));
    drawLines(document.getElementById("dSpark"), [msList.length >= 2 ? msList : [0, msList[0] || 0]], 320, 90, 22);

    const setStatus = async (status) => {
      await apiPatch(market, `/factors/${factorId}/status`, { status });
      st.context = await apiGet(market, "/context");
      toast(`${name} 已更新为 ${status}`, "ok");
      await renderRegistryPage(navigate);
      await openFactorDetail(market, st, factorId, navigate);
    };
    document.getElementById("dPromActive").onclick = () => setStatus("active");
    document.getElementById("dPromAcc").onclick = () => setStatus("accepted");
    document.getElementById("dDeprecate").onclick = () => setStatus("deprecated");
    document.getElementById("dGoEval").onclick = () => {
      st.compare.mode = "single";
      st.compare.selectedIds = [factorId];
      st.compare.workflowResult = null;
      if (navigate) navigate(market, "compare");
    };
    document.getElementById("dDel").onclick = async () => {
      await apiDelete(market, `/factors/${factorId}`);
      st.registry.detailFactor = null;
      st.context = await apiGet(market, "/context");
      toast(`已删除因子 ${name}`, "warn");
      panel.innerHTML = '<span class="lbl">因子已删除</span>';
      await renderRegistryPage(navigate);
    };
  } catch (err) {
    panel.innerHTML = `<div class="bad">${escapeHtml(err.message)}</div>`;
  }
}
