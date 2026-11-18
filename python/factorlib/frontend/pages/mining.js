// Page 4: 自动化挖掘与 DL 训练中心 (Recipe 组装、后台独立子进程派发、实时进度/日志看板)

import { apiGet, apiPost, showDialog, toast } from "../api.js";
import { escapeHtml, fmtNum } from "../components/charts.js";
import { ms, store } from "../state.js";

function buildRecipePayload(market, st) {
  const kind = st.mining.kind;
  const fsIdEl = document.getElementById("mFsSelect");
  const targetEl = document.getElementById("mLabelSelect");
  const seedEl = document.getElementById("mSeed");
  const featureSetId = fsIdEl && fsIdEl.value ? Number(fsIdEl.value) : null;
  const targetLabel = targetEl ? targetEl.value : "y_5";
  const seed = seedEl ? Number(seedEl.value || 42) : 42;

  const params = {};
  document.querySelectorAll("#recipeForm [data-rparam]").forEach((el) => {
    const k = el.getAttribute("data-rparam");
    const raw = el.value;
    params[k] = el.type === "number" ? Number(raw) : raw;
  });

  return {
    market,
    kind,
    feature_set_id: featureSetId,
    target_label: targetLabel,
    seed,
    params,
  };
}

function renderFoldPreview(market) {
  const el = document.getElementById("folds");
  if (!el) return;
  const years = market === "futures" ? [2020, 2021, 2022, 2023, 2024] : [2021, 2022, 2023, 2024, 2025];
  el.innerHTML =
    years
      .map(
        (y, i) => `
    <div class="flex items-center gap-2">
      <span class="lbl" style="width:54px;">Fold ${i + 1}</span>
      <div class="fold flex-1">
        <span style="width:${30 + i * 8}%;background:#64748b">train</span>
        <span style="width:3%;background:#cbd5e1;color:#334155">P</span>
        <span style="width:10%;background:#d97706">valid</span>
        <span style="width:3%;background:#cbd5e1;color:#334155">P</span>
        <span style="width:12%;background:var(--mk)">test ${y}</span>
        <span style="flex:1;background:transparent"></span>
      </div>
    </div>`
      )
      .join("") +
    `<div class="fold mt-2">
      <span style="width:86%;background:var(--border);color:var(--muted-foreground)">选型期 Selection Period</span>
      <span style="flex:1;background:#0f172a">Lockbox 盲测</span>
    </div>`;
}

export async function renderMiningPage(navigate) {
  const m = store.market;
  const st = ms(m);

  // 加载 FeatureSet 与 Runs
  const [fsets, runs] = await Promise.all([
    apiGet(m, "/feature-sets").catch(() => []),
    apiGet(m, "/runs").catch(() => []),
  ]);
  st.featureSets = fsets;
  st.mining.runs = runs;

  // 子页签切换
  document.querySelectorAll("#minTabs button").forEach((btn) => {
    const k = btn.getAttribute("data-k");
    btn.setAttribute("aria-selected", String(k === st.mining.kind));
    btn.onclick = () => {
      st.mining.kind = k;
      st.mining.lastRecipe = null;
      renderMiningPage(navigate);
    };
  });

  renderRecipeForm(m, st);
  renderFoldPreview(m);
  renderRunsTable(m, st, navigate);
  updateCliPreview(m, st);

  const freezeBtn = document.getElementById("freezeBtn");
  if (freezeBtn) {
    freezeBtn.onclick = async () => {
      try {
        const res = await apiPost(m, "/recipes", buildRecipePayload(m, st));
        st.mining.lastRecipe = res;
        updateCliPreview(m, st);
        toast(`已冻结不可变配方 ${res.recipe_id}`, "ok");
      } catch (err) {
        toast(`冻结配方失败: ${err.message}`, "bad");
      }
    };
  }

  const launchBtn = document.getElementById("launchBtn");
  if (launchBtn) {
    launchBtn.onclick = async () => {
      launchBtn.disabled = true;
      try {
        const payload = buildRecipePayload(m, st);
        if (st.mining.lastRecipe && st.mining.lastRecipe.recipe_id) {
          payload.recipe_id = st.mining.lastRecipe.recipe_id;
        }
        const res = await apiPost(m, "/runs", payload);
        st.mining.lastRecipe = {
          recipe_id: res.recipe_id,
          content_hash: "frozen",
          cli_command: res.cli_command,
        };
        updateCliPreview(m, st);
        toast(`已在后台派发子进程: ${res.run_id} (PID=${res.pid})`, "ok");
        await refreshRuns(m, st, navigate);
      } catch (err) {
        toast(`派发失败: ${err.message}`, "bad");
      } finally {
        launchBtn.disabled = false;
      }
    };
  }

  const copyBtn = document.getElementById("copyCliBtn");
  if (copyBtn) {
    copyBtn.onclick = async () => {
      if (!st.mining.lastRecipe) {
        try {
          st.mining.lastRecipe = await apiPost(m, "/recipes", buildRecipePayload(m, st));
          updateCliPreview(m, st);
        } catch (err) {
          toast(`生成 CLI 命令失败: ${err.message}`, "bad");
          return;
        }
      }
      const cmd = document.getElementById("cliPreview")?.textContent || "";
      navigator.clipboard?.writeText(cmd).catch(() => {});
      showDialog("已生成标准 CLI 命令", cmd);
    };
  }

  const refBtn = document.getElementById("refreshRunsBtn");
  if (refBtn) {
    refBtn.onclick = () => refreshRuns(m, st, navigate);
  }
}

function renderRecipeForm(market, st) {
  const formEl = document.getElementById("recipeForm");
  if (!formEl) return;
  const fsets = st.featureSets || [];
  const fsOpts = fsets
    .map((f) => `<option value="${f.id}">${escapeHtml(f.name)} (v${f.version})</option>`)
    .join("");
  const labels = ((st.context && st.context.panel_info && st.context.panel_info.labels) || ["y_1", "y_5", "y_10"])
    .filter((l) => l.startsWith("y_"))
    .map((l) => `<option value="${escapeHtml(l)}" ${l === "y_5" ? "selected" : ""}>${escapeHtml(l)}</option>`)
    .join("");

  const kind = st.mining.kind;
  const dlHint =
    market === "futures"
      ? "小截面期货 PyTorch 模板：轻量 GRU(hidden=32) / Gated-MLP / TCN · 高 Dropout 0.40 · 品种+板块嵌入防过拟合"
      : "大截面 A 股 PyTorch 模板：时空解耦 GRU + Cross-Attn (O(N·M) 市场原型路由) · 按日打包 Batch";
  const defaultHidden = market === "futures" ? 32 : 64;
  const defaultDrop = market === "futures" ? 0.4 : 0.25;

  const body = {
    dl: `
      <b>DL Alpha 滚动训练配方 (PyTorch 1d)</b>
      <label class="lbl" style="display:block;">FeatureSet
        <select class="inp mt-1" id="mFsSelect">${fsOpts}</select>
      </label>
      <div class="grid grid-2 gap-3">
        <label class="lbl">预测标签 y_h
          <select class="inp mt-1" id="mLabelSelect">${labels}</select>
        </label>
        <label class="lbl">网络骨架 (Arch)
          <select class="inp mt-1" data-rparam="model">
            <option value="gru" ${market === "futures" ? "selected" : ""}>GRU (时序注意力池化)</option>
            <option value="st_attention" ${market === "stocks" ? "selected" : ""}>时空解耦 Attention</option>
            <option value="gated_mlp">Gated-MLP (SwiGLU 基线)</option>
            <option value="tcn">因果扩张 1D-TCN</option>
          </select>
        </label>
        <label class="lbl">回看步长 L (Seq Len)<input class="inp mt-1" type="number" data-rparam="seq_len" value="20" min="4" max="60"></label>
        <label class="lbl">隐层维度 Hidden<input class="inp mt-1" type="number" data-rparam="hidden_dim" value="${defaultHidden}" min="8" max="256"></label>
        <label class="lbl">Dropout 正则<input class="inp mt-1" type="number" step="0.05" data-rparam="dropout" value="${defaultDrop}" min="0" max="0.8"></label>
        <label class="lbl">训练 Epochs<input class="inp mt-1" type="number" data-rparam="epochs" value="12" min="2" max="100"></label>
        <label class="lbl">随机种子数 (Seeds)<input class="inp mt-1" type="number" data-rparam="n_seeds" value="3" min="1" max="5"></label>
        <label class="lbl">折数 (Folds)<input class="inp mt-1" type="number" data-rparam="n_folds" value="4" min="2" max="8"></label>
        <label class="lbl">Purge ≥ h<input class="inp mt-1" type="number" data-rparam="purge_window" value="5" min="1"></label>
        <label class="lbl">换手差分惩罚<input class="inp mt-1" type="number" step="0.01" data-rparam="turnover_penalty" value="0.05" min="0" max="1"></label>
        <label class="lbl">标签去 Beta
          <select class="inp mt-1" data-rparam="label_neutralize">
            <option value="demean" selected>截面去均值 (demean)</option>
            <option value="sector">行业/板块中性 (sector)</option>
            <option value="none">不中性化 (raw)</option>
          </select>
        </label>
        <label class="lbl">Lockbox 盲测占比<input class="inp mt-1" type="number" step="0.05" data-rparam="lockbox_ratio" value="0.2" min="0" max="0.4"></label>
        <label class="lbl">基础 Seed<input class="inp mt-1" type="number" id="mSeed" value="42"></label>
      </div>
      <div class="mk-soft p-2 lbl" style="border-radius:6px;">💡 ${escapeHtml(dlHint)}</div>`,
    gp: `
      <b>STGP 批量演化挖掘配方</b>
      <label class="lbl" style="display:block;">FeatureSet
        <select class="inp mt-1" id="mFsSelect">${fsOpts}</select>
      </label>
      <div class="grid grid-2 gap-3">
        <label class="lbl">预测标签 y_h
          <select class="inp mt-1" id="mLabelSelect">${labels}</select>
        </label>
        <label class="lbl">种群规模<input class="inp mt-1" type="number" data-rparam="population" value="60" min="10"></label>
        <label class="lbl">演化代数<input class="inp mt-1" type="number" data-rparam="generations" value="3" min="1"></label>
        <label class="lbl">岛屿数<input class="inp mt-1" type="number" data-rparam="islands" value="2" min="1"></label>
        <label class="lbl">随机种子<input class="inp mt-1" type="number" id="mSeed" value="42"></label>
      </div>`,
    ml: `
      <b>二层 Walk-Forward 合成与消融配方</b>
      <label class="lbl" style="display:block;">FeatureSet / 池范围
        <select class="inp mt-1" id="mFsSelect">${fsOpts}</select>
      </label>
      <div class="grid grid-2 gap-3">
        <label class="lbl">目标标签
          <select class="inp mt-1" id="mLabelSelect">${labels}</select>
        </label>
        <label class="lbl">训练窗模式
          <select class="inp mt-1" data-rparam="train_window">
            <option value="expanding">expanding</option>
            <option value="rolling">rolling</option>
          </select>
        </label>
        <label class="lbl">Purge ≥ h<input class="inp mt-1" type="number" data-rparam="purge_window" value="5"></label>
        <label class="lbl">基础 Seed<input class="inp mt-1" type="number" id="mSeed" value="42"></label>
      </div>`,
  }[kind];

  formEl.innerHTML = body;
}

function updateCliPreview(market, st) {
  const r = st.mining.lastRecipe;
  const ridEl = document.getElementById("rid");
  const hashEl = document.getElementById("rhash");
  const cliEl = document.getElementById("cliPreview");
  const scriptName = st.mining.kind === "gp" ? "run_gp.py" : st.mining.kind === "ml" ? "run_ml.py" : "run_dl.py";
  if (r && r.recipe_id) {
    if (ridEl) ridEl.textContent = r.recipe_id;
    if (hashEl) hashEl.textContent = `sha256: ${(r.content_hash || "").slice(0, 12)}…`;
    if (cliEl) cliEl.textContent = r.cli_command || `python scripts/${scriptName} --recipe ${r.recipe_id} --market ${market}`;
  } else {
    if (ridEl) ridEl.textContent = "未冻结 (启动时自动生成)";
    if (hashEl) hashEl.textContent = "sha256 内容寻址";
    if (cliEl) cliEl.textContent = `python scripts/${scriptName} --recipe <recipe_id> --market ${market}`;
  }
}

export async function refreshRuns(market, st, navigate) {
  try {
    const runs = await apiGet(market, "/runs");
    st.mining.runs = runs;
    renderRunsTable(market, st, navigate);
    const activeCount = runs.filter((r) => r.status === "running" || r.status === "queued").length;
    const rcEl = document.getElementById("runCount");
    if (rcEl) rcEl.textContent = String(activeCount);
  } catch (err) {
    toast(`刷新任务列表失败: ${err.message}`, "warn");
  }
}

function renderRunsTable(market, st, navigate) {
  const tbody = document.getElementById("runsTbody");
  if (!tbody) return;
  const runs = st.mining.runs || [];
  tbody.innerHTML =
    runs
      .map((r) => {
        const prog = r.progress || {};
        const pct = Math.min(100, Math.max(0, Math.round((Number(prog.pct) || (r.status === "completed" ? 1 : 0)) * 100)));
        const stageText = prog.stage || r.status;
        const icirText = prog.valid_icir !== undefined ? fmtNum(prog.valid_icir, 3) : "-";
        const stCls = r.status === "running" ? "warn" : r.status === "completed" ? "ok" : "bad";
        const isRunning = r.status === "running" || r.status === "queued";
        return `
      <tr>
        <td><code class="code">${escapeHtml(r.run_id)}</code></td>
        <td><span class="chip">${escapeHtml(r.kind)}</span></td>
        <td><code class="code">${escapeHtml(r.recipe_id || "-")}</code></td>
        <td>
          <div class="bar"><i style="width:${pct}%;${r.status === "failed" ? "background:#dc2626" : ""}"></i></div>
          <div class="lbl mt-1">${escapeHtml(stageText)} (${pct}%)</div>
        </td>
        <td>${escapeHtml(icirText)}</td>
        <td>${escapeHtml(r.pid ?? "—")}</td>
        <td class="${stCls}">${escapeHtml(r.status)}</td>
        <td style="white-space:nowrap;">
          <button type="button" class="btn btn-sm" data-rlog="${escapeHtml(r.run_id)}">日志</button>
          ${isRunning ? `<button type="button" class="btn btn-sm" data-rcancel="${escapeHtml(r.run_id)}">中止</button>` : ""}
          ${r.status === "completed" ? `<button type="button" class="btn btn-sm" data-rcand="${escapeHtml(r.run_id)}">候选 →</button>` : ""}
        </td>
      </tr>`;
      })
      .join("") || '<tr><td colspan="8" class="lbl">暂无后台任务记录，可在上方配置配方并启动。</td></tr>';

  tbody.querySelectorAll("[data-rlog]").forEach((btn) => {
    btn.onclick = async () => {
      const rid = btn.getAttribute("data-rlog");
      const drawer = document.getElementById("logDrawer");
      if (!drawer) return;
      if (st.mining.openLogRunId === rid && !drawer.hidden) {
        drawer.hidden = true;
        st.mining.openLogRunId = null;
        return;
      }
      st.mining.openLogRunId = rid;
      drawer.hidden = false;
      drawer.textContent = "加载实时日志中…";
      try {
        const detail = await apiGet(market, `/runs/${encodeURIComponent(rid)}`);
        drawer.textContent = detail.log_tail || detail.summary_md || `[${rid}] 暂无日志输出`;
      } catch (err) {
        drawer.textContent = `加载日志失败: ${err.message}`;
      }
    };
  });

  tbody.querySelectorAll("[data-rcancel]").forEach((btn) => {
    btn.onclick = async () => {
      const rid = btn.getAttribute("data-rcancel");
      try {
        await apiPost(market, `/runs/${encodeURIComponent(rid)}/cancel`, {});
        toast(`已发送中止信号: ${rid}`, "warn");
        await refreshRuns(market, st, navigate);
      } catch (err) {
        toast(`中止失败: ${err.message}`, "bad");
      }
    };
  });

  tbody.querySelectorAll("[data-rcand]").forEach((btn) => {
    btn.onclick = () => {
      const rid = btn.getAttribute("data-rcand");
      st.registry.view = "cand";
      st.registry.runFilter = rid;
      if (navigate) navigate(market, "registry");
    };
  });
}
