// 高密度 SVG 图表渲染组件 (K线、均线/BOLL、多曲线、柱状图、相关性热力图)

export const CHART_COLORS = [
  "var(--mk)",
  "#16a34a",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#ea580c",
  "#65a30d",
  "#4338ca",
  "#b91c1c",
];

export function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function fmtPct(v, digits = 2) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "-";
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

export function fmtNum(v, digits = 4) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "-";
  return Number(v).toFixed(digits);
}

export function drawLines(svg, seriesList, width, height, pad = 34, xLabels = [], options = {}) {
  if (!svg) return;
  const cleanSeries = seriesList
    .map((arr) => (arr || []).map((v) => (v === null || v === undefined || !Number.isFinite(Number(v)) ? null : Number(v))))
    .filter((arr) => arr.some((v) => v !== null));

  if (!cleanSeries.length) {
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle" font-size="12" fill="var(--muted-foreground)">暂无曲线数据</text>`;
    return;
  }

  const allVals = cleanSeries.flat().filter((v) => v !== null);
  let mn = Math.min(...allVals);
  let mx = Math.max(...allVals);
  if (Math.abs(mx - mn) < 1e-12) {
    mn -= 1;
    mx += 1;
  }
  const n = Math.max(...cleanSeries.map((a) => a.length), 2);
  const padL = pad + 8;
  const padR = 10;
  const padT = 12;
  const padB = 22;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const X = (i) => padL + (i * plotW) / Math.max(1, n - 1);
  const Y = (v) => padT + ((mx - v) / (mx - mn)) * plotH;

  let g = "";
  const fmtAxis = options.percent
    ? (v) => `${(v * 100).toFixed(1)}%`
    : (v) => (Math.abs(mx - mn) < 1 ? v.toFixed(3) : v.toFixed(1));

  for (let i = 0; i <= 4; i += 1) {
    const val = mn + ((mx - mn) * i) / 4;
    const y = Y(val);
    g += `<line x1="${padL}" x2="${width - padR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-width="1"/>`;
    g += `<text x="2" y="${(y + 4).toFixed(1)}" font-size="11" fill="var(--muted-foreground)">${fmtAxis(val)}</text>`;
  }

  if (mn < 0 && mx > 0) {
    const y0 = Y(0);
    g += `<line x1="${padL}" x2="${width - padR}" y1="${y0.toFixed(1)}" y2="${y0.toFixed(1)}" stroke="var(--muted-foreground)" stroke-dasharray="3,3" stroke-width="1"/>`;
  }

  const labels = xLabels && xLabels.length ? xLabels : [];
  if (labels.length > 0) {
    const step = Math.max(1, Math.floor(labels.length / 5));
    for (let i = 0; i < labels.length; i += step) {
      g += `<text x="${X(i).toFixed(1)}" y="${height - 4}" font-size="11" fill="var(--muted-foreground)">${escapeHtml(String(labels[i]).slice(0, 10))}</text>`;
    }
  }

  const colors = options.colors || CHART_COLORS;
  cleanSeries.forEach((arr, idx) => {
    let seg = [];
    const col = colors[idx % colors.length];
    arr.forEach((v, i) => {
      if (v === null) {
        if (seg.length > 1) {
          g += `<polyline fill="none" stroke="${col}" stroke-width="1.7" points="${seg.join(" ")}"/>`;
        }
        seg = [];
      } else {
        seg.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
      }
    });
    if (seg.length > 1) {
      g += `<polyline fill="none" stroke="${col}" stroke-width="1.7" points="${seg.join(" ")}"/>`;
    }
  });

  if (options.legend && options.legend.length) {
    g += options.legend
      .map((name, i) => `<text x="${padL + 6 + i * 110}" y="14" font-size="11.5" fill="${colors[i % colors.length]}">● ${escapeHtml(name)}</text>`)
      .join("");
  }

  svg.innerHTML = g;
}

function rollingMean(bars, win) {
  return bars.map((_, i) => {
    if (i < win - 1) return null;
    let sum = 0;
    for (let j = i - win + 1; j <= i; j += 1) sum += bars[j].c;
    return sum / win;
  });
}

export function drawKlineChart(svg, klineData, kState, ohlcLabel) {
  if (!svg || !klineData) return;
  const W = 1300;
  const H = 440;

  if (kState.mode === "basket") {
    const series = (klineData.basket_series || []).map((s) => s.values);
    const names = (klineData.basket_series || []).map((s) => s.symbol);
    drawLines(svg, series, W, H, 42, klineData.dates || [], { legend: names });
    if (ohlcLabel) {
      ohlcLabel.textContent = "组合内各标的起点归一 = 100 · 用于横向对比走势";
    }
    return;
  }

  const bars = klineData.bars || [];
  if (!bars.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" font-size="13" fill="var(--muted-foreground)">暂无 K 线数据</text>`;
    return;
  }

  const n = bars.length;
  const pad = { l: 58, r: 14, t: 14, b: 22 };
  const mainH = 305;
  const subTop = mainH + pad.t + 10;
  const subH = H - subTop - pad.b;

  const hi = Math.max(...bars.map((b) => b.h));
  const lo = Math.min(...bars.map((b) => b.l));
  const span = Math.max(hi - lo, 1e-6);
  const cw = (W - pad.l - pad.r) / Math.max(1, n);
  const X = (i) => pad.l + (i + 0.5) * cw;
  const Y = (v) => pad.t + ((hi - v) / span) * (mainH - pad.t);

  let g = "";
  for (let i = 0; i <= 4; i += 1) {
    const v = lo + (span * i) / 4;
    const y = Y(v);
    g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--border)"/>`;
    g += `<text x="2" y="${(y + 4).toFixed(1)}" font-size="11.5" fill="var(--muted-foreground)">${v > 500 ? v.toFixed(0) : v.toFixed(2)}</text>`;
  }

  bars.forEach((k, i) => {
    const up = k.c >= k.o;
    const col = up ? "#dc2626" : "#16a34a";
    const x = X(i);
    const yTop = Y(Math.max(k.o, k.c));
    const rectH = Math.max(1.2, Math.abs(Y(k.o) - Y(k.c)));
    const bw = Math.max(1.5, cw * 0.68);
    g += `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${Y(k.h).toFixed(1)}" y2="${Y(k.l).toFixed(1)}" stroke="${col}" stroke-width="1"/>`;
    g += `<rect x="${(x - bw / 2).toFixed(1)}" y="${yTop.toFixed(1)}" width="${bw.toFixed(1)}" height="${rectH.toFixed(1)}" fill="${col}"/>`;
  });

  const addOverlay = (arr, color) => {
    const pts = arr
      .map((v, i) => (v === null ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`))
      .filter(Boolean);
    if (pts.length > 1) {
      g += `<polyline fill="none" stroke="${color}" stroke-width="1.4" points="${pts.join(" ")}"/>`;
    }
  };
  if (kState.ind.MA5) addOverlay(rollingMean(bars, 5), "#d97706");
  if (kState.ind.MA20) addOverlay(rollingMean(bars, 20), "#2563eb");
  if (kState.ind.MA60) addOverlay(rollingMean(bars, 60), "#7c3aed");
  if (kState.ind.BOLL) {
    const m20 = rollingMean(bars, 20);
    const sd = bars.map((_, i) => {
      if (i < 19) return null;
      const mean = m20[i];
      let s2 = 0;
      for (let j = i - 19; j <= i; j += 1) s2 += (bars[j].c - mean) ** 2;
      return Math.sqrt(s2 / 20);
    });
    addOverlay(m20.map((v, i) => (v === null ? null : v + 2 * sd[i])), "#94a3b8");
    addOverlay(m20.map((v, i) => (v === null ? null : v - 2 * sd[i])), "#94a3b8");
  }

  // Subchart
  const subKey = kState.sub || "vol";
  const sv = bars.map((b) => (subKey === "vol" ? b.v : subKey === "oi" ? b.oi : b.ret));
  const smx = Math.max(...sv.map((x) => Math.abs(x)), 1e-6);
  const smn = subKey === "ret" ? -smx : 0;
  const SY = (v) => subTop + (1 - (v - smn) / Math.max(1e-9, smx - smn)) * subH;

  if (subKey === "oi") {
    const pts = sv.map((v, i) => `${X(i).toFixed(1)},${SY(v).toFixed(1)}`).join(" ");
    g += `<polyline fill="none" stroke="var(--mk)" stroke-width="1.5" points="${pts}"/>`;
  } else {
    sv.forEach((v, i) => {
      const up = bars[i].c >= bars[i].o;
      const col = up ? "#dc2626" : "#16a34a";
      const y0 = SY(0);
      const yv = SY(v);
      const bw = Math.max(1.5, cw * 0.68);
      g += `<rect x="${(X(i) - bw / 2).toFixed(1)}" y="${Math.min(y0, yv).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, Math.abs(yv - y0)).toFixed(1)}" fill="${col}" opacity="0.65"/>`;
    });
  }

  const subTitle = subKey === "vol" ? "成交量" : subKey === "oi" ? "持仓/换手" : "日收益";
  g += `<text x="2" y="${subTop + 12}" font-size="11.5" fill="var(--muted-foreground)">${subTitle}</text>`;
  g += `<text x="${pad.l}" y="${H - 4}" font-size="11.5" fill="var(--muted-foreground)">${escapeHtml(bars[0].date)}</text>`;
  g += `<text x="${W - 88}" y="${H - 4}" font-size="11.5" fill="var(--muted-foreground)">${escapeHtml(bars[n - 1].date)}</text>`;
  g += `<line id="klCross" y1="14" y2="${H - 22}" stroke="var(--muted-foreground)" stroke-dasharray="3,3" style="display:none"/>`;
  g += `<rect id="klHitArea" x="${pad.l}" y="0" width="${W - pad.l - pad.r}" height="${H}" fill="transparent"/>`;

  svg.innerHTML = g;
  const hit = svg.querySelector("#klHitArea");
  const cross = svg.querySelector("#klCross");
  if (hit && cross && ohlcLabel) {
    ohlcLabel.textContent = `最近 ${bars[n - 1].date} · C ${bars[n - 1].c.toFixed(2)} · 悬停查看 OHLC`;
    hit.onmousemove = (e) => {
      const bb = svg.getBoundingClientRect();
      const x = ((e.clientX - bb.left) / Math.max(1, bb.width)) * W;
      const idx = Math.max(0, Math.min(n - 1, Math.floor((x - pad.l) / cw)));
      const k = bars[idx];
      cross.style.display = "";
      cross.setAttribute("x1", X(idx).toFixed(1));
      cross.setAttribute("x2", X(idx).toFixed(1));
      ohlcLabel.innerHTML = `${escapeHtml(k.date)} · O <b>${k.o.toFixed(2)}</b> H <b>${k.h.toFixed(2)}</b> L <b>${k.l.toFixed(2)}</b> C <b class="${k.c >= k.o ? "pos" : "neg"}">${k.c.toFixed(2)}</b> · 涨跌 <b class="${k.ret >= 0 ? "pos" : "neg"}">${(k.ret * 100).toFixed(2)}%</b>`;
    };
    hit.onmouseleave = () => {
      cross.style.display = "none";
    };
  }
}

export function drawHeatmap(container, names, matrix) {
  if (!container) return;
  if (!names || names.length < 2 || !matrix || !matrix.length) {
    container.innerHTML = `<span class="lbl">请选择 2~6 个因子并执行对比</span>`;
    return;
  }
  const head = `<tr><th></th>${names.map((n) => `<th>${escapeHtml(n.slice(0, 12))}</th>`).join("")}</tr>`;
  const rows = matrix
    .map((row, i) => {
      const cells = row
        .map((val, j) => {
          const v = Number(val || 0);
          const absV = Math.min(1, Math.abs(v));
          const warnBold = i !== j && absV >= 0.6 ? ";font-weight:700;color:#dc2626" : "";
          return `<td style="background:color-mix(in srgb,var(--mk) ${(absV * 55).toFixed(0)}%,transparent);text-align:center${warnBold}">${v.toFixed(2)}</td>`;
        })
        .join("");
      return `<tr><td class="lbl">${escapeHtml(names[i].slice(0, 12))}</td>${cells}</tr>`;
    })
    .join("");
  container.innerHTML = `<table class="t"><thead>${head}</thead><tbody>${rows}</tbody></table><div class="lbl mt-1">红色加粗 = |ρ| ≥ 0.60 去重红线</div>`;
}
