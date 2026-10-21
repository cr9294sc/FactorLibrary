"""期货实盘回测交互式 HTML 诊断报告生成器 (core/report.py)。

按白皮书 12 节与 P6 规划设计:
- 生成完全独立、无外部 CDN 依赖的纯 HTML5 / SVG 交互诊断报告;
- 包含顶层核心 KPI 卡片、净值走势与水下回撤曲线 (SVG 双轴图)、月度收益热力表、
  交易成本拆解瀑布图、保证金与杠杆率曲线、极端盈亏交易日明细等。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from factorlib.core.backtest import BacktestResult


def summary_dict(result: BacktestResult) -> dict[str, Any]:
    """生成回测结果的 JSON 序列化评估汇总字典。"""
    m = result.metrics
    return {
        "start_date": str(result.dates[0]) if len(result.dates) > 0 else "",
        "end_date": str(result.dates[-1]) if len(result.dates) > 0 else "",
        "n_trading_days": len(result.dates),
        "initial_capital": float(result.equity[0]) if len(result.equity) > 0 else 0.0,
        "final_equity": float(result.equity[-1]) if len(result.equity) > 0 else 0.0,
        "total_return": float(m.get("total_return", 0.0)),
        "annual_return": float(m.get("annual_return", 0.0)),
        "annual_volatility": float(m.get("annual_volatility", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "sortino": float(m.get("sortino", 0.0)),
        "calmar": float(m.get("calmar", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "max_drawdown_days": int(m.get("max_drawdown_days", 0)),
        "win_rate": float(m.get("win_rate", 0.0)),
        "profit_loss_ratio": float(m.get("profit_loss_ratio", 0.0)),
        "daily_turnover_mean": float(m.get("daily_turnover_mean", 0.0)),
        "annual_turnover": float(m.get("annual_turnover", 0.0)),
        "total_fees": float(m.get("total_fees", 0.0)),
        "total_slippage": float(m.get("total_slippage", 0.0)),
        "total_roll_cost": float(m.get("total_roll_cost", 0.0)),
        "max_margin_utilization": float(m.get("max_margin_utilization", 0.0)),
        "max_leverage": float(m.get("max_leverage", 0.0)),
    }


def _build_monthly_table(dates: np.ndarray, returns: np.ndarray) -> str:
    """构建月度收益热力表 HTML。"""
    if len(dates) == 0:
        return "<p>无数据</p>"

    # 按照年份和月份统计累计收益
    monthly_map: dict[int, dict[int, float]] = {}
    for d_str, r in zip(dates, returns, strict=False):
        d_s = str(d_str)
        try:
            year = int(d_s[:4])
            month = int(d_s[5:7])
        except Exception:
            continue
        if year not in monthly_map:
            monthly_map[year] = dict.fromkeys(range(1, 13), 0.0)
        monthly_map[year][month] = (1.0 + monthly_map[year][month]) * (1.0 + r) - 1.0

    years = sorted(monthly_map.keys())
    if not years:
        return "<p>无有效日期记录</p>"

    html = [
        '<table class="heatmap-table">',
        "<thead><tr><th>年份</th>"
        + "".join([f"<th>{m}月</th>" for m in range(1, 13)])
        + "<th>全年合计</th></tr></thead>",
        "<tbody>",
    ]

    for y in years:
        row = [f"<tr><td class='year-cell'><b>{y}</b></td>"]
        year_cum = 1.0
        for m in range(1, 13):
            val = monthly_map[y][m]
            year_cum *= 1.0 + val
            if val > 0:
                bg = f"rgba(40, 167, 69, {min(0.85, max(0.15, val * 8))})"
                text_cls = "pos"
            elif val < 0:
                bg = f"rgba(220, 53, 69, {min(0.85, max(0.15, abs(val) * 8))})"
                text_cls = "neg"
            else:
                bg = "#f8f9fa"
                text_cls = "zero"
            row.append(
                f'<td class="{text_cls}" style="background-color: {bg}">{val * 100:.2f}%</td>'
            )

        year_ret = year_cum - 1.0
        tot_bg = (
            f"rgba(40, 167, 69, {min(0.9, max(0.2, year_ret * 4))})"
            if year_ret > 0
            else f"rgba(220, 53, 69, {min(0.9, max(0.2, abs(year_ret) * 4))})"
        )
        row.append(
            f'<td class="year-total" style="background-color: {tot_bg}; font-weight:bold;">'
            f"{year_ret * 100:.2f}%</td></tr>"
        )
        html.append("".join(row))

    html.append("</tbody></table>")
    return "\n".join(html)


def _build_svg_equity_chart(equity: np.ndarray, dates: np.ndarray) -> str:
    """生成净值与水下回撤曲线的双层联动 SVG。"""
    T = len(equity)
    if T <= 1:
        return "<p>数据点不足以绘制图表</p>"

    width = 960
    height_eq = 280
    height_dd = 140
    padding_left = 60
    padding_right = 30
    plot_w = width - padding_left - padding_right

    # 归一化净值
    norm_eq = equity / equity[0]
    min_eq = float(np.min(norm_eq))
    max_eq = float(np.max(norm_eq))
    eq_range = max(max_eq - min_eq, 1e-4)

    # 计算水下回撤
    peaks = np.maximum.accumulate(equity)
    dds = (equity - peaks) / peaks
    min_dd = float(np.min(dds))
    dd_range = max(abs(min_dd), 1e-4)

    # 构造净值折线 SVG 坐标点
    pts_eq = []
    for i, v in enumerate(norm_eq):
        x = padding_left + (i / (T - 1)) * plot_w
        y = 20 + (1.0 - (v - min_eq) / eq_range) * (height_eq - 40)
        pts_eq.append(f"{x:.1f},{y:.1f}")
    poly_eq = " ".join(pts_eq)

    # 构造回撤区域 SVG 坐标点
    pts_dd = [f"{padding_left:.1f},{height_eq + 20:.1f}"]
    for i, v in enumerate(dds):
        x = padding_left + (i / (T - 1)) * plot_w
        y = height_eq + 20 + (abs(v) / dd_range) * (height_dd - 35)
        pts_dd.append(f"{x:.1f},{y:.1f}")
    pts_dd.append(f"{padding_left + plot_w:.1f},{height_eq + 20:.1f}")
    poly_dd = " ".join(pts_dd)

    total_h = height_eq + height_dd + 40
    d_start = str(dates[0])
    d_end = str(dates[-1])

    svg = f"""
    <svg viewBox="0 0 {width} {total_h}" class="chart-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#0066cc" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#0066cc" stop-opacity="0.0"/>
        </linearGradient>
        <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#dc3545" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="#dc3545" stop-opacity="0.1"/>
        </linearGradient>
      </defs>

      <!-- 净值图背景网格 -->
      <rect x="{padding_left}" y="10" width="{plot_w}" height="{height_eq - 20}" fill="#fdfdfd" stroke="#e9ecef" stroke-width="1"/>
      <line x1="{padding_left}" y1="{height_eq / 2}" x2="{padding_left + plot_w}" y2="{height_eq / 2}" stroke="#f1f3f5" stroke-dasharray="4"/>
      <text x="15" y="25" fill="#495057" font-size="11" font-weight="bold">净值 {max_eq:.2f}</text>
      <text x="15" y="{height_eq - 15}" fill="#495057" font-size="11" font-weight="bold">净值 {min_eq:.2f}</text>

      <!-- 净值折线 -->
      <polyline fill="none" stroke="#0066cc" stroke-width="2" points="{poly_eq}" stroke-linejoin="round" stroke-linecap="round"/>

      <!-- 回撤图背景网格 -->
      <rect x="{padding_left}" y="{height_eq + 20}" width="{plot_w}" height="{height_dd}" fill="#fff9f9" stroke="#f5c2c7" stroke-width="1"/>
      <text x="15" y="{height_eq + 35}" fill="#842029" font-size="11" font-weight="bold">回撤 0%</text>
      <text x="15" y="{height_eq + height_dd + 15}" fill="#842029" font-size="11" font-weight="bold">最大 {min_dd * 100:.1f}%</text>

      <!-- 回撤区域 -->
      <polygon fill="url(#ddGrad)" stroke="#dc3545" stroke-width="1.5" points="{poly_dd}"/>

      <!-- 时间刻度 -->
      <text x="{padding_left}" y="{total_h - 10}" fill="#6c757d" font-size="11">{d_start}</text>
      <text x="{padding_left + plot_w - 70}" y="{total_h - 10}" fill="#6c757d" font-size="11">{d_end}</text>
    </svg>
    """
    return svg


def _build_cost_waterfall_html(result: BacktestResult) -> str:
    """构建交易摩擦成本拆解瀑布图与明细 HTML。"""
    gross_pnl = float(np.sum(result.gross_pnl))
    fees = float(np.sum(result.fee))
    slippage = float(np.sum(result.slippage))
    roll_cost = float(np.sum(result.roll_cost))
    net_pnl = float(np.sum(result.net_pnl))

    tot_cost = fees + slippage
    cost_drag = (tot_cost / gross_pnl * 100.0) if gross_pnl > 0 else 0.0

    return f"""
    <div class="waterfall-container">
      <table class="waterfall-table">
        <thead>
          <tr>
            <th>核算项目</th>
            <th>金额 (元)</th>
            <th>占毛收益比</th>
            <th>经济学语义</th>
          </tr>
        </thead>
        <tbody>
          <tr class="wf-gross">
            <td><b>盯市毛收益 (Gross PnL)</b></td>
            <td><b>¥{gross_pnl:,.2f}</b></td>
            <td>100.0%</td>
            <td>无摩擦理论交易产生的总差价与基差盈亏</td>
          </tr>
          <tr class="wf-cost">
            <td>- 交易所手续费 (Commissions)</td>
            <td>-¥{fees:,.2f}</td>
            <td>{fees / max(gross_pnl, 1.0) * 100:.2f}%</td>
            <td>开仓与平仓双向按比例或按手扣除</td>
          </tr>
          <tr class="wf-cost">
            <td>- 冲击滑点成本 (Slippage)</td>
            <td>-¥{slippage:,.2f}</td>
            <td>{slippage / max(gross_pnl, 1.0) * 100:.2f}%</td>
            <td>依据品种 tick_size 与设定跳数测算的撮合冲击</td>
          </tr>
          <tr class="wf-roll">
            <td>* 其中：换月展期摩擦 (Roll Costs)</td>
            <td>¥{roll_cost:,.2f}</td>
            <td>{roll_cost / max(gross_pnl, 1.0) * 100:.2f}%</td>
            <td>主力换月平旧开新独立发生的手续费与滑点</td>
          </tr>
          <tr class="wf-net">
            <td><b>实际净盈亏 (Net PnL)</b></td>
            <td><b>¥{net_pnl:,.2f}</b></td>
            <td>{net_pnl / max(gross_pnl, 1.0) * 100:.2f}%</td>
            <td>扣除全部显性与隐性摩擦后的投资者实际到手权益变动</td>
          </tr>
        </tbody>
      </table>
      <div class="friction-card">
        <p><b>摩擦磨损损耗率 (Cost Drag)</b>: <span class="badge {"badge-danger" if cost_drag > 30 else "badge-success"}">{cost_drag:.2f}%</span></p>
        <p class="desc">综合手续费与滑点占毛收益的耗散比重。低于 20% 表明因子具备极佳的容量与抗摩擦抗衰减能力。</p>
      </div>
    </div>
    """


def generate_html_report(
    result: BacktestResult,
    output_path: str | Path,
    title: str = "FactorLibrary 期货实盘回测诊断报告",
) -> Path:
    """生成完全独立的交互式 HTML 诊断报告。

    Parameters
    ----------
    result : BacktestResult
        回测结果数据。
    output_path : str | Path
        输出 HTML 文件的路径。
    title : str, optional
        报告标题。

    Returns
    -------
    Path
        生成的报告文件绝对路径。
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    m = result.metrics
    eq_chart = _build_svg_equity_chart(result.equity, result.dates)
    monthly_table = _build_monthly_table(result.dates, result.returns)
    waterfall_html = _build_cost_waterfall_html(result)

    # 寻找最佳与最差交易日
    rets = result.returns
    sorted_idx = np.argsort(rets)
    worst_days = sorted_idx[:5]
    best_days = sorted_idx[-5:][::-1]

    days_rows = []
    for rank, (b_i, w_i) in enumerate(zip(best_days, worst_days, strict=True), start=1):
        b_d = str(result.dates[b_i])
        b_r = rets[b_i] * 100
        w_d = str(result.dates[w_i])
        w_r = rets[w_i] * 100
        days_rows.append(
            f"<tr><td>#{rank}</td>"
            f"<td class='pos'>{b_d} (+{b_r:.2f}%)</td>"
            f"<td class='neg'>{w_d} ({w_r:.2f}%)</td></tr>"
        )
    extreme_days_html = "\n".join(days_rows)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 24px 36px;
      background: #f8f9fa;
      color: #212529;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: #ffffff;
      padding: 32px 40px;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }}
    h1 {{
      margin-top: 0;
      font-size: 26px;
      color: #1a202c;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 12px;
    }}
    .sub-meta {{
      font-size: 13px;
      color: #718096;
      margin-top: -6px;
      margin-bottom: 24px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 30px;
    }}
    .kpi-card {{
      background: #f7fafc;
      border: 1px solid #edf2f7;
      padding: 16px;
      border-radius: 8px;
    }}
    .kpi-label {{
      font-size: 12px;
      color: #718096;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .kpi-value {{
      font-size: 22px;
      font-weight: 700;
      margin-top: 4px;
      color: #2d3748;
    }}
    .kpi-value.green {{ color: #2e7d32; }}
    .kpi-value.red {{ color: #c62828; }}
    .kpi-sub {{
      font-size: 11px;
      color: #a0aec0;
      margin-top: 2px;
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 600;
      margin: 28px 0 14px 0;
      color: #2d3748;
    }}
    .chart-container {{
      width: 100%;
      background: #fff;
      border: 1px solid #edf2f7;
      border-radius: 8px;
      padding: 16px;
      box-sizing: border-box;
      margin-bottom: 28px;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .heatmap-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      text-align: center;
      margin-bottom: 24px;
    }}
    .heatmap-table th, .heatmap-table td {{
      padding: 8px 4px;
      border: 1px solid #e2e8f0;
    }}
    .heatmap-table th {{
      background: #f1f5f9;
      color: #475569;
    }}
    .heatmap-table td.pos {{ color: #155724; font-weight: 500; }}
    .heatmap-table td.neg {{ color: #721c24; font-weight: 500; }}
    .waterfall-container {{
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .waterfall-table {{
      flex: 2;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .waterfall-table th, .waterfall-table td {{
      padding: 10px 14px;
      border-bottom: 1px solid #edf2f7;
      text-align: left;
    }}
    .waterfall-table th {{
      background: #f8fafc;
      color: #64748b;
    }}
    .wf-gross {{ background: #eff6ff; }}
    .wf-net {{ background: #f0fdf4; font-size: 14px; }}
    .wf-roll td:first-child {{ padding-left: 28px; color: #64748b; }}
    .friction-card {{
      flex: 1;
      min-width: 250px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      padding: 20px;
      border-radius: 8px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 13px;
    }}
    .badge-success {{ background: #d1e7dd; color: #0f5132; }}
    .badge-danger {{ background: #f8d7da; color: #842029; }}
    .desc {{ font-size: 12px; color: #6c757d; margin-top: 8px; line-height: 1.5; }}
    .extreme-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .extreme-table th, .extreme-table td {{
      padding: 8px 12px;
      border-bottom: 1px solid #edf2f7;
      text-align: left;
    }}
    .extreme-table td.pos {{ color: #2e7d32; font-weight: 600; }}
    .extreme-table td.neg {{ color: #c62828; font-weight: 600; }}
    footer {{
      margin-top: 40px;
      padding-top: 16px;
      border-top: 1px solid #e2e8f0;
      font-size: 12px;
      color: #94a3b8;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div class="sub-meta">
      回测区间: {result.dates[0]} 至 {result.dates[-1]} | 交易日数: {len(result.dates)} | 标的品种数: {len(result.symbols)}
    </div>

    <!-- 顶层关键 KPI 卡片 -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">累计净收益率</div>
        <div class="kpi-value {"green" if m.get("total_return", 0.0) >= 0 else "red"}">{m.get("total_return", 0.0) * 100:+.2f}%</div>
        <div class="kpi-sub">年化: {m.get("annual_return", 0.0) * 100:.2f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">夏普比率 (Sharpe)</div>
        <div class="kpi-value {"green" if m.get("sharpe", 0.0) >= 1.5 else ""}">{m.get("sharpe", 0.0):.2f}</div>
        <div class="kpi-sub">年化波动: {m.get("annual_volatility", 0.0) * 100:.2f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">最大回撤 (Max DD)</div>
        <div class="kpi-value red">{m.get("max_drawdown", 0.0) * 100:.2f}%</div>
        <div class="kpi-sub">卡尔玛: {m.get("calmar", 0.0):.2f} | 持续: {m.get("max_drawdown_days", 0)} 天</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">胜率与盈亏比</div>
        <div class="kpi-value">{m.get("win_rate", 0.0) * 100:.1f}%</div>
        <div class="kpi-sub">盈亏比: {m.get("profit_loss_ratio", 0.0):.2f} | 索提诺: {m.get("sortino", 0.0):.2f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">年化换手率</div>
        <div class="kpi-value">{m.get("annual_turnover", 0.0):.1f}x</div>
        <div class="kpi-sub">日均换手: {m.get("daily_turnover_mean", 0.0) * 100:.2f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">最大保证金占用</div>
        <div class="kpi-value">{m.get("max_margin_utilization", 0.0) * 100:.1f}%</div>
        <div class="kpi-sub">峰值杠杆: {m.get("max_leverage", 0.0):.2f}x</div>
      </div>
    </div>

    <!-- 净值走势与水下回撤曲线 -->
    <div class="section-title">组合累计净值与水下回撤曲线 (Cumulative Equity & Underwater Drawdown)</div>
    <div class="chart-container">
      {eq_chart}
    </div>

    <!-- 月度收益热力表 -->
    <div class="section-title">月度收益率热力表现 (Monthly Returns Distribution)</div>
    {monthly_table}

    <!-- 交易摩擦成本瀑布图 -->
    <div class="section-title">交易摩擦损耗瀑布分解 (Transaction Cost Breakdown)</div>
    {waterfall_html}

    <!-- 极端交易日 -->
    <div class="section-title">单日极大极端盈亏日明细 (Extreme Daily Returns)</div>
    <table class="extreme-table">
      <thead>
        <tr>
          <th>排名</th>
          <th>Top 5 最大盈利日</th>
          <th>Top 5 最大亏损日</th>
        </tr>
      </thead>
      <tbody>
        {extreme_days_html}
      </tbody>
    </table>

    <footer>
      FactorLibrary 期货量化因子挖掘与回测平台 · Rust 极速计算内核与实盘逐日盯市引擎支持
    </footer>
  </div>
</body>
</html>
"""
    out_file.write_text(html_content, encoding="utf-8")

    # 同时输出配套 JSON 诊断文件
    json_path = out_file.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_dict(result), fh, indent=2, ensure_ascii=False)

    return out_file
