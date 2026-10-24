"""P7 整体验收测试：端到端全链路量化研发与回测流水线 (tests/python/test_e2e_pipeline.py)。

严格遵循白皮书第 16.2 节三层自动化测试验证金字塔顶层规范：
1. 验证从面板加载 -> 种子因子注册 -> GP 批量挖掘 -> ML 多因子合成 -> 风险平价组合优化 -> Rust 期货实盘状态机回测 -> 交互式 HTML 诊断报告的全流程串通；
2. 验证逐日盯市守恒定律：E_d - E_{d-1} == net_pnl_d；
3. 验证命令行 CLI 工具 (scripts/run_pipeline.py) 可行性与产出完备性；
4. 验证 SQLite 实验追踪与因子库无缝联动。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from factorlib.core.backtest import BacktestResult

from scripts.run_pipeline import run_pipeline


@pytest.fixture
def panel_dir() -> Path:
    p = Path("data/derived/futures/v1")
    if not p.exists():
        pytest.skip(f"标准研究面板目录不存在: {p}")
    return p


def test_e2e_pipeline_single_factor_flow(panel_dir: Path, tmp_path: Path) -> None:
    """测试单因子直接贯通流水线 (快速求值、权重构建、Rust 撮合、报告产出)。"""
    out_dir = tmp_path / "single_factor_run"
    db_path = tmp_path / "test_factors.sqlite"

    res = run_pipeline(
        panel_dir=panel_dir,
        db_path=db_path,
        factor_expr="pct_change(close, 20)",
        output_dir=out_dir,
        skip_seed=True,
    )

    assert res["status"] == "success"
    assert out_dir.exists()

    # 1. 验证必要产出文件
    report_html = out_dir / "report.html"
    summary_json = out_dir / "summary.json"
    equity_csv = out_dir / "equity.csv"
    pipeline_summary = out_dir / "pipeline_summary.json"

    assert report_html.exists() and report_html.stat().st_size > 1000
    assert summary_json.exists() and summary_json.stat().st_size > 50
    assert equity_csv.exists() and equity_csv.stat().st_size > 100
    assert pipeline_summary.exists()

    # 2. 验证指标合理性
    with open(summary_json, encoding="utf-8") as f:
        metrics = json.load(f)
    assert "annual_return" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "daily_turnover_mean" in metrics
    assert "total_return" in metrics

    # 3. 验证 HTML 报告内容结构
    html_content = report_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content or "<html>" in html_content
    assert "FactorLibrary" in html_content
    assert "single_factor_run" in html_content

    # 4. 验证逐日盯市守恒定律 E_d - E_{d-1} == net_pnl_d
    bt_result: BacktestResult = res["backtest_result"]
    T = len(bt_result.dates)
    for d in range(1, T):
        diff = abs(bt_result.equity[d] - (bt_result.equity[d - 1] + bt_result.net_pnl[d]))
        assert diff < 1e-6, f"Day {d} 逐日资金恒等式失衡: diff={diff}"


def test_e2e_pipeline_multi_factor_synthesis_flow(panel_dir: Path, tmp_path: Path) -> None:
    """测试多因子资产库联动流水线 (种子因子自动入库、ML 合成或等权兜底、回测)。"""
    out_dir = tmp_path / "multi_factor_run"
    db_path = tmp_path / "multi_factors.sqlite"

    res = run_pipeline(
        panel_dir=panel_dir,
        db_path=db_path,
        factor_expr=None,
        ml_model="equal_weight",
        output_dir=out_dir,
        skip_seed=False,
    )

    assert res["status"] == "success"
    assert (out_dir / "report.html").exists()
    assert (out_dir / "summary.json").exists()

    # 验证 SQLite 数据库中确实沉淀了因子资产
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM factor WHERE status='accepted'")
    accepted_count = cur.fetchone()[0]
    conn.close()
    assert accepted_count >= 1, "因子库中应至少包含自动注册的种子因子"


def test_e2e_pipeline_with_gp_discovery(panel_dir: Path, tmp_path: Path) -> None:
    """测试集成遗传规划 (STGP) 批量挖掘的完整流水线。"""
    out_dir = tmp_path / "gp_pipeline_run"
    db_path = tmp_path / "gp_factors.sqlite"

    res = run_pipeline(
        panel_dir=panel_dir,
        db_path=db_path,
        factor_expr=None,
        run_gp=True,
        gp_generations=1,
        gp_population=10,
        output_dir=out_dir,
        skip_seed=False,
    )

    assert res["status"] == "success"
    summary_file = out_dir / "pipeline_summary.json"
    assert summary_file.exists()
    with open(summary_file, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("run_gp") is True
    assert "gp_admitted_count" in meta


def test_e2e_pipeline_cli_invocation(panel_dir: Path, tmp_path: Path) -> None:
    """测试命令行 CLI 工具 (scripts/run_pipeline.py) 端到端真实子进程执行。"""
    cli_out_dir = tmp_path / "cli_pipeline_run"
    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--panel-dir",
        str(panel_dir),
        "--factor",
        "ts_zscore(close, 20)",
        "--output-dir",
        str(cli_out_dir),
        "--skip-seed",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI 运行失败:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    assert (cli_out_dir / "report.html").exists()
    assert (cli_out_dir / "summary.json").exists()
    assert (cli_out_dir / "equity.csv").exists()
