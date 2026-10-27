"""FactorLibrary Streamlit 因子研发与回测监控工作台 (frontend/streamlit_app.py)。

支持直连本地核心服务 (FactorWorkbenchService)，提供 4 大功能模块：
1. 6 阶段因子研发工作台 (Stage 0–5 目录、双模态编辑、后处理与探针)；
2. 双层回测与归因看板 (Layer 1 v4 统计分布 + Layer 2 期货实盘逐日盯市)；
3. SQLite 因子资产与版本中心 (状态机流转与版本快照)；
4. 全流程流水线与实验中心 (GP 演化挖掘、ML 合成与 runs 历史)。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import streamlit as st  # noqa: E402
from factorlib.core.catalog import RECOMMENDED_POSTPROCESS_PRESETS  # noqa: E402
from factorlib.core.probe import FactorDefinition  # noqa: E402
from factorlib.core.workbench import FactorWorkbenchService  # noqa: E402


@st.cache_resource
def get_service() -> FactorWorkbenchService:
    panel_dir = os.environ.get("PANEL_DIR", "data/derived/futures/v1")
    db_path = os.environ.get("DB_PATH", "data/registry/factors.sqlite")
    runs_dir = os.environ.get("RUNS_DIR", "runs")
    return FactorWorkbenchService(panel_dir=panel_dir, db_path=db_path, runs_dir=runs_dir)


def main() -> None:
    st.set_page_config(page_title="FactorLibrary Quant Studio", layout="wide", page_icon="⚡")
    st.title("⚡ FactorLibrary 期货量化因子研发工作台")

    svc = get_service()
    ctx = svc.get_context()
    p_info = ctx.get("panel_info", {})
    r_sum = ctx.get("registry_summary", {})

    with st.sidebar:
        st.header("环境与状态")
        is_real = p_info.get("source") == "real"
        st.markdown(
            f"**当前面板:** `{'真实研究面板' if is_real else '合成演示面板'}`\n\n"
            f"- 交易日数: `{p_info.get('n_days', 0)}` 天\n"
            f"- 品种数量: `{p_info.get('n_symbols', 0)}` 个\n"
            f"- 覆盖区间: `{p_info.get('start_date', '')}` 至 `{p_info.get('end_date', '')}`\n"
            f"- 因子库总数: `{r_sum.get('total', 0)}` 个"
        )
        if st.button("切换数据源 (真实 / 合成)"):
            target_mode = "synthetic" if is_real else "auto"
            svc.switch_panel_source(mode=target_mode)
            st.rerun()

        st.markdown("---")
        st.caption("FactorLibrary · Rust 计算内核 × 实盘逐日盯市引擎")

    tab_ide, tab_analytics, tab_registry, tab_pipeline = st.tabs(
        [
            "🛠️ 6 阶段因子工作台",
            "📊 双层回测与归因看板",
            "🗄️ SQLite 因子库与版本",
            "🚀 全流程流水线与实验",
        ]
    )

    # =========================================================================
    # Tab 1: 6 阶段因子工作台
    # =========================================================================
    with tab_ide:
        col_left, col_mid, col_right = st.columns([1, 2, 1.2])

        with col_left:
            st.subheader("Stage 0/1: 因子目录")
            cat_filter = st.selectbox(
                "目录过滤", ["all", "variables", "functions", "postprocess", "styles"]
            )
            q = st.text_input("关键词过滤", value="")
            items = svc.catalog_search(query=q, catalog=cat_filter, limit=30)
            st.caption(f"匹配条目: {len(items)}")
            for it in items[:15]:
                with st.expander(f"{it.get('name')} ({it.get('category')})"):
                    st.write(it.get("description", ""))
                    if it.get("formula"):
                        st.code(it.get("formula"), language="text")

        with col_mid:
            st.subheader("Stage 2: 因子编写与声明式后处理")
            templates = ctx.get("authoring_templates", [])
            tpl_titles = ["自定义"] + [t.get("title", "") for t in templates]
            chosen_tpl = st.selectbox("载入经典模板", tpl_titles)

            factor_name = st.text_input("因子名称", value="my_alpha_factor")
            factor_type = st.radio("编写模态", ["expression", "code"], horizontal=True, index=0)

            default_expr = "cs_zscore(ry_ann) + cs_zscore(pct_change(close, 20))"
            default_code = (
                "def compute(context):\n"
                '    c = context.history("close", 20)\n'
                "    return c[-1] / c[0] - 1.0\n"
            )

            if chosen_tpl != "自定义":
                selected_t = next((t for t in templates if t.get("title") == chosen_tpl), None)
                if selected_t:
                    factor_name = selected_t.get("name", factor_name)
                    factor_type = selected_t.get("factor_type", factor_type)
                    default_expr = selected_t.get("expression", default_expr)
                    default_code = selected_t.get("code", default_code)

            if factor_type == "expression":
                expr_code = st.text_area("DSL 向量化表达式", value=default_expr, height=90)
                py_code = ""
            else:
                py_code = st.text_area("Python 因果沙箱代码", value=default_code, height=140)
                expr_code = ""

            st.markdown("**声明式后处理预设 (12 算子链):**")
            pp_preset = st.selectbox(
                "后处理预设链",
                ["standard", "style_neutral", "minimal", "none"],
                index=0,
            )
            pp_steps: list[dict[str, Any]] = []
            if pp_preset in RECOMMENDED_POSTPROCESS_PRESETS:
                pp_steps = RECOMMENDED_POSTPROCESS_PRESETS[pp_preset]

            st.caption(f"当前后处理步骤: {[s.get('op') for s in pp_steps]}")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_save_draft = st.button("💾 保存因子草稿")
            with col_btn2:
                btn_run_wf = st.button("🚀 执行双层回测与版本锁定", type="primary")

        with col_right:
            st.subheader("Stage 3: 静态探针诊断")
            defn_probe = FactorDefinition(
                name=factor_name,
                factor_type=factor_type,
                expression=expr_code,
                code=py_code,
                postprocess=pp_steps,
            )
            probe_rep = svc.probe(defn_probe)
            is_ok = probe_rep.get("status") == "ok"
            if is_ok:
                st.success("探针状态: PASS (零未来函数与语法合规)")
            else:
                st.error("探针状态: FAIL")
                for err in probe_rep.get("errors", []):
                    st.warning(err)

            req = probe_rep.get("execution_requirements", {})
            st.metric("总预热天数 (Total Lookback)", f"{req.get('total_required_lookback', 1)} 日")
            st.metric("Warmup Bars", req.get("warmup_bars", 0))
            st.caption(f"代数指纹: `{probe_rep.get('canon_hash', '-')}`")

            if probe_rep.get("stage_order_warnings"):
                for w in probe_rep["stage_order_warnings"]:
                    st.info(f"⚠️ {w}")

        if btn_save_draft:
            try:
                res = svc.save_draft(
                    {
                        "name": factor_name,
                        "factor_type": factor_type,
                        "expression": expr_code,
                        "code": py_code,
                        "postprocess": pp_steps,
                    }
                )
                st.success(f"草稿已保存 (Factor ID: {res.get('factor_id')})")
            except Exception as e:
                st.error(f"保存草稿失败: {e}")

        if btn_run_wf:
            with st.spinner("正在调用 Rust 内核执行双层回测与版本快照锁定..."):
                try:
                    wf_res = svc.run_workflow(
                        {
                            "name": factor_name,
                            "factor_type": factor_type,
                            "expression": expr_code,
                            "code": py_code,
                            "postprocess": pp_steps,
                        }
                    )
                    st.session_state["last_wf_res"] = wf_res
                    st.success(f"回测成功！版本快照已锁定: v{wf_res.get('version_locked', 1)}")
                except Exception as e:
                    st.error(f"回测失败: {e}")

    # =========================================================================
    # Tab 2: 双层回测与归因看板
    # =========================================================================
    with tab_analytics:
        wf_res = st.session_state.get("last_wf_res")
        if not wf_res:
            st.info("尚未执行回测，请先在「6 阶段因子工作台」提交回测任务。")
        else:
            st.subheader(
                f"回测与归因看板: {wf_res.get('definition', {}).get('name')} (v{wf_res.get('version_locked', 1)})"
            )
            v4 = wf_res.get("v4_stats", {})
            bt = wf_res.get("backtest", {})
            ic_sum = v4.get("ic_analysis", {}).get("summary", {})
            long_sum = v4.get("long_analysis", {}).get("summary", {})
            m = bt.get("metrics", {})

            st.markdown("### Layer 1: Schema v4 因子统计分布评价")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rank IC 均值", f"{ic_sum.get('rank_ic_mean', 0.0):+.4f}")
            c2.metric("Rank ICIR", f"{ic_sum.get('rank_ic_ir', 0.0):.2f}")
            c3.metric("首尾20% PercIC", f"{ic_sum.get('perc_ic_mean', 0.0):+.4f}")
            c4.metric(
                "多头年化超额 (vs 等权池)",
                f"{long_sum.get('annual_excess_return', 0.0) * 100:+.2f}%",
            )

            st.markdown("### Layer 2: 期货实盘逐日盯市 (MTM) 绩效")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("累计收益率", f"{m.get('total_return', 0.0) * 100:+.2f}%")
            b2.metric("夏普比率", f"{m.get('sharpe', 0.0):.2f}")
            b3.metric("最大回撤", f"{m.get('max_drawdown', 0.0) * 100:.2f}%")
            b4.metric("日均换手率", f"{m.get('daily_turnover_mean', 0.0) * 100:.2f}%")

            # 绘制实盘净值曲线
            eq_series = bt.get("series", {}).get("equity", [])
            if eq_series:
                import pandas as pd

                df_eq = pd.DataFrame(
                    {"equity": eq_series},
                    index=pd.to_datetime(bt.get("dates", [])),
                )
                st.line_chart(df_eq["equity"])

            if wf_res.get("report_path"):
                st.success(f"已生成独立 HTML 诊断报告: `{wf_res.get('report_path')}`")

    # =========================================================================
    # Tab 3: SQLite 因子库与版本中心
    # =========================================================================
    with tab_registry:
        st.subheader("SQLite 因子资产全生命周期注册库 (factors.sqlite)")
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            st.caption("因子资产生命周期状态机: candidate -> accepted -> active -> deprecated")
        with col_f2:
            if st.button("+ 初始化经典种子因子"):
                svc.init_seed_factors()
                st.success("4 大经典文献种子因子已注册入库！")
                st.rerun()

        factors = svc.list_factors()
        if not factors:
            st.info("因子库为空，点击右上角初始化种子因子。")
        else:
            for f in factors:
                with st.expander(
                    f"#{f.get('id')} {f.get('name')} | 状态: {f.get('status')} | 来源: {f.get('source')} | 版本数: {f.get('version_count')}"
                ):
                    st.code(f.get("canon_expr", ""), language="text")
                    st.caption(f"代数指纹: `{f.get('canon_hash')}`")
                    detail = svc.get_factor_detail(f.get("id"))
                    if detail and detail.get("versions"):
                        st.markdown("**历史锁定版本快照:**")
                        for v in detail["versions"]:
                            st.text(f"- v{v.get('version')}: {v.get('canon_expr')}")

    # =========================================================================
    # Tab 4: 全流程流水线与实验中心
    # =========================================================================
    with tab_pipeline:
        st.subheader("端到端量化研发流水线 (run_pipeline)")
        pipe_expr = st.text_input("单因子 DSL 表达式 (留空执行多因子机器学习合成)", value="")
        pipe_gp = st.checkbox("同时开启 STGP 遗传规划演化挖掘", value=False)
        pipe_ml = st.selectbox("多因子合成模型", ["equal_weight", "ridge", "lgbm"])

        if st.button("启动端到端全流程流水线 🚀", type="primary"):
            with st.spinner("流水线运行中 (数据加载 -> 种子入库 -> GP/ML -> 回测)..."):
                try:
                    res_pipe = svc.run_e2e_pipeline(
                        {
                            "factor_expr": pipe_expr,
                            "run_gp": pipe_gp,
                            "ml_model": pipe_ml,
                            "gp_generations": 2,
                            "gp_population": 30,
                        }
                    )
                    st.success(f"流水线执行成功！Run ID: {res_pipe.get('run_id')}")
                    st.session_state["last_wf_res"] = res_pipe
                except Exception as e:
                    st.error(f"流水线执行失败: {e}")

        st.markdown("---")
        st.subheader("历史实验记录 (runs/)")
        runs = svc.list_runs()
        if not runs:
            st.caption("暂无历史实验记录。")
        else:
            for r in runs[:10]:
                s = r.get("summary", {})
                ret_val = (
                    f"{s.get('annual_return', 0.0) * 100:.2f}%"
                    if s.get("annual_return") is not None
                    else "-"
                )
                sharpe_val = f"{s.get('sharpe', 0.0):.2f}" if s.get("sharpe") is not None else "-"
                st.text(
                    f"[{r.get('run_id')}] 类型: {r.get('kind')} | 状态: {r.get('status')} | 年化收益: {ret_val} | 夏普: {sharpe_val}"
                )


if __name__ == "__main__":
    main()
