from __future__ import annotations
import os
import requests
import streamlit as st

API_URL = os.environ.get("PYRUST_BT_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="pyrust-bt", layout="wide")
st.title("pyrust-bt 回测监控与可视化")

with st.sidebar:
    st.header("新建回测")
    csv_path = st.text_input("CSV 路径", value="examples/data/sample.csv")
    window = st.number_input("SMA 窗口", min_value=1, value=5)
    size = st.number_input("下单手数", min_value=0.0, value=1.0)
    commission_rate = st.number_input("手续费率", min_value=0.0, value=0.0005, format="%f")
    slippage_bps = st.number_input("滑点(bps)", min_value=0.0, value=2.0, format="%f")
    if st.button("提交回测"):
        resp = requests.post(
            f"{API_URL}/runs",
            json={
                "csv_path": csv_path,
                "commission_rate": commission_rate,
                "slippage_bps": slippage_bps,
                "strategy": "sma",
                "window": window,
                "size": size,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            st.success(f"创建成功：{resp.json()['run_id']}")
        else:
            st.error(resp.text)

st.header("任务列表")
resp = requests.get(f"{API_URL}/runs", timeout=60)
if resp.status_code != 200:
    st.error("无法获取任务列表")
else:
    runs = resp.json()
    for r in runs:
        rid = r["run_id"]
        st.subheader(f"Run {rid}")
        st.write(r)
        if r.get("result"):
            res = r["result"]
            eq = res.get("equity_curve", [])
            stats = res.get("stats", {})
            st.write("Stats:", stats)
            if eq:
                import pandas as pd

                df = pd.DataFrame(eq)
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.set_index("datetime")
                st.line_chart(df["equity"]) 