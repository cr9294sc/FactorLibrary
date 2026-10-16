from __future__ import annotations
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Ensure local package import
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pyrust_bt.api import BacktestConfig, BacktestEngine
from pyrust_bt.data import load_csv_to_bars
from pyrust_bt.strategy import Strategy


class SMAStrategy(Strategy):
    def __init__(self, window: int = 5, size: float = 1.0) -> None:
        self.window = window
        self.size = size
        self._closes: List[float] = []

    def next(self, bar: Dict[str, Any]) -> Dict[str, Any] | None:
        close = float(bar["close"])  # type: ignore[assignment]
        self._closes.append(close)
        if len(self._closes) < self.window:
            return None
        sma = sum(self._closes[-self.window :]) / self.window
        if close > sma:
            return {"action": "BUY", "type": "market", "size": self.size}
        elif close < sma:
            return {"action": "SELL", "type": "market", "size": self.size}
        return None


class RunRequest(BaseModel):
    csv_path: str = Field(..., description="CSV 文件路径，含列 datetime,open,high,low,close,volume")
    start: str = "2020-01-01"
    end: str = "2020-12-31"
    cash: float = 10000.0
    commission_rate: float = 0.0005
    slippage_bps: float = 2.0
    strategy: str = Field("sma", description="策略标识，目前仅支持 'sma'")
    window: int = 5
    size: float = 1.0


class RunStatus(BaseModel):
    run_id: str
    status: str  # pending|running|done|error
    message: Optional[str] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None


app = FastAPI(title="pyrust-bt API", version="0.1.0")

RUNS: Dict[str, RunStatus] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _run_task(run_id: str, req: RunRequest) -> None:
    st = RUNS[run_id]
    st.status = "running"
    st.progress = 0.0
    try:
        cfg = BacktestConfig(
            start=req.start,
            end=req.end,
            cash=req.cash,
            commission_rate=req.commission_rate,
            slippage_bps=req.slippage_bps,
        )
        engine = BacktestEngine(cfg)
        bars = load_csv_to_bars(req.csv_path)
        if not bars:
            raise RuntimeError("CSV 数据为空或无法读取")
        # 粗略进度（开始与结束），引擎内部不回调逐步进度
        st.progress = 0.1
        if req.strategy == "sma":
            strat = SMAStrategy(window=req.window, size=req.size)
        else:
            raise RuntimeError(f"未知策略: {req.strategy}")
        res = engine.run(strat, bars)
        st.result = res  # type: ignore[assignment]
        st.progress = 1.0
        st.status = "done"
    except Exception as e:  # noqa: BLE001
        st.status = "error"
        st.message = str(e)
        st.progress = 1.0


@app.post("/runs", response_model=RunStatus)
def create_run(req: RunRequest) -> RunStatus:
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=400, detail="csv_path 不存在")
    run_id = uuid.uuid4().hex
    status = RunStatus(run_id=run_id, status="pending", progress=0.0)
    RUNS[run_id] = status
    EXECUTOR.submit(_run_task, run_id, req)
    return status


@app.get("/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str) -> RunStatus:
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id 不存在")
    return st


@app.get("/runs", response_model=List[RunStatus])
def list_runs() -> List[RunStatus]:
    return list(RUNS.values()) 