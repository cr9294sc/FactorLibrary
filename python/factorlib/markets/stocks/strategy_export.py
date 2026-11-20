"""`factorsim` 兼容策略文件一键导出器 (markets/stocks/strategy_export.py)。

按照 Phase 3 P1-2 契约，把平台声明式 YAML 参数**无歧义**地渲染为
`factorsim.pcmaster.StrategyBase` 子类实现（`strategy.py`）与作业配置文件（`config.yaml`），
从而做到「同一份 YAML 既能驱动平台原生 Rust 回测，又能同步导出/更新 `local_backtest/strategy.py`」。

与 `local_backtest/strategy.py` 旧版的差异（本次导出的策略严格补齐了缺失的数学实现）：

1. **`adaptive_stress_guard`**：以 Barra `SIZE` 暴露漂移（`Δ` 截面均值序列）10 日累计冲击
   与 252 日滚动 Z-Score 双阈值检测微盘流动性危机；触发防守体制时同时
   - 上调低波惩罚 `vol_weight × vol_multiplier`；
   - 压降头部幂次 `power × power_scale`；
   - 注入大市值正向倾斜 `+ size_tilt × rank_score(SIZE)`。
   （平台原生 `DynamicRiskGuard` 直接消费 `CNE5FactorRet['SIZE']`；导出策略以等价可得的
   暴露漂移代理实现，二者在微盘危机期同向触发。）
2. **`asymmetric_trade_penalty: {buy, sell}`**：非对称买卖换手惩罚
   `- buy_pen·Σ(w−w_prev)_+ - sell_pen·Σ(w_prev−w)_+`，实现「快速止损、谨慎建仓」。
3. **`reserve_cash` 归一化**：`w_prev = reserve_cash · last_hold / Σ last_hold`，
   彻底修复 0.9 现金预留导致的换手虚耗。
4. **`buy_hurdle` 追高门槛**：`- hurdle_penalty · (jump_+ · (w−w_prev)_+)`，
   `jump = max(alpha_t − ema_alpha, 0)`。
5. **双期 Horizon 规划**：`+ discount·(forecastᵀ w2 − trade_penalty·‖w2−w‖₁)`。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_STRATEGY_PARAMS: dict[str, Any] = {
    "h_max": 0.00625,
    "tvr": 0.5,
    "mode": "horizon",
    "trade_penalty": 0.85,
    "persistence": 0.35,
    "discount": 0.8,
    "memory": 0.75,
    "vol_weight": 0.12,
    "ind_neutral_weight": 0.42,
    "power": 1.8,
    "adaptive_stress_guard": True,
    "stress_lookback": 10,
    "stress_z_window": 252,
    "stress_cumret_threshold": -0.03,
    "stress_z_threshold": 1.5,
    "vol_multiplier": 1.8,
    "power_scale": 0.85,
    "size_tilt": 0.15,
    "buy_penalty": 1.00,
    "sell_penalty": 0.45,
    "buy_hurdle": True,
    "hurdle_penalty": 0.45,
    "reserve_cash": 0.9,
}

DEFAULT_BACKTEST_PARAMS: dict[str, Any] = {
    "cash": 20_000_000.0,
    "fee_rate": 0.003,
    "start_ds": 20200702,
    "end_ds": 20250702,
    "verbose": True,
    "universe": "base",
    "signal_path": "./runtime_assets/combo/signal/signal_data/shifted_signal.parquet",
    "output_path": "./results",
}


class StrategyExportError(ValueError):
    """策略导出参数非法。"""


@dataclass(frozen=True)
class FactorsimStrategyParams:
    """导出到 `factorsim` 策略的完整参数集。"""

    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_STRATEGY_PARAMS))

    def __post_init__(self) -> None:
        merged = dict(DEFAULT_STRATEGY_PARAMS)
        merged.update({k: v for k, v in self.params.items() if v is not None})
        validate_strategy_params(merged)
        object.__setattr__(self, "params", merged)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.params)


def validate_strategy_params(params: Mapping[str, Any]) -> None:
    """强校验导出参数区间，杜绝导出后静默失效的配置。"""
    p = dict(params)
    if not (0.0 < float(p["h_max"]) <= 1.0):
        raise StrategyExportError(f"h_max 必须位于 (0, 1]，实际 {p['h_max']}")
    if not (0.0 < float(p["tvr"]) <= 10.0):
        raise StrategyExportError(f"tvr 必须位于 (0, 10]，实际 {p['tvr']}")
    if str(p["mode"]) not in ("single", "horizon"):
        raise StrategyExportError(f"mode 必须为 'single' 或 'horizon'，实际 {p['mode']!r}")
    for key in ("trade_penalty", "vol_weight", "ind_neutral_weight", "hurdle_penalty", "size_tilt"):
        if float(p[key]) < 0.0:
            raise StrategyExportError(f"{key} 不得为负，实际 {p[key]}")
    if not (0.0 <= float(p["ind_neutral_weight"]) <= 1.0):
        raise StrategyExportError("ind_neutral_weight 必须位于 [0, 1]")
    if float(p["power"]) < 1.0:
        raise StrategyExportError(f"power 必须 >= 1.0，实际 {p['power']}")
    if not (0.0 <= float(p["memory"]) < 1.0):
        raise StrategyExportError(f"memory 必须位于 [0, 1)，实际 {p['memory']}")
    if not (0.0 < float(p["reserve_cash"]) <= 1.0):
        raise StrategyExportError(f"reserve_cash 必须位于 (0, 1]，实际 {p['reserve_cash']}")
    if int(p["stress_lookback"]) < 2:
        raise StrategyExportError("stress_lookback 必须 >= 2")
    if int(p["stress_z_window"]) < 20:
        raise StrategyExportError("stress_z_window 必须 >= 20")
    if float(p["vol_multiplier"]) <= 0.0:
        raise StrategyExportError("vol_multiplier 必须 > 0")
    if not (0.0 < float(p["power_scale"]) <= 1.0):
        raise StrategyExportError("power_scale 必须位于 (0, 1]")
    for key in ("buy_penalty", "sell_penalty"):
        if float(p[key]) < 0.0:
            raise StrategyExportError(f"{key} 不得为负，实际 {p[key]}")
    if str(p["mode"]) == "horizon" and not (0.0 <= float(p["persistence"]) <= 1.0):
        raise StrategyExportError("persistence 必须位于 [0, 1]")
    if not (0.0 <= float(p["discount"]) <= 1.0):
        raise StrategyExportError("discount 必须位于 [0, 1]")


_STRATEGY_TEMPLATE = '''"""factorsim-compatible strategy exported by FactorLibrary (Phase 3 P1-2).

DO NOT EDIT BY HAND — regenerate with:
    python scripts/export_factorsim_strategy.py --out-dir local_backtest

本策略由 `factorlib.markets.stocks.strategy_export` 从平台声明式 YAML 一键渲染，
完整实现：居中秩信号塑形 → 软行业中性化 → 低波倾斜 → 幂次放大 → EMA 慢速记忆 →
自适应微盘危机风控守卫 → 非对称买卖换手惩罚 + 防追高门槛 → 双期 Horizon 规划 →
reserve_cash 归一化的真实持仓漂移闭环。
"""

import warnings
from collections import deque

import cvxpy as cp
import numpy as np
import pandas as pd
from factorsim.pcmaster.StrategyBase import StrategyBase

warnings.filterwarnings("ignore", category=FutureWarning)

PARAMS = __PARAMS__


def rank_score(values):
    """Centered mid-ranks mapped to [-1, 1]; missing values -> 0.0."""
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    count = int(clean.count())
    if count <= 1:
        return pd.Series(0.0, index=clean.index)
    ranked = clean.rank()
    return ((ranked - (count + 1) / 2.0) * (2.0 / (count - 1))).fillna(0.0)


import re

_KEY_CACHE = {}


def security_key(label):
    s = str(label)
    cached = _KEY_CACHE.get(s)
    if cached is not None:
        return cached
    parts = re.fullmatch(
        r"(?:(?:SH|SZ|BJ)[.]?)?(\\d{1,6})(?:[.](?:SH|SZ|BJ|XSHG|XSHE))?", s.upper()
    )
    res = parts.group(1).zfill(6) if parts else s
    _KEY_CACHE[s] = res
    return res


def prior_snapshot(loader, field, stocks):
    """Barra 风格暴露的前一日快照（内存安全，~1MB）。"""
    panel = loader.get_Barra(field, 5)
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("%s: expected DataFrame from Barra loader" % field)
    keys = pd.Index([security_key(i) for i in stocks])
    rows = pd.Index([security_key(i) for i in panel.index])
    cols = pd.Index([security_key(i) for i in panel.columns])
    if rows.isin(keys).sum() > cols.isin(keys).sum():
        panel = panel.T
        cols = rows
    value = panel.iloc[-1].copy()
    value.index = cols
    value = value.reindex(keys)
    value.index = stocks
    return value


class LowVolIndustryNeutralStrategy(StrategyBase):
    """三层模块化融合策略（信号塑形 + 动态风控守卫 + 凸优化求解）。"""

    DEFAULTS = dict(PARAMS)

    def _cfg(self, key):
        return self.config.get(key, self.DEFAULTS.get(key))

    # ------------------------------------------------------------------
    # 动态风控守卫：基于 Barra SIZE 暴露漂移的微盘流动性危机检测
    # ------------------------------------------------------------------
    def _update_size_stress(self, index):
        """维护 SIZE 暴露截面均值的 10 日累计冲击与 252 日滚动 Z-Score。"""
        if not bool(self._cfg("adaptive_stress_guard")):
            return 0.0
        try:
            size = pd.to_numeric(prior_snapshot(self.dataloader, "SIZE", index), errors="coerce")
        except Exception:
            return 0.0
        level = float(size.mean()) if size.notna().any() else float("nan")
        if not np.isfinite(level):
            return 0.0
        hist = getattr(self, "_size_level_hist", None)
        if hist is None:
            hist = deque(maxlen=int(self._cfg("stress_z_window")) + 5)
            self._size_level_hist = hist
        hist.append(level)
        lookback = int(self._cfg("stress_lookback"))
        if len(hist) <= lookback:
            return 0.0
        cum = level - hist[-1 - lookback]
        recent = np.asarray(list(hist)[:-1], dtype=float)
        stress = 0.0
        if cum <= float(self._cfg("stress_cumret_threshold")):
            stress = 1.0
        if recent.size >= 20:
            mu = float(np.nanmean(recent))
            sd = float(np.nanstd(recent, ddof=1))
            if sd > 1e-12:
                z = (level - mu) / sd
                if z <= -float(self._cfg("stress_z_threshold")):
                    stress = 1.0
        self._size_stress_state = stress
        return stress

    # ------------------------------------------------------------------
    # 信号塑形层
    # ------------------------------------------------------------------
    def _shape_signal(self, signals, index, stress):
        raw_score = rank_score(signals)

        ind_w = float(self._cfg("ind_neutral_weight"))
        if ind_w > 0.0:
            try:
                ind = prior_snapshot(self.dataloader, "INDUSTRY", index)
                valid_ind = ind.notna()
                ind_centered = raw_score.copy()
                if int(valid_ind.sum()) > 50:
                    means = raw_score[valid_ind].groupby(ind[valid_ind]).transform("mean")
                    ind_centered[valid_ind] = raw_score[valid_ind] - ind_w * means
            except Exception:
                ind_centered = raw_score
        else:
            ind_centered = raw_score

        vol_w = float(self._cfg("vol_weight"))
        if stress > 0.0 and bool(self._cfg("adaptive_stress_guard")):
            vol_w *= float(self._cfg("vol_multiplier"))
        if vol_w > 0.0:
            try:
                resvol = pd.to_numeric(
                    prior_snapshot(self.dataloader, "RESVOL", index), errors="coerce"
                )
                adjusted = rank_score(ind_centered) - vol_w * rank_score(resvol)
            except Exception:
                adjusted = rank_score(ind_centered)
        else:
            adjusted = rank_score(ind_centered)

        if stress > 0.0 and bool(self._cfg("adaptive_stress_guard")):
            tilt = float(self._cfg("size_tilt"))
            if tilt > 0.0:
                try:
                    size_raw = pd.to_numeric(
                        prior_snapshot(self.dataloader, "SIZE", index), errors="coerce"
                    )
                    adjusted = adjusted + tilt * rank_score(size_raw)
                except Exception:
                    pass

        power = float(self._cfg("power"))
        if stress > 0.0 and bool(self._cfg("adaptive_stress_guard")):
            power = max(1.0, power * float(self._cfg("power_scale")))
        if power > 1.0:
            current = np.sign(adjusted) * (np.abs(adjusted) ** power)
        else:
            current = adjusted

        slow = getattr(self, "_slow_score", None)
        if slow is None:
            slow = current
        else:
            slow = slow.reindex(index).fillna(current)
        memory = float(self._cfg("memory"))
        self._slow_score_next = memory * slow + (1.0 - memory) * current
        return current, slow

    # ------------------------------------------------------------------
    # 持仓漂移闭环：reserve_cash 归一化
    # ------------------------------------------------------------------
    def _previous_weights(self, last_hold, index):
        if last_hold is None:
            return None
        prev = last_hold.reindex(index).fillna(0.0).astype(float)
        total = float(prev.sum())
        if total <= 1e-12:
            return None
        reserve = float(self._cfg("reserve_cash"))
        return (prev / total * reserve).to_numpy()

    # ------------------------------------------------------------------
    # 凸优化求解层
    # ------------------------------------------------------------------
    def _solve(self, current, slow, previous):
        cap = float(self._cfg("h_max"))
        limit = float(self._cfg("tvr"))
        reserve = float(self._cfg("reserve_cash"))
        n = int(len(current))
        target_sum = 1.0

        w = cp.Variable(n)
        constraints = [w >= 0, w <= cap, cp.sum(w) == target_sum]
        objective = current.to_numpy() @ w

        if previous is not None:
            # 非对称买卖换手惩罚 (asymmetric trade penalty): 快速止损、谨慎建仓
            buy_pen = float(self._cfg("buy_penalty"))
            sell_pen = float(self._cfg("sell_penalty"))
            delta = w - previous
            buys = cp.pos(delta)
            sells = cp.pos(-delta)
            turnover = cp.sum(buys) + cp.sum(sells)
            constraints.append(turnover <= limit)
            objective -= buy_pen * cp.sum(buys) + sell_pen * cp.sum(sells)

            if bool(self._cfg("buy_hurdle")):
                jump = (current - slow).clip(lower=0.0).to_numpy()
                objective -= float(self._cfg("hurdle_penalty")) * (jump @ buys)

        if str(self._cfg("mode")) == "horizon":
            persistence = float(self._cfg("persistence"))
            discount = float(self._cfg("discount"))
            trade_penalty = float(self._cfg("trade_penalty"))
            forecast = persistence * current + (1.0 - persistence) * slow
            future = cp.Variable(n)
            trades = cp.norm1(future - w)
            constraints += [
                future >= 0,
                future <= cap,
                cp.sum(future) == target_sum,
                trades <= limit,
            ]
            objective += discount * (forecast.to_numpy() @ future - trade_penalty * trades)

        problem = cp.Problem(cp.Maximize(objective), constraints)
        mosek_params = {"MSK_IPAR_NUM_THREADS": 1}
        try:
            problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)
        except Exception:
            problem.solve(solver=cp.CLARABEL)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            try:
                problem.solve(
                    solver=cp.MOSEK,
                    mosek_params={
                        "MSK_IPAR_NUM_THREADS": 1,
                        "MSK_IPAR_OPTIMIZER": "MSK_OPTIMIZER_DUAL_SIMPLEX",
                    },
                )
            except Exception:
                problem.solve(solver=cp.CLARABEL)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            if previous is not None:
                return np.asarray(previous, dtype=float) / max(reserve, 1e-9)
            return np.full(n, target_sum / n)

        result = pd.Series(np.asarray(w.value, dtype=float), index=current.index)
        return result.clip(lower=0.0, upper=cap).to_numpy()

    def generate_positions(self, signals, last_hold):
        index = signals.index
        cap = float(self._cfg("h_max"))
        limit = float(self._cfg("tvr"))
        reserve = float(self._cfg("reserve_cash"))

        stress = self._update_size_stress(index)
        current, slow = self._shape_signal(signals, index, stress)
        previous = self._previous_weights(last_hold, index)

        target = self._solve(current, slow, previous)

        result = pd.Series(np.asarray(target, dtype=float), index=index)
        result = result.clip(lower=0.0, upper=cap)
        total_sum = float(result.sum())
        if not np.isfinite(total_sum) or total_sum <= 0.0:
            raise RuntimeError("Degenerate weights produced (sum=%r)" % total_sum)
        if abs(total_sum - 1.0) > 0.05:
            raise RuntimeError("Degenerate weights produced (sum=%.4f)" % total_sum)
        result = result / total_sum

        if previous is not None:
            prev_series = pd.Series(previous, index=index)
            l1 = float(np.abs(result.to_numpy() - previous).sum())
            if l1 > limit + 1e-4:
                result = prev_series + (result - prev_series) * (limit / l1)
            result = result.clip(lower=0.0, upper=cap)
            result = result / float(result.sum())

        self._slow_score = getattr(self, "_slow_score_next", current)
        return result
'''


def render_factorsim_strategy(params: Mapping[str, Any] | None = None) -> str:
    """把参数集渲染为 `factorsim` 兼容 `strategy.py` 源码（可 `compile()` 校验）。"""
    resolved = FactorsimStrategyParams(params or {}).to_dict()
    body = _STRATEGY_TEMPLATE.replace("__PARAMS__", _format_params_literal(resolved))
    return body


def _format_params_literal(params: Mapping[str, Any]) -> str:
    """生成确定顺序、可读的 Python 字面量（避免 JSON 的 true/false）。"""
    lines = ["{"]
    for key in sorted(params):
        value = params[key]
        if isinstance(value, bool):
            rendered = "True" if value else "False"
        elif isinstance(value, str):
            rendered = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, float):
            rendered = repr(float(value))
        else:
            rendered = repr(value)
        lines.append(f"    {key!r}: {rendered},")
    lines.append("}")
    return "\n".join(lines)


def resolve_strategy_params(
    raw_strategy: Mapping[str, Any] | None = None,
    raw_backtest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从平台 YAML 的 `strategy` / `backtest` 段归并出导出参数（含嵌套非对称惩罚）。"""
    merged: dict[str, Any] = dict(DEFAULT_STRATEGY_PARAMS)
    raw_strategy = dict(raw_strategy or {})
    raw_backtest = dict(raw_backtest or {})

    asym = raw_strategy.pop("asymmetric_trade_penalty", None)
    if isinstance(asym, Mapping):
        if "buy" in asym:
            merged["buy_penalty"] = float(asym["buy"])
        if "sell" in asym:
            merged["sell_penalty"] = float(asym["sell"])
    merged.update(
        {
            k: v
            for k, v in raw_strategy.items()
            if k in merged or k in {"buy_penalty", "sell_penalty"}
        }
    )
    if "reserve_cash" in raw_backtest:
        merged["reserve_cash"] = float(raw_backtest["reserve_cash"])
    if "fee_rate" in raw_backtest:
        merged["_fee_rate"] = float(raw_backtest["fee_rate"])
    validate_strategy_params({k: v for k, v in merged.items() if not k.startswith("_")})
    return merged


def render_factorsim_config(
    params: Mapping[str, Any] | None = None,
    backtest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """渲染 `factorsim` 作业 `config.yaml` 的字典结构。"""
    resolved = FactorsimStrategyParams(params or {}).to_dict()
    bt = dict(DEFAULT_BACKTEST_PARAMS)
    bt.update({k: v for k, v in dict(backtest or {}).items() if v is not None})

    strategy_config: dict[str, Any] = {
        "strategy_class": "LowVolIndustryNeutralStrategy",
        "strategy_path": "./strategy.py",
    }
    for key in sorted(resolved):
        if key in ("buy_penalty", "sell_penalty", "reserve_cash"):
            continue
        strategy_config[key] = resolved[key]
    strategy_config["asymmetric_trade_penalty"] = {
        "buy": float(resolved["buy_penalty"]),
        "sell": float(resolved["sell_penalty"]),
    }

    return {
        "type": "backtest",
        "universe": str(bt.get("universe", "base")),
        "signal_path": str(bt.get("signal_path", DEFAULT_BACKTEST_PARAMS["signal_path"])),
        "output_path": str(bt.get("output_path", DEFAULT_BACKTEST_PARAMS["output_path"])),
        "strategy_config": strategy_config,
        "backtest_config": {
            "cash": float(bt["cash"]),
            "reserve_cash": float(resolved["reserve_cash"]),
            "fee_rate": float(bt["fee_rate"]),
            "start_ds": int(bt["start_ds"]),
            "end_ds": int(bt["end_ds"]),
            "output_path": str(bt.get("output_path", DEFAULT_BACKTEST_PARAMS["output_path"])),
            "verbose": bool(bt.get("verbose", True)),
        },
    }


def load_strategy_yaml(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取平台声明式 YAML，返回 `(strategy_params, backtest_params)`。"""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise StrategyExportError(f"{p} 顶层必须为映射")
    strategy = raw.get("strategy") or raw.get("portfolio", {}).get("optimizer") or {}
    backtest = raw.get("backtest") or {}
    if not isinstance(strategy, Mapping) or not isinstance(backtest, Mapping):
        raise StrategyExportError(f"{p} 的 strategy/backtest 段必须为映射")
    return dict(strategy), dict(backtest)


def export_factorsim_strategy(
    out_dir: str | Path,
    params: Mapping[str, Any] | None = None,
    backtest: Mapping[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """把渲染结果落盘为 `strategy.py` + `config.yaml`，返回写入路径字典。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    strategy_path = out / "strategy.py"
    config_path = out / "config.yaml"

    if not overwrite:
        for target in (strategy_path, config_path):
            if target.exists():
                raise StrategyExportError(
                    f"目标文件已存在: {target}（如需覆盖请显式 overwrite=True）"
                )

    source = render_factorsim_strategy(params)
    compile(source, str(strategy_path), "exec")  # 语法自校验，绝不允许落盘非法源码
    strategy_path.write_text(source, encoding="utf-8")

    cfg = render_factorsim_config(params, backtest)
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {"strategy": strategy_path, "config": config_path}
