"""纯 Python A 股持仓级逐日账本参考实现 (tests/python/oracles/stock_ledger_ref.py)。

用于与 Rust `factorlib_core.run_stock_backtest` 做逐日资金、持仓、税费与阻塞订单对账，
要求 ``|Δequity| < 1e-9``。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _is_valid_price(p: float) -> bool:
    return math.isfinite(p) and p > 0.0


def _order_costs(
    delta: int,
    p_exec: float,
    commission_rate: float,
    min_commission: float,
    stamp_duty_rate: float,
    transfer_fee_rate: float,
    slip_ticks: float,
    tick_size: float,
    slip_bps: float,
) -> tuple[float, float, float]:
    if delta == 0:
        return 0.0, 0.0, 0.0
    qty = float(abs(delta))
    notional = qty * p_exec
    comm = (
        max(notional * commission_rate, min_commission)
        if (commission_rate > 0.0 or min_commission > 0.0)
        else 0.0
    )
    transfer = notional * transfer_fee_rate
    tax = notional * stamp_duty_rate if delta < 0 else 0.0
    slip = qty * (slip_ticks * tick_size) + notional * (slip_bps / 10_000.0)
    return comm + transfer, tax, slip


def run_stock_ledger_ref(
    target_weights: np.ndarray,
    exec_price: np.ndarray,
    close_price: np.ndarray,
    can_buy_mask: np.ndarray,
    can_sell_mask: np.ndarray,
    is_suspended: np.ndarray,
    *,
    init_cash: float = 10_000_000.0,
    lot_size: int = 100,
    t_plus_one: bool = True,
    allow_short: bool = False,
    commission_rate: float = 0.0001,
    min_commission: float = 5.0,
    stamp_duty_rate: float = 0.0005,
    transfer_fee_rate: float = 0.00001,
    slip_ticks: float = 1.0,
    tick_size: float = 0.01,
    slip_bps: float = 0.0,
) -> dict[str, Any]:
    """纯 Python 逐日现金+持仓账本，逐笔模拟 A 股整手、T+1、涨跌停阻断与停牌盯市。"""
    if allow_short:
        raise ValueError("A 股融券做空回测暂未实现 (allow_short=False)")

    T, N = target_weights.shape
    equity = np.zeros(T, dtype=np.float64)
    returns = np.zeros(T, dtype=np.float64)
    cash = np.zeros(T, dtype=np.float64)
    gross_pnl = np.zeros(T, dtype=np.float64)
    net_pnl = np.zeros(T, dtype=np.float64)
    fee = np.zeros(T, dtype=np.float64)
    slippage = np.zeros(T, dtype=np.float64)
    tax = np.zeros(T, dtype=np.float64)
    turnover = np.zeros(T, dtype=np.float64)
    blocked_orders = np.zeros(T, dtype=np.int64)
    positions = np.zeros((T, N), dtype=np.int64)

    if T == 0:
        return {
            "equity": equity,
            "returns": returns,
            "cash": cash,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "fee": fee,
            "slippage": slippage,
            "tax": tax,
            "turnover": turnover,
            "blocked_orders": blocked_orders,
            "positions": positions,
        }

    shares = [0] * N
    prev_close = [0.0] * N
    for i in range(N):
        p_c = float(close_price[0, i])
        p_e = float(exec_price[0, i])
        if not bool(is_suspended[0, i]) and _is_valid_price(p_c):
            prev_close[i] = p_c
        elif not bool(is_suspended[0, i]) and _is_valid_price(p_e):
            prev_close[i] = p_e
        else:
            prev_close[i] = 0.0

    cash_bal = float(init_cash)
    equity[0] = cash_bal
    cash[0] = cash_bal
    lot_f = float(lot_size)

    for d in range(1, T):
        prev_e = equity[d - 1]
        sellable = list(shares)
        exec_delta = [0] * N
        day_blocked = 0

        for i in range(N):
            w = float(target_weights[d - 1, i])
            p_exec = float(exec_price[d, i])
            susp = bool(is_suspended[d, i])

            if not math.isfinite(w):
                continue

            if w <= 0.0:
                if shares[i] > 0:
                    if susp or not bool(can_sell_mask[d, i]) or not _is_valid_price(p_exec):
                        day_blocked += 1
                    else:
                        exec_delta[i] = -(sellable[i] if t_plus_one else shares[i])
                continue

            if susp or not _is_valid_price(p_exec):
                day_blocked += 1
                continue

            target_val = w * max(prev_e, 0.0)
            raw_target = target_val / p_exec
            cur_s = float(shares[i])
            if raw_target >= cur_s:
                buy_qty = max(int(math.floor((raw_target - cur_s) / lot_f)), 0) * lot_size
                if buy_qty > 0:
                    if not bool(can_buy_mask[d, i]):
                        day_blocked += 1
                    else:
                        exec_delta[i] = buy_qty
            else:
                sell_qty = max(int(math.floor((cur_s - raw_target) / lot_f)), 0) * lot_size
                max_sell = sellable[i] if t_plus_one else shares[i]
                capped_sell = min(sell_qty, max_sell)
                if capped_sell > 0:
                    if not bool(can_sell_mask[d, i]):
                        day_blocked += 1
                    else:
                        exec_delta[i] = -capped_sell

        avail_cash = cash_bal
        for i in range(N):
            if exec_delta[i] < 0:
                p_exec = float(exec_price[d, i])
                f, tx, sl = _order_costs(
                    exec_delta[i],
                    p_exec,
                    commission_rate,
                    min_commission,
                    stamp_duty_rate,
                    transfer_fee_rate,
                    slip_ticks,
                    tick_size,
                    slip_bps,
                )
                avail_cash += float(-exec_delta[i]) * p_exec - (f + tx + sl)

        def _buy_need(deltas: list[int], day_idx: int = d) -> float:
            tot = 0.0
            for idx in range(N):
                if deltas[idx] > 0:
                    pe = float(exec_price[day_idx, idx])
                    f_i, _tx_i, sl_i = _order_costs(
                        deltas[idx],
                        pe,
                        commission_rate,
                        min_commission,
                        stamp_duty_rate,
                        transfer_fee_rate,
                        slip_ticks,
                        tick_size,
                        slip_bps,
                    )
                    tot += float(deltas[idx]) * pe + f_i + sl_i
            return tot

        cash_cap = max(avail_cash, 0.0)
        buy_need = _buy_need(exec_delta)
        if buy_need > cash_cap:
            if cash_cap <= 0.0:
                for i in range(N):
                    if exec_delta[i] > 0:
                        exec_delta[i] = 0
            else:
                scale = min(max(cash_cap / buy_need, 0.0), 1.0)
                for i in range(N):
                    if exec_delta[i] > 0:
                        exec_delta[i] = (
                            max(int(math.floor((float(exec_delta[i]) * scale) / lot_f)), 0)
                            * lot_size
                        )
                buy_need = _buy_need(exec_delta)
                while buy_need > cash_cap + 1e-9:
                    trimmed = False
                    for i in range(N):
                        if exec_delta[i] >= lot_size:
                            exec_delta[i] -= lot_size
                            trimmed = True
                            buy_need = _buy_need(exec_delta)
                            if buy_need <= cash_cap + 1e-9:
                                break
                    if not trimmed:
                        break

        d_fee = 0.0
        d_tax = 0.0
        d_slip = 0.0
        d_notional = 0.0
        d_gross = 0.0
        mv = 0.0

        for i in range(N):
            delta = exec_delta[i]
            p_ex_raw = float(exec_price[d, i])
            p_cl_raw = float(close_price[d, i])
            susp = bool(is_suspended[d, i])

            if delta != 0:
                f, tx, sl = _order_costs(
                    delta,
                    p_ex_raw,
                    commission_rate,
                    min_commission,
                    stamp_duty_rate,
                    transfer_fee_rate,
                    slip_ticks,
                    tick_size,
                    slip_bps,
                )
                d_fee += f
                d_tax += tx
                d_slip += sl
                d_notional += float(abs(delta)) * p_ex_raw
                cash_bal -= float(delta) * p_ex_raw + f + tx + sl

            if not susp and _is_valid_price(p_cl_raw):
                p_mtm = p_cl_raw
            elif not susp and _is_valid_price(p_ex_raw):
                p_mtm = p_ex_raw
            else:
                p_mtm = prev_close[i]

            p_ex_pnl = p_ex_raw if _is_valid_price(p_ex_raw) else p_mtm
            d_gross += float(shares[i]) * (p_mtm - prev_close[i]) + float(delta) * (
                p_mtm - p_ex_pnl
            )

            shares[i] += delta
            prev_close[i] = p_mtm
            positions[d, i] = shares[i]
            mv += float(shares[i]) * p_mtm

        d_net = d_gross - d_fee - d_tax - d_slip
        curr_e = cash_bal + mv
        equity[d] = curr_e
        cash[d] = cash_bal
        gross_pnl[d] = d_gross
        net_pnl[d] = d_net
        fee[d] = d_fee
        slippage[d] = d_slip
        tax[d] = d_tax
        turnover[d] = d_notional / max(prev_e, 1.0)
        returns[d] = d_net / max(prev_e, 1.0)
        blocked_orders[d] = day_blocked

    return {
        "equity": equity,
        "returns": returns,
        "cash": cash,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fee": fee,
        "slippage": slippage,
        "tax": tax,
        "turnover": turnover,
        "blocked_orders": blocked_orders,
        "positions": positions,
    }
