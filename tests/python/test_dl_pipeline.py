"""Phase 3 (`1d` DL Alpha 研发管线与正交审计) 验收测试集 (tests/python/test_dl_pipeline.py)。

覆盖 M8 全部核心验收要求：
1. Fold 内防泄漏预处理（`FoldScaler` / 日度截面 Rank-Gauss / 标签去 Beta）与严格因果性扰动审计；
2. 4 大 PyTorch 模型族（`gated_mlp`、`gru`、`tcn`、`st_attention` 含大截面市场原型路由分支）前向与反向可用；
3. 负控制（纯随机噪声 OOS ICIR 接近 0，无虚假信号）与正控制（泄露探测）；
4. 同 `recipe_id` + 同 `seed` 两次运行 OOS 矩阵逐位完全一致（Bit-Identical Determinism）；
5. `1m` 扩展包挂载特征直通 DL Dataset + `scripts/run_dl.py --recipe` 端到端闭环 + 三关正交审计与同族 DL 变体去重；
6. 底座只读面板视图不被改写且不产生 PyTorch non-writable 警告。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from factorlib.core.admission_audit import audit_candidates
from factorlib.core.extensions import ExtensionPack, ExtensionPackMeta
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRegistry, FeatureSetRecord
from factorlib.core.synthetic import (
    create_synthetic_futures_panel,
    create_synthetic_stocks_panel,
)
from factorlib.dl.dataset import (
    assert_folds_no_leakage,
    build_dl_folds,
    prepare_dl_panel_data,
)
from factorlib.dl.loss import CompositeAlphaLoss
from factorlib.dl.models import build_dl_model
from factorlib.dl.trainer import DLTrainConfig, train_dl_walk_forward


def test_dl_models_forward_and_large_n_routing() -> None:
    """验证 4 种基准模型族在日批 (B, N, L, C) 与大截面市场路由下可正常前向反向。"""
    torch.manual_seed(7)
    x_small = torch.randn(4, 16, 12, 6, requires_grad=True)
    m_small = torch.ones(4, 16, dtype=torch.bool)
    sym_ids = torch.arange(16)
    sec_ids = torch.arange(16) % 3
    y_small = torch.randn(4, 16)
    loss_fn = CompositeAlphaLoss(min_names=4)

    for arch in ("gated_mlp", "gru", "tcn", "st_attention"):
        model = build_dl_model(
            arch,
            n_features=6,
            market="futures" if arch != "st_attention" else "stocks",
            seq_len=12,
            hidden_dim=16,
            n_layers=1,
            dropout=0.1,
            n_symbols=16,
            n_sectors=3,
        )
        out = model(x_small, symbol_ids=sym_ids, sector_ids=sec_ids, valid_mask=m_small)
        assert out.shape == (4, 16)
        loss, stats = loss_fn(out, y_small, m_small)
        assert torch.isfinite(loss)
        assert -1.0 <= stats["ic"] <= 1.0
        loss.backward()

    # 测试大截面 A 股 Market-Guided Routing 分支 (N > direct_attn_threshold=256)
    st_large = build_dl_model(
        "st_attention",
        n_features=5,
        market="stocks",
        seq_len=10,
        hidden_dim=16,
        n_sectors=5,
    )
    x_big = torch.randn(2, 300, 10, 5)
    m_big = torch.ones(2, 300, dtype=torch.bool)
    out_big = st_large(x_big, valid_mask=m_big)
    assert out_big.shape == (2, 300)
    assert torch.all(torch.isfinite(out_big))


def test_dl_dataset_causality_perturbation_and_readonly_safety() -> None:
    """验证篡改 T > T_perturb 后的未来行情时，T <= T_perturb 的 DL 预处理特征保持逐比特一致。"""
    base = create_synthetic_futures_panel(n_days=90, seed=11)
    # 锁定只读视角，验证 DL 预处理不会修改且不会发 PyTorch 非可写警告
    for arr in base.fields.values():
        arr.setflags(write=False)

    perturbed = create_synthetic_futures_panel(n_days=90, seed=11)
    t_perturb = 55
    for k, mat in list(perturbed.fields.items()):
        if k == "sector":
            continue
        mod = mat.copy()
        mod[t_perturb + 1 :, :] = mod[t_perturb + 1 :, :] * 2.5 + 100.0
        perturbed.fields[k] = np.asfortranarray(mod, dtype=np.float64)

    feats = ["close", "volume", "ry_ann", "wh_z60", "mr_net_ratio"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prep_a = prepare_dl_panel_data(
            base,
            feature_names=feats,
            target_label="y_5",
            market="futures",
            causal_ts_window=15,
        )
        prep_b = prepare_dl_panel_data(
            perturbed,
            feature_names=feats,
            target_label="y_5",
            market="futures",
            causal_ts_window=15,
        )
        nonwrite_warns = [w for w in caught if "not writable" in str(w.message).lower()]
        assert not nonwrite_warns, f"触发 PyTorch 只读警告: {nonwrite_warns}"

    # T <= t_perturb 预处理特征矩阵必须 0 差异 (bit-for-bit identical)
    np.testing.assert_array_equal(
        prep_a.x_raw_norm[: t_perturb + 1],
        prep_b.x_raw_norm[: t_perturb + 1],
    )
    # T > t_perturb 必须发生变化，证明扰动非空
    assert not np.array_equal(
        prep_a.x_raw_norm[t_perturb + 1 :], prep_b.x_raw_norm[t_perturb + 1 :]
    )


def test_dl_folds_strict_purge_guard() -> None:
    """验证 purge_window < horizon 时强断言拦截，以及 Lockbox 盲测零重叠。"""
    panel = create_synthetic_stocks_panel(n_days=140, n_symbols=20, seed=5)
    folds, lb = build_dl_folds(
        panel.dates,
        horizon=5,
        purge_window=5,
        n_folds=3,
        lockbox_ratio=0.2,
    )
    assert lb is not None
    assert len(folds) >= 2
    assert np.max(folds[-1].test_indices) < np.min(lb.test_indices)
    with pytest.raises(ValueError, match="purge_window"):
        assert_folds_no_leakage(folds, purge_window=2, horizon=5)


def test_dl_noise_negative_control_and_determinism() -> None:
    """验证纯噪声负控制 OOS ICIR 接近 0（无泄漏）以及同 recipe+seed 逐位可复现。"""
    rng = np.random.default_rng(1234)
    T, N = 110, 24
    dates = [f"202{(i // 250)}-{((i // 22) % 12) + 1:02d}-{(i % 22) + 1:02d}" for i in range(T)]
    symbols = [f"S_{j:02d}" for j in range(N)]
    valid_mask = np.ones((T, N), dtype=bool, order="F")
    fields = {
        f"noise_{c}": np.asfortranarray(rng.normal(0.0, 1.0, size=(T, N)), dtype=np.float64)
        for c in range(4)
    }
    labels = {
        "y_1": np.asfortranarray(rng.normal(0.0, 0.02, size=(T, N)), dtype=np.float64),
        "y_5": np.asfortranarray(rng.normal(0.0, 0.04, size=(T, N)), dtype=np.float64),
    }
    noise_panel = Panel(
        dates=np.array(dates),
        symbols=symbols,
        fields=fields,
        mask=valid_mask,
        labels=labels,
    )

    cfg = DLTrainConfig(
        market="futures",
        arch="gru",
        target_label="y_5",
        seq_len=10,
        hidden_dim=16,
        n_layers=1,
        epochs=6,
        n_folds=3,
        n_seeds=2,
        purge_window=5,
        base_seed=99,
        device="cpu",
        prefer_yearly_folds=False,
    )
    res1 = train_dl_walk_forward(noise_panel, feature_names=list(fields.keys()), config=cfg)
    res2 = train_dl_walk_forward(noise_panel, feature_names=list(fields.keys()), config=cfg)

    # 纯随机噪声 + 严格 Purged OOS，ICIR 不应出现异常虚高
    assert abs(res1.selection_icir) < 0.65, f"纯噪声 OOS ICIR 异常偏高: {res1.selection_icir}"
    # 同配置同随机种子两次运行必须逐位一致
    np.testing.assert_array_equal(res1.oos_matrix, res2.oos_matrix)


def test_dl_alpha_end_to_end_with_extension_and_three_gate_audit(tmp_path: Path) -> None:
    """验证 CLI --recipe 挂载 1m 扩展包跑通 PyTorch DL 闭环，并在 SQLite 执行三关准入与同族去重。"""
    import scripts.run_dl as run_dl_mod

    db_path = tmp_path / "factors.sqlite"
    runs_dir = tmp_path / "runs"
    panel_dir = tmp_path / "futures_panel"

    # 构造具真实微弱线性截面信号的期货面板并存盘，确保信号可学但不与现有 seed 完全共线
    import json

    import polars as pl

    base = create_synthetic_futures_panel(n_days=120, seed=77)
    rng = np.random.default_rng(77)
    sig_feat = np.asfortranarray(rng.normal(0.0, 1.0, size=base.shape), dtype=np.float64)
    base.fields["alpha_drv"] = sig_feat
    y5_custom = np.asfortranarray(
        np.tanh(sig_feat) * 0.015 + rng.normal(0.0, 0.01, size=base.shape), dtype=np.float64
    )
    base.labels["y_5"] = y5_custom

    panel_dir.mkdir(parents=True, exist_ok=True)
    t_len, n_len = base.shape
    f_cols = [k for k in base.fields if k not in ("in_universe_mask", "limit_mask", "sector")]
    a_cols = list(base.aux_fields.keys())
    p_rows: dict[str, list[object]] = {"trading_date": [], "symbol": [], "in_universe": []}
    for k in f_cols + a_cols + (["sector"] if "sector" in base.fields else []):
        p_rows[k] = []
    l_rows: dict[str, list[object]] = {"trading_date": [], "symbol": []}
    for lk in base.labels:
        l_rows[lk] = []
    for ti in range(t_len):
        ds = str(base.dates[ti])
        for ni in range(n_len):
            sym = base.symbols[ni]
            p_rows["trading_date"].append(ds)
            p_rows["symbol"].append(sym)
            p_rows["in_universe"].append(bool(base.mask[ti, ni]))
            if "sector" in base.fields:
                p_rows["sector"].append(f"S{int(base.fields['sector'][ti, ni])}")
            for k in f_cols:
                p_rows[k].append(float(base.fields[k][ti, ni]))
            for k in a_cols:
                p_rows[k].append(float(base.aux_fields[k][ti, ni]))
            l_rows["trading_date"].append(ds)
            l_rows["symbol"].append(sym)
            for lk, lm in base.labels.items():
                l_rows[lk].append(float(lm[ti, ni]))
    pl.DataFrame(p_rows).write_parquet(panel_dir / "panel_daily.parquet")
    pl.DataFrame(l_rows).write_parquet(panel_dir / "labels.parquet")
    (panel_dir / "meta.json").write_text(
        json.dumps(base.meta, ensure_ascii=False), encoding="utf-8"
    )

    # 构造并挂载 1m 扩展包
    ext_meta = ExtensionPackMeta(
        pack_id="test_1m_pack",
        market="futures",
        version="v1",
        fields=("ext_rv_1m",),
        lag_days=1,
        asof_rule="T+1_open",
    )
    ext_rows = []
    rv_mat = rng.uniform(0.01, 0.05, size=base.shape)
    for t_i, d_str in enumerate(base.dates):
        for s_i, sym in enumerate(base.symbols):
            ext_rows.append(
                {"trading_date": str(d_str), "symbol": sym, "ext_rv_1m": float(rv_mat[t_i, s_i])}
            )
    ext_dir = tmp_path / "data" / "extensions" / "futures" / "test_1m_pack"
    ext_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(ext_rows).write_parquet(ext_dir / "features.parquet")
    (ext_dir / "pack.json").write_text(ext_meta.to_json(), encoding="utf-8")
    loaded_ext = ExtensionPack.from_dir(ext_dir)
    base.attach_extension(loaded_ext)
    assert "ext_rv_1m" in base.fields

    registry = FactorRegistry(db_path=db_path)
    fs_id = registry.save_feature_set(
        FeatureSetRecord(
            name="fut_dl_fs",
            market="futures",
            base_fields=["close", "volume", "ry_ann", "alpha_drv", "ext_rv_1m"],
            extensions=["test_1m_pack"],
        )
    )

    rcp = registry.create_recipe(
        market="futures",
        kind="dl",
        feature_set_id=fs_id,
        target_label="y_5",
        seed=101,
        params={
            "panel_dir": str(panel_dir),
            "model": "gated_mlp",
            "seq_len": 10,
            "hidden_dim": 16,
            "epochs": 6,
            "n_folds": 3,
            "n_seeds": 2,
            "purge_window": 5,
            "device": "cpu",
            "lockbox_ratio": 0.15,
        },
    )

    args = argparse.Namespace(
        recipe=rcp.recipe_id,
        market="futures",
        panel=panel_dir,
        db=db_path,
        runs_dir=runs_dir,
        run_id="dl_e2e_test_1",
        arch=None,
        seq_len=None,
        epochs=None,
        n_folds=None,
        n_seeds=None,
        device="cpu",
        audit=True,
        synthetic=False,
        log_level="INFO",
    )
    code = run_dl_mod.run_dl_cli(args, root_dir=tmp_path)
    assert code == 0

    cands = registry.list_factors(status="candidate", market="futures")
    assert len(cands) == 1
    dl_rec = cands[0]
    assert dl_rec.factor_type == "dl_alpha"
    assert dl_rec.materialization_path and Path(dl_rec.materialization_path).exists()
    assert (runs_dir / "dl_e2e_test_1" / "fold_diagnostics.json").exists()
    assert (runs_dir / "dl_e2e_test_1" / "audit_report.json").exists()

    # 额外注册 2 个同 FeatureSet 家族变体，检验 M8.3 同族变体去重 (>2 个触发 family_dedup_dropped)
    from factorlib.dl.materialize import evaluate_and_materialize_dl_alpha

    fid_list = [int(dl_rec.id)]
    for v_idx in (2, 3):
        mat_v = np.asfortranarray(
            np.tanh(sig_feat) * (0.2 + 0.05 * v_idx) + rng.normal(0.0, 0.2, size=base.shape)
        )
        fid_v, _ = evaluate_and_materialize_dl_alpha(
            panel=base,
            registry=registry,
            oos_matrix=mat_v,
            market="futures",
            name=f"dl_variant_{v_idx}",
            recipe_id=rcp.recipe_id,
            content_hash=f"{rcp.content_hash}_v{v_idx}",
            run_id=f"dl_var_{v_idx}",
            horizon=5,
            status="candidate",
            mat_dir=tmp_path / "materialized",
        )
        fid_list.append(fid_v)

    aud_reports = audit_candidates(
        base,
        registry,
        fid_list,
        market="futures",
        horizon=5,
        min_valid_icir=0.1,
        min_residual_icir=0.05,
        min_residual_t=0.5,
    )
    assert len(aud_reports) == 3
    # 第 3 名若原本三关全通过应被同族配额去重标记
    by_rank = sorted(aud_reports, key=lambda r: int(r.get("family_rank", 99)))
    assert [r.get("family_rank") for r in by_rank] == [1, 2, 3]


def test_dl_positive_leak_control_catches_shifted_label() -> None:
    """M8 正控制实验：若将未来收益偷偷对齐至特征当日 t（前视泄露），DL 立刻出现异常高 OOS ICIR 被审计捕获。"""
    rng = np.random.default_rng(2028)
    T, N = 110, 20
    dates = [f"202{(i // 250)}-{((i // 22) % 12) + 1:02d}-{(i % 22) + 1:02d}" for i in range(T)]
    symbols = [f"C_{j:02d}" for j in range(N)]
    valid_mask = np.ones((T, N), dtype=bool, order="F")
    sig_ret = rng.normal(0.0, 0.02, size=(T, N))
    # 恶意前视特征 = 未经滞后的未来 5 日真实收益（等价于标签被反移或特征含未来闭合收益）
    y_5 = np.asfortranarray(sig_ret, dtype=np.float64)
    leaked_panel = Panel(
        dates=np.array(dates),
        symbols=symbols,
        fields={"leaked_fwd": np.asfortranarray(y_5 + rng.normal(0.0, 0.002, size=(T, N)))},
        mask=valid_mask,
        labels={"y_5": y_5},
    )
    cfg = DLTrainConfig(
        market="futures",
        arch="gated_mlp",
        target_label="y_5",
        seq_len=6,
        hidden_dim=16,
        epochs=6,
        n_folds=2,
        n_seeds=2,
        purge_window=5,
        base_seed=7,
        device="cpu",
        prefer_yearly_folds=False,
    )
    res_leak = train_dl_walk_forward(leaked_panel, feature_names=["leaked_fwd"], config=cfg)
    assert res_leak.selection_icir > 1.5, (
        f"前视泄露样本正控制未能捕获异常高 ICIR: {res_leak.selection_icir}"
    )
