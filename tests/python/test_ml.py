"""P5 验收测试：Walk-Forward 机器学习多因子合成管线 (tests/python/test_ml.py)。

覆盖白皮书第 11 章与 P5 阶段全部验收指标：
1. `ml/splits.py` (Purged & Embargoed 扩展/滚动窗口切分器)：
   - 验证无重叠严格因果隔离：训练集、验证集、测试集与 Purge 隔离区两两互斥；
   - 验证 Purge 机制在训练集与验证集末尾精确剔除 h 个交易日，消除前瞻标签泄漏；
   - 验证 Embargo 机制在验证集/测试集前端的静默隔离；
   - 验证 expanding 与 rolling 窗口行为。
2. `ml/dataset.py` (特征工程标准流)：
   - 验证方向对齐 (align_directions)；
   - 验证截面中位数极值截断 (MAD Winsorization) 有效抑制异常值且保真正常值；
   - 验证截面排序高斯化 (Rank Gauss Transform) 单调保序、均值近零、方差近 1、有限无 Inf；
   - 验证 `MLDatasetBuilder` 平铺拼装与面板回填二维重构。
3. `ml/models` (基线与核心模型族)：
   - 验证 Equal Weight、Rolling ICIR、Ridge 与 LightGBM 拟合与预测；
   - 验证 LightGBM 超参硬约束、早停机制与特征增益提取。
4. `ml/pipeline.py` (端到端合成与优势验收)：
   - 验证 WalkForwardPipeline 全流程串通；
   - 验证导出 `pred_panel.parquet` 格式完备；
   - 验证在带非线性交互的信号面板上，LightGBM 样本外 RankIC 表现稳健优异。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from factorlib.core.config import LightGBMConfig, MLConfig, WalkForwardConfig
from factorlib.core.panel import Panel
from factorlib.ml.dataset import (
    MLDatasetBuilder,
    align_directions,
    mad_winsorize,
    preprocess_factors,
    rank_gauss_transform,
)
from factorlib.ml.models import (
    LightGBMModel,
    create_model,
)
from factorlib.ml.pipeline import WalkForwardPipeline, _calc_rank_ic
from factorlib.ml.splits import PurgedWalkForwardSplitter, WalkForwardFold


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def sample_multi_year_dates() -> np.ndarray:
    """生成 2011 到 2020 年，每年 250 个交易日的日期数组。"""
    dates = []
    for y in range(2011, 2021):
        for d in range(1, 251):
            month = ((d - 1) // 25) + 1
            day = ((d - 1) % 25) + 1
            dates.append(f"{y}-{month:02d}-{day:02d}")
    return np.array(dates)


@pytest.fixture
def synthetic_ml_panel() -> tuple[Panel, dict[str, np.ndarray], dict[str, int]]:
    """构建带非线性特征组合与多时段的合成面板与因子库。"""
    rng = np.random.default_rng(20261020)
    # 构造 2014 到 2020 年每年 60 天，共 420 天
    dates_list = []
    for y in range(2014, 2021):
        for d in range(1, 61):
            month = ((d - 1) // 20) + 1
            day = ((d - 1) % 20) + 1
            dates_list.append(f"{y}-{month:02d}-{day:02d}")

    dates = np.array(dates_list)
    T = len(dates)
    N = 12
    symbols = [f"SYM_{i:02d}" for i in range(N)]

    # 基础特征
    f1 = rng.normal(0.0, 1.0, size=(T, N))  # 动量型
    f2 = rng.normal(0.0, 1.0, size=(T, N))  # 展期型
    f3 = rng.normal(0.0, 1.0, size=(T, N))  # 噪声型

    # 注入轻微缺失
    f1[rng.random((T, N)) < 0.05] = np.nan

    # 制造非线性与交互关系的真实标签 y_5:
    # y_5 = 0.5 * f1 + 0.4 * f2 + 0.6 * (f1 * f2) + noise
    clean_f1 = np.nan_to_num(f1, nan=0.0)
    y_5 = 0.50 * clean_f1 + 0.40 * f2 + 0.60 * (clean_f1 * f2) + rng.normal(0.0, 0.2, size=(T, N))

    fields = {
        "close": np.asfortranarray(np.abs(f1) + 100.0),
        "volume": np.asfortranarray(np.abs(f2) * 1000.0 + 500.0),
    }
    labels = {
        "y_5": np.asfortranarray(y_5),
    }
    mask = np.ones((T, N), dtype=bool, order="F")

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields=fields,
        mask=mask,
        labels=labels,
        meta={"panel_version": "v1_test"},
    )

    factor_matrices = {
        "factor_1": np.asfortranarray(f1),
        "factor_2": np.asfortranarray(f2),
        "factor_noise": np.asfortranarray(f3),
    }
    factor_signs = {
        "factor_1": 1,
        "factor_2": 1,
        "factor_noise": 1,
    }

    return panel, factor_matrices, factor_signs


# ------------------------------------------------------------------------------
# 1. PurgedWalkForwardSplitter 单元测试
# ------------------------------------------------------------------------------
def test_purged_walk_forward_expanding(sample_multi_year_dates: np.ndarray):
    """验证扩展窗口切分器的因果防未来隔离红线与 Purge 剔除逻辑。"""
    h = 5
    splitter = PurgedWalkForwardSplitter(
        start_predict_year=2018,
        end_predict_year=2020,
        train_window="expanding",
        min_train_years=4,
        purge_window=h,
        validation_size_days=250,
        embargo_window=0,
    )

    folds = splitter.split(sample_multi_year_dates)
    assert len(folds) == 3
    assert [f.predict_year for f in folds] == [2018, 2019, 2020]

    for fold in folds:
        assert isinstance(fold, WalkForwardFold)
        assert fold.n_train > 0
        assert fold.n_val > 0
        assert fold.n_test > 0

        # 1. 严格时序单调递增无未来泄漏
        max_train = int(np.max(fold.train_indices))
        min_purge_train = int(np.min(fold.purge_train_indices))
        max_purge_train = int(np.max(fold.purge_train_indices))
        min_val = int(np.min(fold.val_indices))
        max_val = int(np.max(fold.val_indices))
        min_purge_val = int(np.min(fold.purge_val_indices))
        max_purge_val = int(np.max(fold.purge_val_indices))
        min_test = int(np.min(fold.test_indices))

        assert max_train < min_purge_train
        assert max_purge_train < min_val
        assert max_val < min_purge_val
        assert max_purge_val < min_test

        # 2. 精确 Purge 剔除天数核验
        assert len(fold.purge_train_indices) == h
        assert len(fold.purge_val_indices) == h

        # 3. 集合两两互斥
        train_set = set(fold.train_indices.tolist())
        pt_set = set(fold.purge_train_indices.tolist())
        val_set = set(fold.val_indices.tolist())
        pv_set = set(fold.purge_val_indices.tolist())
        test_set = set(fold.test_indices.tolist())

        assert len(train_set & pt_set) == 0
        assert len(pt_set & val_set) == 0
        assert len(val_set & pv_set) == 0
        assert len(pv_set & test_set) == 0
        assert len(train_set & test_set) == 0

    # 验证扩展窗口特性：后一折的训练集样本量严格大于前一折
    assert folds[1].n_train > folds[0].n_train
    assert folds[2].n_train > folds[1].n_train


def test_purged_walk_forward_rolling(sample_multi_year_dates: np.ndarray):
    """验证固定滚动窗口 (rolling) 切分器。"""
    splitter = PurgedWalkForwardSplitter(
        start_predict_year=2018,
        end_predict_year=2020,
        train_window="rolling",
        min_train_years=4,
        purge_window=5,
        validation_size_days=250,
        embargo_window=0,
    )
    folds = splitter.split(sample_multi_year_dates)
    assert len(folds) == 3

    # 在 rolling 模式下，训练年份固定为 4 年 (4 * 250 - 5 = 995 样本)
    assert folds[0].n_train == 4 * 250 - 5
    assert folds[1].n_train == 4 * 250 - 5
    assert folds[2].n_train == 4 * 250 - 5


def test_purged_walk_forward_embargo(sample_multi_year_dates: np.ndarray):
    """验证 Embargo 静默观察期天数隔离。"""
    embargo_days = 3
    splitter = PurgedWalkForwardSplitter(
        start_predict_year=2018,
        end_predict_year=2018,
        train_window="expanding",
        min_train_years=4,
        purge_window=5,
        validation_size_days=250,
        embargo_window=embargo_days,
    )
    folds = splitter.split(sample_multi_year_dates)
    assert len(folds) == 1
    f = folds[0]
    assert len(f.embargo_val_indices) == embargo_days
    assert len(f.embargo_test_indices) == embargo_days
    assert np.max(f.embargo_test_indices) < np.min(f.test_indices)


# ------------------------------------------------------------------------------
# 2. 特征工程预处理单元测试
# ------------------------------------------------------------------------------
def test_align_directions():
    """验证因子方向对齐。"""
    mat1 = np.ones((5, 5))
    mat2 = np.ones((5, 5)) * 2.0
    res = align_directions({"f1": mat1, "f2": mat2}, {"f1": 1, "f2": -1})
    np.testing.assert_array_equal(res["f1"], mat1)
    np.testing.assert_array_equal(res["f2"], -mat2)


def test_mad_winsorize():
    """验证截面 MAD 离群点中位数截断。"""
    rng = np.random.default_rng(42)
    # 正常数据服从 N(10, 2)
    row = rng.normal(10.0, 2.0, size=(1, 50))
    # 注入严重离群点
    row[0, 0] = 1000.0
    row[0, 1] = -1000.0

    clipped = mad_winsorize(row, n_mad=5.0)

    # 离群点必须被拉回边界
    assert clipped[0, 0] < 100.0
    assert clipped[0, 1] > -100.0
    # 正常数据基本保持不变
    assert abs(clipped[0, 10] - row[0, 10]) < 1e-9


def test_rank_gauss_transform():
    """验证截面排序高斯化 (Rank Gauss Transform) 数学性质。"""
    rng = np.random.default_rng(42)
    T, N = 10, 100
    mat = rng.uniform(-100.0, 100.0, size=(T, N))
    mat[0, 0] = np.nan  # 测试 NaN 保留

    gauss = rank_gauss_transform(mat)

    assert np.isnan(gauss[0, 0])
    # 检查单调性：对每一行，原值大小排序与变换后大小排序 100% 一致
    for t in range(T):
        valid = ~np.isnan(mat[t, :])
        orig_valid = mat[t, valid]
        gauss_valid = gauss[t, valid]

        orig_order = np.argsort(orig_valid)
        gauss_order = np.argsort(gauss_valid)
        np.testing.assert_array_equal(orig_order, gauss_order)

        # 检查高斯化均值接近 0
        assert abs(np.mean(gauss_valid)) < 1e-12
        # 标准差接近 1 (对于 N=100 均匀分布采样分位数，标准差约 0.95~1.0)
        assert 0.90 <= np.std(gauss_valid) <= 1.05
        # 绝无无穷大
        assert np.all(np.isfinite(gauss_valid))


def test_preprocess_factors_pipeline():
    """验证完整的预处理标准流。"""
    m = np.random.randn(5, 10)
    signs = {"f1": -1}
    pre = preprocess_factors({"f1": m}, factor_signs=signs)
    assert "f1" in pre
    assert pre["f1"].shape == (5, 10)
    assert pre["f1"].flags.f_contiguous


# ------------------------------------------------------------------------------
# 3. 数据集构建与二维面板重构测试
# ------------------------------------------------------------------------------
def test_dataset_builder_and_reconstruct(synthetic_ml_panel):
    """验证 MLDatasetBuilder 平铺切片提取与反向面板重构精度。"""
    panel, factor_matrices, factor_signs = synthetic_ml_panel

    builder = MLDatasetBuilder(
        panel=panel,
        factor_matrices=factor_matrices,
        factor_signs=factor_signs,
        target_label="y_5",
        preprocess=True,
    )

    row_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    ds = builder.build_slice(row_indices)

    assert ds.n_features == 3
    assert ds.n_samples > 0
    assert len(ds.groups) == len(row_indices)

    # 模拟预测
    mock_preds = ds.X[:, 0] * 2.0
    recon = builder.reconstruct_panel(mock_preds, ds.date_indices, ds.symbol_indices)

    assert recon.shape == panel.shape
    assert recon.flags.f_contiguous
    # 校验回填位置的值与 mock_preds 一致
    for i in range(len(mock_preds)):
        t = ds.date_indices[i]
        s = ds.symbol_indices[i]
        assert abs(recon[t, s] - mock_preds[i]) < 1e-9


# ------------------------------------------------------------------------------
# 4. 预测模型族 (Models) 单元测试
# ------------------------------------------------------------------------------
def test_models_fit_and_predict():
    """验证 Equal Weight, Rolling ICIR, Ridge, LightGBM 的训练与预测行为。"""
    rng = np.random.default_rng(42)
    M, K = 120, 4
    X = rng.normal(0.0, 1.0, size=(M, K))
    # 构造 y 与第 0 个特征强正相关，第 1 个特征中等正相关，其余为噪音
    y = 0.8 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(0.0, 0.2, size=M)
    date_indices = np.repeat(np.arange(12), 10)

    # 1. Equal Weight
    ew = create_model("equal_weight").fit(X)
    pred_ew = ew.predict(X)
    assert len(pred_ew) == M
    assert np.all(np.isfinite(pred_ew))

    # 2. Rolling ICIR
    icir = create_model("rolling_icir").fit(X, y, date_indices=date_indices)
    pred_icir = icir.predict(X)
    assert len(pred_icir) == M
    # 第 0 与第 1 个特征的权重必须显著大于第 2 与第 3 个特征
    assert icir.weights_[0] > icir.weights_[2]
    assert icir.weights_[1] > icir.weights_[3]

    # 3. Ridge
    ridge = create_model("ridge").fit(X, y)
    pred_ridge = ridge.predict(X)
    assert len(pred_ridge) == M
    assert ridge.coef_[0] > ridge.coef_[2]

    # 4. LightGBM
    lgb_cfg = LightGBMConfig(n_estimators=50, max_depth=3, min_child_samples=5)
    lgbm = LightGBMModel.from_config(lgb_cfg, feature_names=[f"f{i}" for i in range(K)])
    lgbm.fit(
        X[:100],
        y[:100],
        X_val=X[100:],
        y_val=y[100:],
    )
    pred_lgb = lgbm.predict(X[100:])
    assert len(pred_lgb) == 20
    assert np.all(np.isfinite(pred_lgb))

    imp = lgbm.get_feature_importances()
    assert "f0" in imp
    # 强信号特征的增益应最高
    assert imp["f0"] >= imp["f2"]


# ------------------------------------------------------------------------------
# 5. 端到端 Walk-Forward 流水线与优势验收测试
# ------------------------------------------------------------------------------
def test_walk_forward_pipeline_acceptance(synthetic_ml_panel, tmp_path: Path):
    """P5 核心验收：全流程串通测试、无未来泄漏、LightGBM 表现验证与 Parquet 导出。"""
    panel, factor_matrices, factor_signs = synthetic_ml_panel

    # 配置 2018 到 2020 年的预测
    cfg = MLConfig(
        target_label="y_5",
        walk_forward=WalkForwardConfig(
            start_predict_year=2018,
            end_predict_year=2020,
            train_window="expanding",
            min_train_years=2,
            purge_window=5,
            validation_size_days=60,
        ),
        lightgbm=LightGBMConfig(
            max_depth=3,
            num_leaves=8,
            min_child_samples=10,
            n_estimators=100,
            early_stopping_rounds=15,
            learning_rate=0.05,
        ),
        models=("equal_weight", "rolling_icir", "ridge", "lightgbm"),
    )

    pipeline = WalkForwardPipeline(
        panel=panel,
        config=cfg,
        factor_matrices=factor_matrices,
        factor_signs=factor_signs,
    )

    result = pipeline.run(strict_min_years=False)

    assert isinstance(result.metrics, dict)
    assert "equal_weight" in result.metrics
    assert "rolling_icir" in result.metrics
    assert "ridge" in result.metrics
    assert "lightgbm" in result.metrics

    ew_ic = result.metrics["equal_weight"]["ic_mean"]
    lgb_ic = result.metrics["lightgbm"]["ic_mean"]
    ridge_ic = result.metrics["ridge"]["ic_mean"]

    # 验证模型预测 RankIC 均为正且具备显著有效性
    assert ew_ic > 0.05
    assert ridge_ic > 0.05
    assert lgb_ic > 0.05

    # 验证 LightGBM 在非线性交互数据上的学习能力：RankIC 优于等权基线
    assert lgb_ic >= ew_ic, f"LightGBM IC ({lgb_ic:.4f}) 应优于等权基线 ({ew_ic:.4f})"

    # 验证输出 pred_panel.parquet
    out_parquet = tmp_path / "pred_panel.parquet"
    result.to_parquet(out_parquet)
    assert out_parquet.exists()

    df = pl.read_parquet(out_parquet)
    assert "trading_date" in df.columns
    assert "symbol" in df.columns
    assert "y_5" in df.columns
    assert "pred_equal_weight" in df.columns
    assert "pred_rolling_icir" in df.columns
    assert "pred_ridge" in df.columns
    assert "pred_lightgbm" in df.columns

    # 验证摘要 Markdown 渲染
    md = result.summary_table()
    assert "lightgbm" in md
    assert "equal_weight" in md


def test_calc_rank_ic():
    """验证 RankIC 辅助计算函数的边界与数学正确性。"""
    T, N = 5, 10
    pred = np.zeros((T, N))
    target = np.zeros((T, N))
    mask = np.ones((T, N), dtype=bool)

    # 完美正相关
    for t in range(T):
        pred[t, :] = np.arange(N)
        target[t, :] = np.arange(N)

    stats = _calc_rank_ic(pred, target, mask, min_names=5)
    assert abs(stats["ic_mean"] - 1.0) < 1e-9
    assert abs(stats["ic_win_rate"] - 1.0) < 1e-9
    assert stats["valid_days"] == T


def test_ml_pipeline_enforces_min_purge_from_label_horizon_p0_05(synthetic_ml_panel):
    """验证 P0-05: 当配置 purge_window 小于 target_label 预测周期 h 时，自动提升至 >= h。"""
    panel, factor_matrices, factor_signs = synthetic_ml_panel
    cfg = MLConfig(
        target_label="y_5",
        walk_forward=WalkForwardConfig(
            start_predict_year=2018,
            end_predict_year=2019,
            train_window="expanding",
            min_train_years=2,
            purge_window=1,  # 故意设为 1 < 5
            validation_size_days=60,
        ),
        models=("equal_weight",),
    )
    pipeline = WalkForwardPipeline(
        panel=panel,
        config=cfg,
        factor_matrices=factor_matrices,
        factor_signs=factor_signs,
    )
    res_p1 = pipeline.run(strict_min_years=False)

    cfg_p5 = MLConfig(
        target_label="y_5",
        walk_forward=WalkForwardConfig(
            start_predict_year=2018,
            end_predict_year=2019,
            train_window="expanding",
            min_train_years=2,
            purge_window=5,
            validation_size_days=60,
        ),
        models=("equal_weight",),
    )
    res_p5 = WalkForwardPipeline(
        panel=panel,
        config=cfg_p5,
        factor_matrices=factor_matrices,
        factor_signs=factor_signs,
    ).run(strict_min_years=False)

    assert len(res_p1.fold_metrics) == 2
    assert res_p1.fold_metrics[0]["n_train"] == res_p5.fold_metrics[0]["n_train"]
    assert res_p1.fold_metrics[1]["n_train"] == res_p5.fold_metrics[1]["n_train"]
