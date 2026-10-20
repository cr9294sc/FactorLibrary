"""SQLite 因子注册表 ORM 与因子资产全生命周期管理 (core/registry.py)。

严格遵循白皮书 10.1 节与 10.2 节架构设计：
- Schema 严格包含 factor (元数据), factor_metrics (多周期多切片评价), experiment_run (实验追踪)；
- 维护严格的外键级联约束与 WAL 并发模式；
- 实现因子生命周期流转状态机：candidate -> accepted -> active -> deprecated；
- 记录唯一代数指纹 canon_hash，杜绝同质异构多项式重复入库；
- 支持根据 ICIR、换手率、状态等维度快速过滤与特征组装。
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_STATUSES = {"candidate", "accepted", "active", "deprecated"}
VALID_SOURCES = {"gp", "manual", "seed"}
VALID_MODES = {"cs", "ts"}
VALID_SPLITS = {"train", "valid", "test", "full"}
VALID_RUN_KINDS = {"gp", "ml", "backtest", "eval"}
VALID_RUN_STATUSES = {"running", "completed", "failed"}


@dataclass
class FactorMetricsRecord:
    """因子在特定时段切片与预测窗口下的评估指标明细。"""

    factor_id: int | None = None
    split: str = "train"  # train | valid | test | full
    horizon: int = 5  # 预测窗口天数 (如 1, 3, 5, 10, 20)
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_t: float = 0.0
    ic_win_rate: float = 0.0
    coverage: float = 0.0
    turnover: float = 0.0
    ls_sharpe: float = 0.0
    ls_annual_ret: float = 0.0
    ls_max_dd: float = 0.0
    computed_at: str | None = None

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(f"无效的切片类型: {self.split!r}, 必须在 {VALID_SPLITS} 中")
        if self.computed_at is None:
            self.computed_at = dt.datetime.now(dt.UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorRecord:
    """因子元数据主表记录。"""

    canon_expr: str
    canon_hash: str
    mode: str = "cs"  # cs: 截面, ts: 时序 CTA
    sign: int = 1  # 推荐方向: 1 为正向, -1 为反向
    n_nodes: int = 1
    depth: int = 1
    source: str = "manual"  # gp | manual | seed
    source_run: str | None = None
    data_version: str = "v1"
    status: str = "candidate"  # candidate | accepted | active | deprecated
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metrics: list[FactorMetricsRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"无效的因子状态: {self.status!r}, 必须在 {VALID_STATUSES} 中")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"无效的来源: {self.source!r}, 必须在 {VALID_SOURCES} 中")
        if self.mode not in VALID_MODES:
            raise ValueError(f"无效的模式: {self.mode!r}, 必须在 {VALID_MODES} 中")
        if self.sign not in (-1, 1):
            raise ValueError(f"无效的方向: {self.sign}, 必须为 1 或 -1")
        now_str = dt.datetime.now(dt.UTC).isoformat()
        if self.created_at is None:
            self.created_at = now_str
        if self.updated_at is None:
            self.updated_at = now_str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = [m.to_dict() for m in self.metrics]
        return d


@dataclass
class ExperimentRunRecord:
    """实验运行元数据追踪记录。"""

    run_id: str
    kind: str  # gp | ml | backtest | eval
    config_yaml: str
    data_version: str
    git_sha: str
    n_evaluated: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "running"  # running | completed | failed

    def __post_init__(self) -> None:
        if self.kind not in VALID_RUN_KINDS:
            raise ValueError(f"无效的实验类型: {self.kind!r}, 必须在 {VALID_RUN_KINDS} 中")
        if self.status not in VALID_RUN_STATUSES:
            raise ValueError(f"无效的运行状态: {self.status!r}, 必须在 {VALID_RUN_STATUSES} 中")
        if self.started_at is None:
            self.started_at = dt.datetime.now(dt.UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCHEMA_SQL = """
-- 因子元数据主表
CREATE TABLE IF NOT EXISTS factor (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canon_expr      TEXT NOT NULL,
    canon_hash      TEXT NOT NULL UNIQUE,
    mode            TEXT NOT NULL CHECK (mode IN ('cs', 'ts')),
    sign            INTEGER NOT NULL CHECK (sign IN (-1, 1)),
    n_nodes         INTEGER NOT NULL,
    depth           INTEGER NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('gp', 'manual', 'seed')),
    source_run      TEXT,
    data_version    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'candidate'
                    CHECK (status IN ('candidate', 'accepted', 'active', 'deprecated')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- 因子多时段评估指标明细表
CREATE TABLE IF NOT EXISTS factor_metrics (
    factor_id       INTEGER NOT NULL REFERENCES factor(id) ON DELETE CASCADE,
    split           TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test', 'full')),
    horizon         INTEGER NOT NULL,
    ic_mean         REAL NOT NULL,
    ic_std          REAL NOT NULL,
    ic_ir           REAL NOT NULL,
    ic_t            REAL NOT NULL,
    ic_win_rate     REAL NOT NULL,
    coverage        REAL NOT NULL,
    turnover        REAL NOT NULL,
    ls_sharpe       REAL NOT NULL,
    ls_annual_ret   REAL NOT NULL,
    ls_max_dd       REAL NOT NULL,
    computed_at     TEXT NOT NULL,
    PRIMARY KEY (factor_id, split, horizon)
);

-- 实验运行元数据追踪表
CREATE TABLE IF NOT EXISTS experiment_run (
    run_id          TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    config_yaml     TEXT NOT NULL,
    data_version    TEXT NOT NULL,
    git_sha         TEXT NOT NULL,
    n_evaluated     INTEGER DEFAULT 0,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_factor_status ON factor(status);
CREATE INDEX IF NOT EXISTS idx_factor_hash ON factor(canon_hash);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON factor_metrics(split, horizon);
"""


class FactorRegistry:
    """生产级 SQLite 因子注册表管理器。"""

    def __init__(self, db_path: str | Path = "data/registry/factors.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)

    # =========================================================================
    # 因子元数据与评价指标注册 (CRUD)
    # =========================================================================

    def register_factor(
        self,
        factor: FactorRecord,
        run_id: str | None = None,
        upsert: bool = False,
    ) -> int:
        """原子化插入新因子及其全量评估指标，返回 factor_id。

        Parameters
        ----------
        factor:
            待注册的因子记录
        run_id:
            实验运行唯一标识 (覆盖 factor.source_run)
        upsert:
            若为 True 且哈希已存在，则更新已有因子的元数据并补充指标；
            若为 False 且哈希存在，抛出 ValueError。
        """
        source_run = run_id or factor.source_run
        now_str = dt.datetime.now(dt.UTC).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM factor WHERE canon_hash = ?", (factor.canon_hash,))
            row = cur.fetchone()

            if row is not None:
                if not upsert:
                    raise ValueError(
                        f"因子指纹 '{factor.canon_hash}' 已存在于库中 (id={row['id']})"
                    )
                factor_id = int(row["id"])
                cur.execute(
                    """
                    UPDATE factor
                    SET canon_expr = ?, mode = ?, sign = ?, n_nodes = ?, depth = ?,
                        source = ?, source_run = ?, data_version = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        factor.canon_expr,
                        factor.mode,
                        factor.sign,
                        factor.n_nodes,
                        factor.depth,
                        factor.source,
                        source_run,
                        factor.data_version,
                        factor.status,
                        now_str,
                        factor_id,
                    ),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO factor (
                        canon_expr, canon_hash, mode, sign, n_nodes, depth,
                        source, source_run, data_version, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        factor.canon_expr,
                        factor.canon_hash,
                        factor.mode,
                        factor.sign,
                        factor.n_nodes,
                        factor.depth,
                        factor.source,
                        source_run,
                        factor.data_version,
                        factor.status,
                        factor.created_at or now_str,
                        now_str,
                    ),
                )
                factor_id = int(cur.lastrowid)

            # 插入指标明细
            for m in factor.metrics:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO factor_metrics (
                        factor_id, split, horizon, ic_mean, ic_std, ic_ir, ic_t,
                        ic_win_rate, coverage, turnover, ls_sharpe, ls_annual_ret,
                        ls_max_dd, computed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        factor_id,
                        m.split,
                        m.horizon,
                        m.ic_mean,
                        m.ic_std,
                        m.ic_ir,
                        m.ic_t,
                        m.ic_win_rate,
                        m.coverage,
                        m.turnover,
                        m.ls_sharpe,
                        m.ls_annual_ret,
                        m.ls_max_dd,
                        m.computed_at or now_str,
                    ),
                )

        return factor_id

    def add_metrics(
        self,
        factor_id: int,
        metrics: FactorMetricsRecord | Sequence[FactorMetricsRecord],
    ) -> None:
        """为已有因子补充或刷新评估指标。"""
        if isinstance(metrics, FactorMetricsRecord):
            metrics_list = [metrics]
        else:
            metrics_list = list(metrics)

        now_str = dt.datetime.now(dt.UTC).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            for m in metrics_list:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO factor_metrics (
                        factor_id, split, horizon, ic_mean, ic_std, ic_ir, ic_t,
                        ic_win_rate, coverage, turnover, ls_sharpe, ls_annual_ret,
                        ls_max_dd, computed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        factor_id,
                        m.split,
                        m.horizon,
                        m.ic_mean,
                        m.ic_std,
                        m.ic_ir,
                        m.ic_t,
                        m.ic_win_rate,
                        m.coverage,
                        m.turnover,
                        m.ls_sharpe,
                        m.ls_annual_ret,
                        m.ls_max_dd,
                        m.computed_at or now_str,
                    ),
                )
            conn.execute(
                "UPDATE factor SET updated_at = ? WHERE id = ?",
                (now_str, factor_id),
            )

    def get_factor(self, factor_id: int, include_metrics: bool = True) -> FactorRecord | None:
        """按主键 ID 查询因子详情。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM factor WHERE id = ?", (factor_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_factor(row, conn if include_metrics else None)

    def get_factor_by_hash(
        self, canon_hash: str, include_metrics: bool = True
    ) -> FactorRecord | None:
        """按代数指纹 canon_hash 查询因子详情。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM factor WHERE canon_hash = ?", (canon_hash,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_factor(row, conn if include_metrics else None)

    def list_factors(
        self,
        status: str | None = None,
        mode: str | None = None,
        source: str | None = None,
        min_icir: float | None = None,
        split: str = "train",
        horizon: int = 5,
        limit: int | None = None,
    ) -> list[FactorRecord]:
        """按多重过滤条件检索因子列表。"""
        conditions = []
        params: list[Any] = []

        if status is not None:
            conditions.append("f.status = ?")
            params.append(status)
        if mode is not None:
            conditions.append("f.mode = ?")
            params.append(mode)
        if source is not None:
            conditions.append("f.source = ?")
            params.append(source)

        query = "SELECT DISTINCT f.* FROM factor f"

        if min_icir is not None:
            query += " JOIN factor_metrics m ON f.id = m.factor_id"
            conditions.append("m.split = ?")
            params.append(split)
            conditions.append("m.horizon = ?")
            params.append(horizon)
            conditions.append("ABS(m.ic_ir) >= ?")
            params.append(min_icir)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY f.id ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_factor(r, conn) for r in rows]

    def get_accepted_factors(self) -> list[FactorRecord]:
        """获取准入合格的因子列表 (status = 'accepted')。"""
        return self.list_factors(status="accepted")

    def get_active_factors(self) -> list[FactorRecord]:
        """获取当前生产回测与 ML 特征池生效的因子列表 (status = 'active')。"""
        return self.list_factors(status="active")

    def update_factor_status(
        self,
        factor_id: int,
        status: str,
        sign: int | None = None,
    ) -> None:
        """更新因子的生命周期状态与方向。"""
        if status not in VALID_STATUSES:
            raise ValueError(f"无效的状态: {status!r}, 必须在 {VALID_STATUSES} 中")
        now_str = dt.datetime.now(dt.UTC).isoformat()

        with self._get_connection() as conn:
            if sign is not None:
                if sign not in (-1, 1):
                    raise ValueError(f"无效的方向: {sign}")
                conn.execute(
                    "UPDATE factor SET status = ?, sign = ?, updated_at = ? WHERE id = ?",
                    (status, sign, now_str, factor_id),
                )
            else:
                conn.execute(
                    "UPDATE factor SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now_str, factor_id),
                )

    def delete_factor(self, factor_id: int) -> bool:
        """物理删除因子（级联删除 metrics 明细）。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM factor WHERE id = ?", (factor_id,))
            return cur.rowcount > 0

    # =========================================================================
    # 实验运行元数据追踪 (Experiment Run)
    # =========================================================================

    def record_run(self, run: ExperimentRunRecord) -> None:
        """记录新的实验运行上下文。"""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_run (
                    run_id, kind, config_yaml, data_version, git_sha,
                    n_evaluated, started_at, finished_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.kind,
                    run.config_yaml,
                    run.data_version,
                    run.git_sha,
                    run.n_evaluated,
                    run.started_at,
                    run.finished_at,
                    run.status,
                ),
            )

    def finish_run(
        self,
        run_id: str,
        status: str = "completed",
        n_evaluated: int | None = None,
    ) -> None:
        """更新实验运行状态与结束时间。"""
        now_str = dt.datetime.now(dt.UTC).isoformat()
        with self._get_connection() as conn:
            if n_evaluated is not None:
                conn.execute(
                    """
                    UPDATE experiment_run
                    SET status = ?, finished_at = ?, n_evaluated = ?
                    WHERE run_id = ?
                    """,
                    (status, now_str, n_evaluated, run_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE experiment_run
                    SET status = ?, finished_at = ?
                    WHERE run_id = ?
                    """,
                    (status, now_str, run_id),
                )

    def get_run(self, run_id: str) -> ExperimentRunRecord | None:
        """获取指定实验运行信息。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM experiment_run WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return ExperimentRunRecord(
                run_id=row["run_id"],
                kind=row["kind"],
                config_yaml=row["config_yaml"],
                data_version=row["data_version"],
                git_sha=row["git_sha"],
                n_evaluated=row["n_evaluated"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                status=row["status"],
            )

    def list_runs(
        self, kind: str | None = None, status: str | None = None
    ) -> list[ExperimentRunRecord]:
        """查询实验运行列表。"""
        conditions = []
        params: list[Any] = []
        if kind is not None:
            conditions.append("kind = ?")
            params.append(kind)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        query = "SELECT * FROM experiment_run"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC"

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            return [
                ExperimentRunRecord(
                    run_id=r["run_id"],
                    kind=r["kind"],
                    config_yaml=r["config_yaml"],
                    data_version=r["data_version"],
                    git_sha=r["git_sha"],
                    n_evaluated=r["n_evaluated"],
                    started_at=r["started_at"],
                    finished_at=r["finished_at"],
                    status=r["status"],
                )
                for r in cur.fetchall()
            ]

    # =========================================================================
    # 辅助工具函数
    # =========================================================================

    def _row_to_factor(
        self,
        row: sqlite3.Row,
        conn: sqlite3.Connection | None = None,
    ) -> FactorRecord:
        metrics: list[FactorMetricsRecord] = []
        factor_id = int(row["id"])
        if conn is not None:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM factor_metrics WHERE factor_id = ? ORDER BY split, horizon",
                (factor_id,),
            )
            for m_row in cur.fetchall():
                metrics.append(
                    FactorMetricsRecord(
                        factor_id=factor_id,
                        split=m_row["split"],
                        horizon=int(m_row["horizon"]),
                        ic_mean=float(m_row["ic_mean"]),
                        ic_std=float(m_row["ic_std"]),
                        ic_ir=float(m_row["ic_ir"]),
                        ic_t=float(m_row["ic_t"]),
                        ic_win_rate=float(m_row["ic_win_rate"]),
                        coverage=float(m_row["coverage"]),
                        turnover=float(m_row["turnover"]),
                        ls_sharpe=float(m_row["ls_sharpe"]),
                        ls_annual_ret=float(m_row["ls_annual_ret"]),
                        ls_max_dd=float(m_row["ls_max_dd"]),
                        computed_at=m_row["computed_at"],
                    )
                )

        return FactorRecord(
            id=factor_id,
            canon_expr=row["canon_expr"],
            canon_hash=row["canon_hash"],
            mode=row["mode"],
            sign=int(row["sign"]),
            n_nodes=int(row["n_nodes"]),
            depth=int(row["depth"]),
            source=row["source"],
            source_run=row["source_run"],
            data_version=row["data_version"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metrics=metrics,
        )
