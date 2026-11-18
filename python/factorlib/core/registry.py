"""SQLite 因子注册表 ORM 与混合投研控制面 (core/registry.py)。

严格遵循白皮书 10.1/10.2 节与混合投研架构设计 §3.1 规范：
- 强制开启 SQLite WAL 并发模式 (`journal_mode=WAL`, `busy_timeout=5000`, `foreign_keys=ON`)；
- 多市场域严格隔离 (`market`: `futures` | `stocks`)，`factor` 唯一键为 `(market, canon_hash)`；
- Schema 包含：
  - `factor` / `factor_metrics` / `factor_version`（支持 `expression` / `code` / `dl_alpha` 契约）；
  - `feature_set`（命名特征池：`1d` 基础字段 + `1m` 扩展包 + 算子/窗口白名单 + 辅助因子 + 后处理 Preset）；
  - `research_recipe`（不可变实验配方快照，深拷贝冻结 `feature_set` 与超参，生成内容哈希 `recipe_id`）；
  - `experiment_run`（实验追踪、OS `pid`、进度快照与日志尾行心跳）；
- 兼容旧版单市场 SQLite 文件的幂等原子迁移。
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import shutil
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VALID_MARKETS = {"futures", "stocks"}
VALID_STATUSES = {"candidate", "accepted", "active", "deprecated"}
VALID_SOURCES = {"gp", "manual", "seed", "dl_alpha"}
VALID_MODES = {"cs", "ts"}
VALID_SPLITS = {"train", "valid", "test", "full"}
VALID_RUN_KINDS = {"gp", "ml", "dl", "backtest", "eval"}
VALID_RUN_STATUSES = {"queued", "running", "completed", "failed", "cancelled"}
VALID_FACTOR_TYPES = {"expression", "code", "dl_alpha"}
VALID_RECIPE_KINDS = {"gp", "dl", "ml", "backtest"}

SCHEMA_VERSION = 2


def _normalize_market(market: str) -> str:
    m = str(market).strip().lower()
    if m not in VALID_MARKETS:
        raise ValueError(f"无效的市场域: {market!r}, 必须在 {sorted(VALID_MARKETS)} 中")
    return m


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


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
    top_turnover: float = 0.0
    rank_autocorr_turnover: float = 0.0
    computed_at: str | None = None

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(f"无效的切片类型: {self.split!r}, 必须在 {VALID_SPLITS} 中")
        if self.top_turnover == 0.0 and self.turnover != 0.0:
            self.top_turnover = self.turnover
        if self.turnover == 0.0 and self.top_turnover != 0.0:
            self.turnover = self.top_turnover
        if self.computed_at is None:
            self.computed_at = dt.datetime.now(dt.UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorVersionRecord:
    """因子不可变历史版本锁定记录 (对标 FactorLab Stage 4/5 Version Snapshot)。"""

    factor_id: int
    version: int
    canon_expr: str
    canon_hash: str
    definition_json: str = "{}"
    materialization_path: str | None = None
    probe_json: str | None = None
    eval_summary_json: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = dt.datetime.now(dt.UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorRecord:
    """因子元数据主表记录（支持多市场隔离与 `dl_alpha` 因子类型）。"""

    canon_expr: str
    canon_hash: str
    mode: str = "cs"  # cs: 截面, ts: 时序 CTA
    sign: int = 1  # 推荐方向: 1 为正向, -1 为反向
    n_nodes: int = 1
    depth: int = 1
    source: str = "manual"  # gp | manual | seed | dl_alpha
    source_run: str | None = None
    data_version: str = "v1"
    status: str = "candidate"  # candidate | accepted | active | deprecated
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metrics: list[FactorMetricsRecord] = field(default_factory=list)
    name: str | None = None
    factor_type: str = "expression"  # expression | code | dl_alpha
    definition_json: str | None = None
    draft_revision: int = 1
    materialization_path: str | None = None
    market: str = "futures"
    recipe_id: str | None = None

    def __post_init__(self) -> None:
        self.market = _normalize_market(self.market)
        if self.status not in VALID_STATUSES:
            raise ValueError(f"无效的因子状态: {self.status!r}, 必须在 {VALID_STATUSES} 中")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"无效的来源: {self.source!r}, 必须在 {VALID_SOURCES} 中")
        if self.mode not in VALID_MODES:
            raise ValueError(f"无效的模式: {self.mode!r}, 必须在 {VALID_MODES} 中")
        if self.sign not in (-1, 1):
            raise ValueError(f"无效的方向: {self.sign}, 必须为 1 或 -1")
        if self.factor_type not in VALID_FACTOR_TYPES:
            raise ValueError(
                f"无效的因子类型: {self.factor_type!r}, 必须在 {VALID_FACTOR_TYPES} 中"
            )
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
class FeatureSetRecord:
    """命名特征池记录（Web 编排与 CLI 冻结的数据源头）。"""

    name: str
    market: str = "futures"
    description: str = ""
    base_fields: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    operators: list[str] | None = None
    windows: list[int] | None = None
    constants: list[float] | None = None
    aux_factor_ids: list[int] = field(default_factory=list)
    default_postprocess: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        self.market = _normalize_market(self.market)
        if not str(self.name).strip():
            raise ValueError("FeatureSetRecord.name 不能为空")
        self.name = str(self.name).strip()
        self.base_fields = [str(f) for f in self.base_fields]
        self.extensions = [str(e) for e in self.extensions]
        if self.operators is not None:
            self.operators = [str(op) for op in self.operators]
        if self.windows is not None:
            self.windows = [int(w) for w in self.windows]
        if self.constants is not None:
            self.constants = [float(c) for c in self.constants]
        self.aux_factor_ids = [int(fid) for fid in self.aux_factor_ids]
        now_str = dt.datetime.now(dt.UTC).isoformat()
        if self.created_at is None:
            self.created_at = now_str
        if self.updated_at is None:
            self.updated_at = now_str

    def to_snapshot_dict(self) -> dict[str, Any]:
        """导出用于 Recipe 深拷贝冻结的特征池内容字典。"""
        return copy.deepcopy(
            {
                "id": self.id,
                "name": self.name,
                "market": self.market,
                "version": self.version,
                "description": self.description,
                "base_fields": list(self.base_fields),
                "extensions": list(self.extensions),
                "operators": list(self.operators) if self.operators is not None else None,
                "windows": list(self.windows) if self.windows is not None else None,
                "constants": list(self.constants) if self.constants is not None else None,
                "aux_factor_ids": list(self.aux_factor_ids),
                "default_postprocess": copy.deepcopy(self.default_postprocess),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchRecipeRecord:
    """不可变实验配方快照记录（锁定 FeatureSet、标签、切分、超参、指纹与种子）。"""

    market: str
    kind: str  # gp | dl | ml | backtest
    feature_set: dict[str, Any] = field(default_factory=dict)
    target_label: str = "y_5"
    splits: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    extensions: list[str] = field(default_factory=list)
    data_version: str = "v1"
    seed: int = 42
    name: str = ""
    feature_set_id: int | None = None
    recipe_id: str | None = None
    content_hash: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        self.market = _normalize_market(self.market)
        self.kind = str(self.kind).strip().lower()
        if self.kind not in VALID_RECIPE_KINDS:
            raise ValueError(
                f"无效的配方类型 kind={self.kind!r}, 必须在 {sorted(VALID_RECIPE_KINDS)} 中"
            )
        if not str(self.target_label).strip():
            raise ValueError("ResearchRecipeRecord.target_label 不能为空")

        # 深拷贝冻结所有可变集合，彻底隔离后续外部或 FeatureSet 修改
        self.feature_set = copy.deepcopy(self.feature_set)
        self.splits = copy.deepcopy(self.splits)
        self.params = copy.deepcopy(self.params)
        self.extensions = list(self.extensions)
        if not self.extensions and isinstance(self.feature_set.get("extensions"), list):
            self.extensions = [str(x) for x in self.feature_set["extensions"]]
        self.seed = int(self.seed)

        if not self.content_hash:
            canonical_payload = {
                "market": self.market,
                "kind": self.kind,
                "feature_set": self.feature_set,
                "target_label": self.target_label,
                "splits": self.splits,
                "params": self.params,
                "extensions": self.extensions,
                "data_version": self.data_version,
                "seed": self.seed,
            }
            self.content_hash = hashlib.sha256(
                _canonical_json(canonical_payload).encode("utf-8")
            ).hexdigest()
        if not self.recipe_id:
            self.recipe_id = f"rcp_{self.content_hash[:12]}"
        if self.created_at is None:
            self.created_at = dt.datetime.now(dt.UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))


@dataclass
class ExperimentRunRecord:
    """实验运行元数据与心跳追踪记录。"""

    run_id: str
    kind: str  # gp | ml | dl | backtest | eval
    config_yaml: str
    data_version: str
    git_sha: str
    n_evaluated: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "running"  # queued | running | completed | failed | cancelled
    market: str = "futures"
    recipe_id: str | None = None
    pid: int | None = None
    progress_json: str | None = None
    heartbeat_at: str | None = None
    run_dir: str | None = None
    log_tail: str | None = None

    def __post_init__(self) -> None:
        self.market = _normalize_market(self.market)
        if self.kind not in VALID_RUN_KINDS:
            raise ValueError(f"无效的实验类型: {self.kind!r}, 必须在 {VALID_RUN_KINDS} 中")
        if self.status not in VALID_RUN_STATUSES:
            raise ValueError(f"无效的运行状态: {self.status!r}, 必须在 {VALID_RUN_STATUSES} 中")
        if self.started_at is None:
            self.started_at = dt.datetime.now(dt.UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCHEMA_SQL = """
-- 因子元数据主表（支持多市场隔离与 dl_alpha）
CREATE TABLE IF NOT EXISTS factor (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    market                  TEXT NOT NULL DEFAULT 'futures',
    canon_expr              TEXT NOT NULL,
    canon_hash              TEXT NOT NULL,
    mode                    TEXT NOT NULL CHECK (mode IN ('cs', 'ts')),
    sign                    INTEGER NOT NULL CHECK (sign IN (-1, 1)),
    n_nodes                 INTEGER NOT NULL,
    depth                   INTEGER NOT NULL,
    source                  TEXT NOT NULL CHECK (source IN ('gp', 'manual', 'seed', 'dl_alpha')),
    source_run              TEXT,
    recipe_id               TEXT,
    data_version            TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'candidate'
                            CHECK (status IN ('candidate', 'accepted', 'active', 'deprecated')),
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    name                    TEXT,
    factor_type             TEXT NOT NULL DEFAULT 'expression'
                            CHECK (factor_type IN ('expression', 'code', 'dl_alpha')),
    definition_json         TEXT,
    draft_revision          INTEGER NOT NULL DEFAULT 1,
    materialization_path    TEXT,
    UNIQUE(market, canon_hash)
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

-- 因子不可变历史版本锁定表
CREATE TABLE IF NOT EXISTS factor_version (
    factor_id               INTEGER NOT NULL REFERENCES factor(id) ON DELETE CASCADE,
    version                 INTEGER NOT NULL,
    canon_expr              TEXT NOT NULL,
    canon_hash              TEXT NOT NULL,
    definition_json         TEXT NOT NULL,
    materialization_path    TEXT,
    probe_json              TEXT,
    eval_summary_json       TEXT,
    created_at              TEXT NOT NULL,
    PRIMARY KEY (factor_id, version)
);

-- 命名特征池表
CREATE TABLE IF NOT EXISTS feature_set (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    market                  TEXT NOT NULL DEFAULT 'futures',
    name                    TEXT NOT NULL,
    description             TEXT NOT NULL DEFAULT '',
    base_fields_json        TEXT NOT NULL DEFAULT '[]',
    extensions_json         TEXT NOT NULL DEFAULT '[]',
    operators_json          TEXT,
    windows_json            TEXT,
    constants_json          TEXT,
    aux_factor_ids_json     TEXT NOT NULL DEFAULT '[]',
    default_postprocess_json TEXT NOT NULL DEFAULT '[]',
    version                 INTEGER NOT NULL DEFAULT 1,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    UNIQUE(market, name)
);

-- 不可变实验配方快照表
CREATE TABLE IF NOT EXISTS research_recipe (
    recipe_id               TEXT PRIMARY KEY,
    content_hash            TEXT NOT NULL,
    market                  TEXT NOT NULL DEFAULT 'futures',
    kind                    TEXT NOT NULL CHECK (kind IN ('gp', 'dl', 'ml', 'backtest')),
    name                    TEXT NOT NULL DEFAULT '',
    feature_set_id          INTEGER,
    feature_set_json        TEXT NOT NULL,
    target_label            TEXT NOT NULL DEFAULT 'y_5',
    splits_json             TEXT NOT NULL DEFAULT '{}',
    params_json             TEXT NOT NULL DEFAULT '{}',
    extensions_json         TEXT NOT NULL DEFAULT '[]',
    data_version            TEXT NOT NULL DEFAULT 'v1',
    seed                    INTEGER NOT NULL DEFAULT 42,
    created_at              TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS trg_research_recipe_immutable_update
BEFORE UPDATE ON research_recipe
BEGIN
    SELECT RAISE(ABORT, 'research_recipe is immutable');
END;

-- 实验运行元数据与心跳追踪表
CREATE TABLE IF NOT EXISTS experiment_run (
    run_id          TEXT PRIMARY KEY,
    market          TEXT NOT NULL DEFAULT 'futures',
    kind            TEXT NOT NULL,
    recipe_id       TEXT,
    config_yaml     TEXT NOT NULL,
    data_version    TEXT NOT NULL,
    git_sha         TEXT NOT NULL,
    n_evaluated     INTEGER DEFAULT 0,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    pid             INTEGER,
    progress_json   TEXT,
    heartbeat_at    TEXT,
    run_dir         TEXT,
    log_tail        TEXT
);

CREATE INDEX IF NOT EXISTS idx_factor_status ON factor(status);
CREATE INDEX IF NOT EXISTS idx_factor_hash ON factor(canon_hash);
CREATE INDEX IF NOT EXISTS idx_factor_market_status ON factor(market, status);
CREATE INDEX IF NOT EXISTS idx_factor_market_hash ON factor(market, canon_hash);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON factor_metrics(split, horizon);
CREATE INDEX IF NOT EXISTS idx_feature_set_market ON feature_set(market);
CREATE INDEX IF NOT EXISTS idx_recipe_market_kind ON research_recipe(market, kind);
CREATE INDEX IF NOT EXISTS idx_run_market_status ON experiment_run(market, status);
"""


class FactorRegistry:
    """生产级 SQLite 因子注册表与控制面管理器（强制 WAL）。"""

    def __init__(self, db_path: str | Path = "data/registry/factors.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def journal_mode(self) -> str:
        """返回当前 SQLite 日志模式（正常应为 `'wal'`）。"""
        with self._get_connection() as conn:
            row = conn.execute("PRAGMA journal_mode;").fetchone()
            return str(row[0]).lower() if row is not None else ""

    def _backup_before_migration(self) -> None:
        if self.db_path.exists() and self.db_path.stat().st_size > 0:
            ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            bak_path = Path(f"{self.db_path}.bak.{ts}")
            shutil.copy2(self.db_path, bak_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='factor'")
            has_factor_table = cur.fetchone() is not None

            if has_factor_table:
                cur.execute("PRAGMA table_info(factor)")
                factor_cols = {row["name"] for row in cur.fetchall()}
                cur.execute("PRAGMA table_info(experiment_run)")
                run_cols = {row["name"] for row in cur.fetchall()}

                needs_factor_rebuild = "market" not in factor_cols
                needs_run_rebuild = "market" not in run_cols

                if needs_factor_rebuild or needs_run_rebuild:
                    self._backup_before_migration()

                # 先确保缺失列（供非重建迁移）
                if not needs_factor_rebuild:
                    migrations = [
                        ("name", "ALTER TABLE factor ADD COLUMN name TEXT"),
                        (
                            "factor_type",
                            "ALTER TABLE factor ADD COLUMN factor_type TEXT NOT NULL DEFAULT 'expression'",
                        ),
                        ("definition_json", "ALTER TABLE factor ADD COLUMN definition_json TEXT"),
                        (
                            "draft_revision",
                            "ALTER TABLE factor ADD COLUMN draft_revision INTEGER NOT NULL DEFAULT 1",
                        ),
                        (
                            "materialization_path",
                            "ALTER TABLE factor ADD COLUMN materialization_path TEXT",
                        ),
                        ("recipe_id", "ALTER TABLE factor ADD COLUMN recipe_id TEXT"),
                    ]
                    for col_name, ddl in migrations:
                        if col_name not in factor_cols:
                            conn.execute(ddl)
                else:
                    # 从旧版单市场 schema 原子重建 factor 表：补充 market + dl_alpha + UNIQUE(market, canon_hash)
                    conn.execute("PRAGMA foreign_keys = OFF;")
                    try:
                        conn.execute("BEGIN IMMEDIATE;")
                        conn.execute(
                            """
                            CREATE TABLE factor_new (
                                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                                market                  TEXT NOT NULL DEFAULT 'futures',
                                canon_expr              TEXT NOT NULL,
                                canon_hash              TEXT NOT NULL,
                                mode                    TEXT NOT NULL CHECK (mode IN ('cs', 'ts')),
                                sign                    INTEGER NOT NULL CHECK (sign IN (-1, 1)),
                                n_nodes                 INTEGER NOT NULL,
                                depth                   INTEGER NOT NULL,
                                source                  TEXT NOT NULL CHECK (source IN ('gp', 'manual', 'seed', 'dl_alpha')),
                                source_run              TEXT,
                                recipe_id               TEXT,
                                data_version            TEXT NOT NULL,
                                status                  TEXT NOT NULL DEFAULT 'candidate'
                                                        CHECK (status IN ('candidate', 'accepted', 'active', 'deprecated')),
                                created_at              TEXT NOT NULL,
                                updated_at              TEXT NOT NULL,
                                name                    TEXT,
                                factor_type             TEXT NOT NULL DEFAULT 'expression'
                                                        CHECK (factor_type IN ('expression', 'code', 'dl_alpha')),
                                definition_json         TEXT,
                                draft_revision          INTEGER NOT NULL DEFAULT 1,
                                materialization_path    TEXT,
                                UNIQUE(market, canon_hash)
                            );
                            """
                        )
                        name_expr = "name" if "name" in factor_cols else "NULL"
                        ftype_expr = (
                            "COALESCE(factor_type, 'expression')"
                            if "factor_type" in factor_cols
                            else "'expression'"
                        )
                        def_expr = "definition_json" if "definition_json" in factor_cols else "NULL"
                        rev_expr = (
                            "COALESCE(draft_revision, 1)"
                            if "draft_revision" in factor_cols
                            else "1"
                        )
                        mat_expr = (
                            "materialization_path"
                            if "materialization_path" in factor_cols
                            else "NULL"
                        )
                        conn.execute(
                            f"""
                            INSERT INTO factor_new (
                                id, market, canon_expr, canon_hash, mode, sign, n_nodes, depth,
                                source, source_run, recipe_id, data_version, status, created_at, updated_at,
                                name, factor_type, definition_json, draft_revision, materialization_path
                            )
                            SELECT
                                id, 'futures', canon_expr, canon_hash, mode, sign, n_nodes, depth,
                                source, source_run, NULL, data_version, status, created_at, updated_at,
                                {name_expr}, {ftype_expr}, {def_expr}, {rev_expr}, {mat_expr}
                            FROM factor;
                            """
                        )
                        conn.execute("DROP TABLE factor;")
                        conn.execute("ALTER TABLE factor_new RENAME TO factor;")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                    finally:
                        conn.execute("PRAGMA foreign_keys = ON;")

                if needs_run_rebuild:
                    conn.execute("BEGIN IMMEDIATE;")
                    try:
                        conn.execute(
                            """
                            CREATE TABLE experiment_run_new (
                                run_id          TEXT PRIMARY KEY,
                                market          TEXT NOT NULL DEFAULT 'futures',
                                kind            TEXT NOT NULL,
                                recipe_id       TEXT,
                                config_yaml     TEXT NOT NULL,
                                data_version    TEXT NOT NULL,
                                git_sha         TEXT NOT NULL,
                                n_evaluated     INTEGER DEFAULT 0,
                                started_at      TEXT NOT NULL,
                                finished_at     TEXT,
                                status          TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
                                pid             INTEGER,
                                progress_json   TEXT,
                                heartbeat_at    TEXT,
                                run_dir         TEXT,
                                log_tail        TEXT
                            );
                            """
                        )
                        conn.execute(
                            """
                            INSERT INTO experiment_run_new (
                                run_id, market, kind, recipe_id, config_yaml, data_version,
                                git_sha, n_evaluated, started_at, finished_at, status
                            )
                            SELECT
                                run_id, 'futures', kind, NULL, config_yaml, data_version,
                                git_sha, n_evaluated, started_at, finished_at, status
                            FROM experiment_run;
                            """
                        )
                        conn.execute("DROP TABLE experiment_run;")
                        conn.execute("ALTER TABLE experiment_run_new RENAME TO experiment_run;")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise

            conn.executescript(SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")

    # =========================================================================
    # 因子元数据与评价指标注册 (CRUD)
    # =========================================================================

    def register_factor(
        self,
        factor: FactorRecord,
        run_id: str | None = None,
        upsert: bool = False,
    ) -> int:
        """原子化插入新因子及其全量评估指标，返回 factor_id（按 `(market, canon_hash)` 隔离）。"""
        source_run = run_id or factor.source_run
        now_str = dt.datetime.now(dt.UTC).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, draft_revision FROM factor WHERE market = ? AND canon_hash = ?",
                (factor.market, factor.canon_hash),
            )
            row = cur.fetchone()

            if row is not None:
                if not upsert:
                    raise ValueError(
                        f"因子指纹 '{factor.canon_hash}' (market={factor.market}) 已存在于库中 (id={row['id']})"
                    )
                factor_id = int(row["id"])
                next_rev = int(row["draft_revision"] or 1) + 1
                cur.execute(
                    """
                    UPDATE factor
                    SET canon_expr = ?, mode = ?, sign = ?, n_nodes = ?, depth = ?,
                        source = ?, source_run = ?, recipe_id = COALESCE(?, recipe_id),
                        data_version = ?, status = ?, updated_at = ?,
                        name = COALESCE(?, name), factor_type = ?,
                        definition_json = COALESCE(?, definition_json),
                        draft_revision = ?,
                        materialization_path = COALESCE(?, materialization_path)
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
                        factor.recipe_id,
                        factor.data_version,
                        factor.status,
                        now_str,
                        factor.name,
                        factor.factor_type,
                        factor.definition_json,
                        next_rev,
                        factor.materialization_path,
                        factor_id,
                    ),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO factor (
                        market, canon_expr, canon_hash, mode, sign, n_nodes, depth,
                        source, source_run, recipe_id, data_version, status, created_at, updated_at,
                        name, factor_type, definition_json, draft_revision, materialization_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        factor.market,
                        factor.canon_expr,
                        factor.canon_hash,
                        factor.mode,
                        factor.sign,
                        factor.n_nodes,
                        factor.depth,
                        factor.source,
                        source_run,
                        factor.recipe_id,
                        factor.data_version,
                        factor.status,
                        factor.created_at or now_str,
                        now_str,
                        factor.name,
                        factor.factor_type,
                        factor.definition_json,
                        factor.draft_revision,
                        factor.materialization_path,
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

    def register_dl_alpha(
        self,
        *,
        name: str,
        market: str,
        matrix: np.ndarray,
        dates: Sequence[Any],
        symbols: Sequence[str],
        recipe_id: str,
        run_id: str | None = None,
        data_version: str = "v1",
        sign: int = 1,
        mode: str = "cs",
        status: str = "candidate",
        postprocess: list[dict[str, Any]] | None = None,
        fold_diagnostics: list[dict[str, Any]] | None = None,
        metrics: Sequence[FactorMetricsRecord] = (),
        model_meta: dict[str, Any] | None = None,
        mat_dir: str | Path | None = None,
    ) -> int:
        """`dl_alpha` 统一物化与入库契约：登记 OOS 矩阵、`.npz` 落盘并锁定初始版本。"""
        norm_market = _normalize_market(market)
        canon_expr = f"dl_alpha({recipe_id}:{name})"
        hash_payload = {
            "kind": "dl_alpha",
            "market": norm_market,
            "recipe_id": recipe_id,
            "name": name,
        }
        canon_hash = hashlib.sha256(_canonical_json(hash_payload).encode("utf-8")).hexdigest()[:16]
        def_payload = {
            "name": name,
            "factor_type": "dl_alpha",
            "market": norm_market,
            "recipe_id": recipe_id,
            "postprocess": postprocess or [],
            "model_meta": model_meta or {},
            "fold_diagnostics": fold_diagnostics or [],
        }

        rec = FactorRecord(
            market=norm_market,
            canon_expr=canon_expr,
            canon_hash=canon_hash,
            mode=mode,
            sign=sign,
            n_nodes=1,
            depth=1,
            source="dl_alpha",
            source_run=run_id,
            recipe_id=recipe_id,
            data_version=data_version,
            status=status,
            name=name,
            factor_type="dl_alpha",
            definition_json=json.dumps(def_payload, ensure_ascii=False, sort_keys=True),
            metrics=list(metrics),
        )
        fid = self.register_factor(rec, run_id=run_id, upsert=True)
        self.materialize_factor(fid, matrix, dates, symbols, mat_dir=mat_dir)
        self.lock_factor_version(
            fid,
            eval_summary={
                "recipe_id": recipe_id,
                "fold_diagnostics": fold_diagnostics or [],
                "model_meta": model_meta or {},
            },
        )
        return fid

    # =========================================================================
    # 草稿管理、因子物化与不可变版本锁定 (Stage 2/4/5)
    # =========================================================================

    def save_draft(
        self,
        definition: Any,
        probe_report: dict[str, Any] | None = None,
        mode: str = "cs",
        sign: int = 1,
        source: str = "manual",
        data_version: str = "v1",
        market: str | None = None,
    ) -> int:
        """创建或更新因子草稿记录 (Stage 2: factor_create / factor_update_draft)。"""
        from factorlib.core.probe import FactorDefinition, probe_factor

        defn = (
            definition
            if isinstance(definition, FactorDefinition)
            else FactorDefinition.from_dict(definition)
            if isinstance(definition, dict)
            else FactorDefinition(name="draft_factor", expression=str(definition))
        )
        rep = probe_report if probe_report is not None else probe_factor(defn)
        if rep["status"] != "ok":
            raise ValueError(f"草稿静态探针校验失败: {rep['errors']}")

        eff_market = _normalize_market(market or getattr(defn, "market", "futures"))
        rec = FactorRecord(
            market=eff_market,
            canon_expr=rep["canon_expr"],
            canon_hash=rep["canon_hash"],
            mode=mode,
            sign=sign,
            n_nodes=int(rep.get("n_nodes", 1)),
            depth=int(rep.get("depth", 1)),
            source="dl_alpha" if defn.factor_type == "dl_alpha" else source,
            recipe_id=getattr(defn, "recipe_id", None),
            data_version=data_version,
            status="candidate",
            name=defn.name,
            factor_type=defn.factor_type,
            definition_json=defn.to_json(),
            materialization_path=getattr(defn, "materialization_path", None),
        )
        return self.register_factor(rec, upsert=True)

    def materialize_factor(
        self,
        factor_id: int,
        factor_matrix: np.ndarray,
        dates: Sequence[Any],
        symbols: Sequence[str],
        mat_dir: str | Path | None = None,
    ) -> Path:
        """将计算完成的 `[T x N]` 因子矩阵物化落盘 (`.npz` 压缩格式) 并绑定至注册表记录。"""
        rec = self.get_factor(factor_id, include_metrics=False)
        if rec is None:
            raise ValueError(f"factor_id={factor_id} 不存在")

        base_dir = (
            Path(mat_dir)
            if mat_dir is not None
            else (self.db_path.parent / "materialized" / rec.market)
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"factor_{factor_id}_{rec.canon_hash}.npz"

        np.savez_compressed(
            out_path,
            matrix=np.asarray(factor_matrix, dtype=np.float64),
            dates=np.array([str(d) for d in dates]),
            symbols=np.array([str(s) for s in symbols]),
        )

        now_str = dt.datetime.now(dt.UTC).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE factor SET materialization_path = ?, updated_at = ? WHERE id = ?",
                (str(out_path), now_str, factor_id),
            )
        return out_path

    def load_materialized_factor(
        self, factor_id: int
    ) -> tuple[np.ndarray, list[str], list[str]] | None:
        """加载已物化落盘的因子矩阵 `(matrix, dates, symbols)`。"""
        rec = self.get_factor(factor_id, include_metrics=False)
        if rec is None or not rec.materialization_path:
            return None
        p = Path(rec.materialization_path)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as data:
            mat = np.asfortranarray(data["matrix"], dtype=np.float64)
            dates = [str(x) for x in data["dates"].tolist()]
            symbols = [str(x) for x in data["symbols"].tolist()]
        return mat, dates, symbols

    def lock_factor_version(
        self,
        factor_id: int,
        probe_report: dict[str, Any] | None = None,
        eval_summary: dict[str, Any] | None = None,
    ) -> FactorVersionRecord:
        """在物化或回测时为当前因子生成不可变版本号快照 (v1, v2, ...)。"""
        rec = self.get_factor(factor_id, include_metrics=False)
        if rec is None:
            raise ValueError(f"factor_id={factor_id} 不存在")

        now_str = dt.datetime.now(dt.UTC).isoformat()
        probe_str = (
            json.dumps(probe_report, ensure_ascii=False) if probe_report is not None else None
        )
        eval_str = (
            json.dumps(eval_summary, ensure_ascii=False) if eval_summary is not None else None
        )

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_v FROM factor_version WHERE factor_id = ?",
                (factor_id,),
            )
            next_v = int(cur.fetchone()["max_v"]) + 1
            defn_json = rec.definition_json or json.dumps(
                {"name": rec.name or f"factor_{factor_id}", "expression": rec.canon_expr}
            )
            cur.execute(
                """
                INSERT INTO factor_version (
                    factor_id, version, canon_expr, canon_hash, definition_json,
                    materialization_path, probe_json, eval_summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    factor_id,
                    next_v,
                    rec.canon_expr,
                    rec.canon_hash,
                    defn_json,
                    rec.materialization_path,
                    probe_str,
                    eval_str,
                    now_str,
                ),
            )
        return FactorVersionRecord(
            factor_id=factor_id,
            version=next_v,
            canon_expr=rec.canon_expr,
            canon_hash=rec.canon_hash,
            definition_json=defn_json,
            materialization_path=rec.materialization_path,
            probe_json=probe_str,
            eval_summary_json=eval_str,
            created_at=now_str,
        )

    def list_factor_versions(self, factor_id: int) -> list[FactorVersionRecord]:
        """查询指定因子的全部历史不可变版本快照。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM factor_version WHERE factor_id = ? ORDER BY version ASC",
                (factor_id,),
            )
            return [
                FactorVersionRecord(
                    factor_id=int(r["factor_id"]),
                    version=int(r["version"]),
                    canon_expr=r["canon_expr"],
                    canon_hash=r["canon_hash"],
                    definition_json=r["definition_json"],
                    materialization_path=r["materialization_path"],
                    probe_json=r["probe_json"],
                    eval_summary_json=r["eval_summary_json"],
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

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
        self,
        canon_hash: str,
        include_metrics: bool = True,
        market: str | None = None,
    ) -> FactorRecord | None:
        """按代数指纹 `canon_hash`（可选按 `market` 隔离）查询因子详情。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            if market is not None:
                norm_m = _normalize_market(market)
                cur.execute(
                    "SELECT * FROM factor WHERE market = ? AND canon_hash = ?",
                    (norm_m, canon_hash),
                )
            else:
                cur.execute(
                    "SELECT * FROM factor WHERE canon_hash = ? ORDER BY id ASC LIMIT 1",
                    (canon_hash,),
                )
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
        market: str | None = None,
        factor_type: str | None = None,
        source_run: str | None = None,
        recipe_id: str | None = None,
    ) -> list[FactorRecord]:
        """按多重过滤条件检索因子列表（支持 `market`、`factor_type`、`source_run`、`recipe_id`）。"""
        conditions = []
        params: list[Any] = []

        if market is not None:
            conditions.append("f.market = ?")
            params.append(_normalize_market(market))
        if status is not None:
            conditions.append("f.status = ?")
            params.append(status)
        if mode is not None:
            conditions.append("f.mode = ?")
            params.append(mode)
        if source is not None:
            conditions.append("f.source = ?")
            params.append(source)
        if factor_type is not None:
            conditions.append("f.factor_type = ?")
            params.append(factor_type)
        if source_run is not None:
            conditions.append("f.source_run = ?")
            params.append(source_run)
        if recipe_id is not None:
            conditions.append("f.recipe_id = ?")
            params.append(recipe_id)

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

    def get_accepted_factors(self, market: str | None = None) -> list[FactorRecord]:
        """获取准入合格的因子列表 (status = 'accepted')。"""
        return self.list_factors(status="accepted", market=market)

    def get_active_factors(self, market: str | None = None) -> list[FactorRecord]:
        """获取当前生产回测与 ML 特征池生效的因子列表 (status = 'active')。"""
        return self.list_factors(status="active", market=market)

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
        """物理删除因子（级联删除 metrics 与 version 明细）。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM factor WHERE id = ?", (factor_id,))
            return cur.rowcount > 0

    # =========================================================================
    # 命名特征池 (FeatureSet) 管理
    # =========================================================================

    def save_feature_set(self, fs: FeatureSetRecord, upsert: bool = True) -> int:
        """创建或更新命名特征池记录（按 `(market, name)` 唯一），返回 `feature_set.id`。"""
        now_str = dt.datetime.now(dt.UTC).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, version FROM feature_set WHERE market = ? AND name = ?",
                (fs.market, fs.name),
            )
            row = cur.fetchone()
            op_json = _canonical_json(fs.operators) if fs.operators is not None else None
            win_json = _canonical_json(fs.windows) if fs.windows is not None else None
            const_json = _canonical_json(fs.constants) if fs.constants is not None else None

            if row is not None:
                if not upsert:
                    raise ValueError(
                        f"特征池 '{fs.name}' 已存在于市场 '{fs.market}' 中 (id={row['id']})"
                    )
                fs_id = int(row["id"])
                next_ver = int(row["version"] or 1) + 1
                cur.execute(
                    """
                    UPDATE feature_set
                    SET description = ?, base_fields_json = ?, extensions_json = ?,
                        operators_json = ?, windows_json = ?, constants_json = ?,
                        aux_factor_ids_json = ?, default_postprocess_json = ?,
                        version = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        fs.description,
                        _canonical_json(fs.base_fields),
                        _canonical_json(fs.extensions),
                        op_json,
                        win_json,
                        const_json,
                        _canonical_json(fs.aux_factor_ids),
                        _canonical_json(fs.default_postprocess),
                        next_ver,
                        now_str,
                        fs_id,
                    ),
                )
                return fs_id

            cur.execute(
                """
                INSERT INTO feature_set (
                    market, name, description, base_fields_json, extensions_json,
                    operators_json, windows_json, constants_json,
                    aux_factor_ids_json, default_postprocess_json,
                    version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fs.market,
                    fs.name,
                    fs.description,
                    _canonical_json(fs.base_fields),
                    _canonical_json(fs.extensions),
                    op_json,
                    win_json,
                    const_json,
                    _canonical_json(fs.aux_factor_ids),
                    _canonical_json(fs.default_postprocess),
                    fs.version,
                    fs.created_at or now_str,
                    now_str,
                ),
            )
            return int(cur.lastrowid)

    def get_feature_set(self, fs_id: int) -> FeatureSetRecord | None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM feature_set WHERE id = ?", (fs_id,))
            row = cur.fetchone()
            return self._row_to_feature_set(row) if row is not None else None

    def get_feature_set_by_name(
        self, name: str, market: str = "futures"
    ) -> FeatureSetRecord | None:
        norm_m = _normalize_market(market)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM feature_set WHERE market = ? AND name = ?",
                (norm_m, name.strip()),
            )
            row = cur.fetchone()
            return self._row_to_feature_set(row) if row is not None else None

    def list_feature_sets(self, market: str | None = None) -> list[FeatureSetRecord]:
        query = "SELECT * FROM feature_set"
        params: list[Any] = []
        if market is not None:
            query += " WHERE market = ?"
            params.append(_normalize_market(market))
        query += " ORDER BY id ASC"
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            return [self._row_to_feature_set(r) for r in cur.fetchall()]

    def delete_feature_set(self, fs_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM feature_set WHERE id = ?", (fs_id,))
            return cur.rowcount > 0

    # =========================================================================
    # 不可变实验配方快照 (ResearchRecipe) 管理
    # =========================================================================

    def create_recipe(
        self,
        recipe: ResearchRecipeRecord | None = None,
        *,
        market: str = "futures",
        kind: str = "gp",
        feature_set: FeatureSetRecord | dict[str, Any] | None = None,
        feature_set_id: int | None = None,
        target_label: str = "y_5",
        splits: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        extensions: Sequence[str] | None = None,
        data_version: str = "v1",
        seed: int = 42,
        name: str = "",
    ) -> ResearchRecipeRecord:
        """创建并冻结不可变实验配方快照（同内容哈希自动幂等复用已有 `recipe_id`）。"""
        if recipe is None:
            fs_snapshot: dict[str, Any]
            resolved_fs_id = feature_set_id
            if isinstance(feature_set, FeatureSetRecord):
                fs_snapshot = feature_set.to_snapshot_dict()
                if resolved_fs_id is None:
                    resolved_fs_id = feature_set.id
            elif isinstance(feature_set, dict):
                fs_snapshot = copy.deepcopy(feature_set)
            elif feature_set_id is not None:
                loaded_fs = self.get_feature_set(feature_set_id)
                if loaded_fs is None:
                    raise ValueError(f"feature_set_id={feature_set_id} 不存在")
                fs_snapshot = loaded_fs.to_snapshot_dict()
                market = loaded_fs.market
            else:
                fs_snapshot = {}

            ext_list = (
                list(extensions)
                if extensions is not None
                else list(fs_snapshot.get("extensions", []))
            )
            recipe = ResearchRecipeRecord(
                market=market,
                kind=kind,
                feature_set=fs_snapshot,
                target_label=target_label,
                splits=splits or {},
                params=params or {},
                extensions=ext_list,
                data_version=data_version,
                seed=seed,
                name=name,
                feature_set_id=resolved_fs_id,
            )

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM research_recipe WHERE recipe_id = ?",
                (recipe.recipe_id,),
            )
            existing = cur.fetchone()
            if existing is not None:
                return self._row_to_recipe(existing)

            cur.execute(
                """
                INSERT INTO research_recipe (
                    recipe_id, content_hash, market, kind, name,
                    feature_set_id, feature_set_json, target_label,
                    splits_json, params_json, extensions_json,
                    data_version, seed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recipe.recipe_id,
                    recipe.content_hash,
                    recipe.market,
                    recipe.kind,
                    recipe.name,
                    recipe.feature_set_id,
                    _canonical_json(recipe.feature_set),
                    recipe.target_label,
                    _canonical_json(recipe.splits),
                    _canonical_json(recipe.params),
                    _canonical_json(recipe.extensions),
                    recipe.data_version,
                    recipe.seed,
                    recipe.created_at,
                ),
            )
        return recipe

    def get_recipe(self, recipe_id: str) -> ResearchRecipeRecord | None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM research_recipe WHERE recipe_id = ?", (recipe_id,))
            row = cur.fetchone()
            return self._row_to_recipe(row) if row is not None else None

    def list_recipes(
        self,
        market: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[ResearchRecipeRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if market is not None:
            conditions.append("market = ?")
            params.append(_normalize_market(market))
        if kind is not None:
            conditions.append("kind = ?")
            params.append(kind.strip().lower())

        query = "SELECT * FROM research_recipe"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            return [self._row_to_recipe(r) for r in cur.fetchall()]

    # =========================================================================
    # 实验运行元数据与心跳追踪 (Experiment Run)
    # =========================================================================

    def record_run(self, run: ExperimentRunRecord) -> None:
        """记录新的实验运行上下文（含 `market`, `recipe_id`, `pid`, `run_dir`）。"""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_run (
                    run_id, market, kind, recipe_id, config_yaml, data_version, git_sha,
                    n_evaluated, started_at, finished_at, status,
                    pid, progress_json, heartbeat_at, run_dir, log_tail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.market,
                    run.kind,
                    run.recipe_id,
                    run.config_yaml,
                    run.data_version,
                    run.git_sha,
                    run.n_evaluated,
                    run.started_at,
                    run.finished_at,
                    run.status,
                    run.pid,
                    run.progress_json,
                    run.heartbeat_at,
                    run.run_dir,
                    run.log_tail,
                ),
            )

    def update_run_heartbeat(
        self,
        run_id: str,
        progress: dict[str, Any] | None = None,
        log_tail: str | None = None,
        pid: int | None = None,
    ) -> None:
        """更新实验运行的进度快照与最新心跳时间。"""
        now_str = dt.datetime.now(dt.UTC).isoformat()
        prog_str = _canonical_json(progress) if progress is not None else None
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE experiment_run
                SET heartbeat_at = ?,
                    progress_json = COALESCE(?, progress_json),
                    log_tail = COALESCE(?, log_tail),
                    pid = COALESCE(?, pid)
                WHERE run_id = ?
                """,
                (now_str, prog_str, log_tail, pid, run_id),
            )

    def finish_run(
        self,
        run_id: str,
        status: str = "completed",
        n_evaluated: int | None = None,
    ) -> None:
        """更新实验运行状态与结束时间。"""
        if status not in VALID_RUN_STATUSES:
            raise ValueError(f"无效的运行状态: {status!r}, 必须在 {VALID_RUN_STATUSES} 中")
        now_str = dt.datetime.now(dt.UTC).isoformat()
        with self._get_connection() as conn:
            if n_evaluated is not None:
                conn.execute(
                    """
                    UPDATE experiment_run
                    SET status = ?, finished_at = ?, heartbeat_at = ?, n_evaluated = ?
                    WHERE run_id = ?
                    """,
                    (status, now_str, now_str, n_evaluated, run_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE experiment_run
                    SET status = ?, finished_at = ?, heartbeat_at = ?
                    WHERE run_id = ?
                    """,
                    (status, now_str, now_str, run_id),
                )

    def get_run(self, run_id: str) -> ExperimentRunRecord | None:
        """获取指定实验运行信息。"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM experiment_run WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            return self._row_to_run(row) if row is not None else None

    def list_runs(
        self,
        kind: str | None = None,
        status: str | None = None,
        market: str | None = None,
    ) -> list[ExperimentRunRecord]:
        """查询实验运行列表（支持按 `market`, `kind`, `status` 过滤）。"""
        conditions = []
        params: list[Any] = []
        if market is not None:
            conditions.append("market = ?")
            params.append(_normalize_market(market))
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
            return [self._row_to_run(r) for r in cur.fetchall()]

    # =========================================================================
    # 辅助工具函数
    # =========================================================================

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> ExperimentRunRecord:
        keys = set(row.keys())
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
            market=row["market"] if "market" in keys and row["market"] else "futures",
            recipe_id=row["recipe_id"] if "recipe_id" in keys else None,
            pid=row["pid"] if "pid" in keys else None,
            progress_json=row["progress_json"] if "progress_json" in keys else None,
            heartbeat_at=row["heartbeat_at"] if "heartbeat_at" in keys else None,
            run_dir=row["run_dir"] if "run_dir" in keys else None,
            log_tail=row["log_tail"] if "log_tail" in keys else None,
        )

    @staticmethod
    def _row_to_feature_set(row: sqlite3.Row) -> FeatureSetRecord:
        op_raw = row["operators_json"]
        win_raw = row["windows_json"]
        const_raw = row["constants_json"]
        return FeatureSetRecord(
            id=int(row["id"]),
            market=row["market"],
            name=row["name"],
            description=row["description"] or "",
            base_fields=json.loads(row["base_fields_json"] or "[]"),
            extensions=json.loads(row["extensions_json"] or "[]"),
            operators=json.loads(op_raw) if op_raw is not None else None,
            windows=json.loads(win_raw) if win_raw is not None else None,
            constants=json.loads(const_raw) if const_raw is not None else None,
            aux_factor_ids=json.loads(row["aux_factor_ids_json"] or "[]"),
            default_postprocess=json.loads(row["default_postprocess_json"] or "[]"),
            version=int(row["version"] or 1),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_recipe(row: sqlite3.Row) -> ResearchRecipeRecord:
        return ResearchRecipeRecord(
            recipe_id=row["recipe_id"],
            content_hash=row["content_hash"],
            market=row["market"],
            kind=row["kind"],
            name=row["name"] or "",
            feature_set_id=row["feature_set_id"],
            feature_set=json.loads(row["feature_set_json"] or "{}"),
            target_label=row["target_label"],
            splits=json.loads(row["splits_json"] or "{}"),
            params=json.loads(row["params_json"] or "{}"),
            extensions=json.loads(row["extensions_json"] or "[]"),
            data_version=row["data_version"],
            seed=int(row["seed"]),
            created_at=row["created_at"],
        )

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

        row_keys = set(row.keys())
        return FactorRecord(
            id=factor_id,
            market=row["market"] if "market" in row_keys and row["market"] else "futures",
            canon_expr=row["canon_expr"],
            canon_hash=row["canon_hash"],
            mode=row["mode"],
            sign=int(row["sign"]),
            n_nodes=int(row["n_nodes"]),
            depth=int(row["depth"]),
            source=row["source"],
            source_run=row["source_run"],
            recipe_id=row["recipe_id"] if "recipe_id" in row_keys else None,
            data_version=row["data_version"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metrics=metrics,
            name=row["name"] if "name" in row_keys else None,
            factor_type=row["factor_type"]
            if "factor_type" in row_keys and row["factor_type"]
            else "expression",
            definition_json=row["definition_json"] if "definition_json" in row_keys else None,
            draft_revision=int(row["draft_revision"])
            if "draft_revision" in row_keys and row["draft_revision"]
            else 1,
            materialization_path=row["materialization_path"]
            if "materialization_path" in row_keys
            else None,
        )
