"""实验追踪与运行上下文记录器 (core/runs.py)。

严格遵循白皮书 Q13、3.1 节与 10.1 节规范：
- 每次挖掘/训练/回测自动落盘唯一 `runs/<run_id>/` 目录；
- 记录完整 YAML 配置快照、Git Commit SHA、数据面板版本、逐代演化日志及准入结果；
- 与 `FactorRegistry` 的 `experiment_run` 表联动，实现实验全生命周期可追溯。
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from factorlib.core.registry import ExperimentRunRecord, FactorRegistry


def get_git_sha() -> str:
    """获取当前 Git HEAD 短哈希，若不在 Git 仓库中则返回 'unknown'。"""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        return res.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class RunTracker:
    """实验运行轨迹与产出目录管理器。"""

    def __init__(
        self,
        kind: str = "gp",
        run_id: str | None = None,
        runs_root: str | Path = "runs",
        config_yaml: str = "",
        data_version: str = "v1",
        registry: FactorRegistry | None = None,
    ) -> None:
        now = dt.datetime.now(dt.UTC)
        if run_id is None:
            ts_str = now.strftime("%Y%m%d_%H%M%S")
            run_id = f"{kind}_{ts_str}"

        self.run_id = run_id
        self.kind = kind
        self.runs_root = Path(runs_root)
        self.run_dir = self.runs_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config_yaml = config_yaml
        self.data_version = data_version
        self.git_sha = get_git_sha()
        self.started_at = now.isoformat()
        self.finished_at: str | None = None
        self.status = "running"
        self.n_evaluated = 0
        self.registry = registry

        # 落盘配置副本
        if self.config_yaml:
            (self.run_dir / "config.yaml").write_text(self.config_yaml, encoding="utf-8")

        # 写入初始元数据
        self._write_meta()

        # 同步至 SQLite 注册表
        if self.registry is not None:
            self.registry.record_run(
                ExperimentRunRecord(
                    run_id=self.run_id,
                    kind=self.kind,
                    config_yaml=self.config_yaml or f"{self.kind}_run",
                    data_version=self.data_version,
                    git_sha=self.git_sha,
                    n_evaluated=0,
                    started_at=self.started_at,
                    status="running",
                )
            )

    def _write_meta(self) -> None:
        meta = {
            "run_id": self.run_id,
            "kind": self.kind,
            "data_version": self.data_version,
            "git_sha": self.git_sha,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "n_evaluated": self.n_evaluated,
        }
        (self.run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def log_generation(self, stats: Any) -> None:
        """向 `generations.jsonl` 追加单代演化统计记录。"""
        payload = asdict(stats) if is_dataclass(stats) else dict(stats)
        line = json.dumps(payload, ensure_ascii=False)
        with (self.run_dir / "generations.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save_json(self, filename: str, data: Any) -> Path:
        """保存任意结构化 JSON 资产至当前运行目录。"""
        out_path = self.run_dir / filename
        out_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return out_path

    def save_text(self, filename: str, content: str) -> Path:
        """保存文本或 Markdown 报告至当前运行目录。"""
        out_path = self.run_dir / filename
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def finish(self, status: str = "completed", n_evaluated: int | None = None) -> None:
        """标记实验运行结束并刷新磁盘与数据库状态。"""
        self.status = status
        self.finished_at = dt.datetime.now(dt.UTC).isoformat()
        if n_evaluated is not None:
            self.n_evaluated = int(n_evaluated)
        self._write_meta()
        if self.registry is not None:
            self.registry.finish_run(self.run_id, status=self.status, n_evaluated=self.n_evaluated)
