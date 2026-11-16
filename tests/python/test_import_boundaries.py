"""静态导入边界检查测试 (tests/python/test_import_boundaries.py)。

验证重构架构红线 (M1.2):
- `python/factorlib/core/` 不得导入 `factorlib.markets`。
"""

from __future__ import annotations

import ast
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[2] / "python" / "factorlib" / "core"


def _collect_imports(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                found.append(f"{'.' * node.level}{mod}")
            else:
                found.append(mod)
    return found


def test_core_never_imports_markets() -> None:
    """确认 core/ 目录内所有 Python 模块均未导入 factorlib.markets。"""
    py_files = sorted(CORE_DIR.rglob("*.py"))
    assert py_files, f"未在 {CORE_DIR} 中找到任何 Python 文件"

    violations: list[str] = []
    for path in py_files:
        for mod in _collect_imports(path):
            if (
                mod == "factorlib.markets"
                or mod.startswith("factorlib.markets.")
                or mod.startswith("..markets")
            ):
                violations.append(f"{path.relative_to(CORE_DIR.parent.parent)}: imports {mod!r}")

    assert not violations, "检测到 core/ 越界导入 factorlib.markets:\n" + "\n".join(violations)
