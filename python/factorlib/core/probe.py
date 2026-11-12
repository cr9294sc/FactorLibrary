"""因子双模态定义、静态探针 (probe_factor) 与声明式后处理执行管线 (core/probe.py)。

对标 FactorLab MCP 架构的 Stage 2 (Authoring)、Stage 3 (Static Probe) 与 Stage 4 (Materialization Compute)：
- 支持 `expression` (Rust DSL 向量化表达式) 与 `code` (Python `def compute(context):` 因果沙箱) 双模态编写；
- 支持 12 个 C++ 级语义的声明式后处理算子链 (`baseUniMask`, `trdMask`, `dropNewStock`, `MAD`, `winsor`,
  `indNeut`, `riskNeut`, `riskNeutMulti`, `zScore`, `rank`, `decay`, `emadecay`)；
- 静态探针 `probe_factor()` 在不运行全量回测的前提下，静态提取显式与隐式依赖、校验无未来函数、
  推导 4 级预热回看窗口 (`expression_lookback`, `postprocess_lookback`, `total_required_lookback`, `warmup_bars`)，
  并精确复现隐式依赖在时序平滑算子 (`decay`/`emadecay`) 下的窗口放大效应。
"""

from __future__ import annotations

import ast as py_ast
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np

from factorlib.core.catalog import POSTPROCESS_SPECS, RECOMMENDED_POSTPROCESS_PRESETS
from factorlib.core.panel import Panel
from factorlib.core.styles import STYLE_DEFINITIONS, STYLE_NAMES, compute_futures_styles

VALID_POSTPROCESS_OPS: dict[str, dict[str, Any]] = {
    spec["name"]: spec for spec in POSTPROCESS_SPECS
}
STYLE_LOOKBACK_MAP: dict[str, int] = {s["name"]: int(s["lookback"]) for s in STYLE_DEFINITIONS}


@dataclass
class PostprocessStep:
    """单个声明式后处理步骤规格。"""

    op: str
    param: float | None = None
    window: int | None = None
    field: str | None = None
    fields: list[str] | None = None
    group_by: str | None = None

    def __post_init__(self) -> None:
        if self.op not in VALID_POSTPROCESS_OPS:
            raise ValueError(
                f"未知的后处理算子: {self.op!r}，有效算子为: {sorted(VALID_POSTPROCESS_OPS.keys())}"
            )
        spec = VALID_POSTPROCESS_OPS[self.op]
        defaults = spec.get("default_args", {})
        if self.op == "dropNewStock" and self.window is None:
            self.window = int(defaults.get("window", 60))
        elif self.op in ("MAD", "winsor") and self.param is None:
            self.param = float(defaults.get("param", 3.0 if self.op == "MAD" else 0.025))
        elif self.op == "indNeut" and self.group_by is None:
            self.group_by = str(defaults.get("group_by", "sector"))
        elif self.op == "riskNeut" and self.field is None:
            self.field = str(defaults.get("field", "SIZE"))
        elif self.op == "riskNeutMulti" and not self.fields:
            self.fields = list(defaults.get("fields", ["SIZE", "VOLATILITY", "CARRY"]))
        elif self.op in ("decay", "emadecay") and self.window is None:
            if self.param is not None:
                self.window = int(self.param)
            else:
                self.window = int(defaults.get("window", 5))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        if self.param is not None and self.op in ("MAD", "winsor"):
            d["param"] = float(self.param)
        if self.window is not None and self.op in ("dropNewStock", "decay", "emadecay"):
            d["window"] = int(self.window)
        if self.group_by is not None and self.op == "indNeut":
            d["group_by"] = self.group_by
        if self.field is not None and self.op == "riskNeut":
            d["field"] = self.field
        if self.fields is not None and self.op == "riskNeutMulti":
            d["fields"] = list(self.fields)
        return d

    def canonical_str(self) -> str:
        """生成确定性的后处理步骤签名字符串。"""
        if self.op in ("baseUniMask", "trdMask", "zScore", "rank"):
            return f"{self.op}()"
        if self.op in ("MAD", "winsor"):
            return f"{self.op}({float(self.param or 0.0):.6g})"
        if self.op in ("dropNewStock", "decay", "emadecay"):
            return f"{self.op}({int(self.window or 1)})"
        if self.op == "indNeut":
            return f"indNeut({self.group_by or 'sector'})"
        if self.op == "riskNeut":
            return f"riskNeut({self.field or 'SIZE'})"
        if self.op == "riskNeutMulti":
            f_str = ",".join(self.fields or [])
            return f"riskNeutMulti([{f_str}])"
        return f"{self.op}()"

    @classmethod
    def from_any(cls, item: str | dict[str, Any] | PostprocessStep) -> PostprocessStep:
        if isinstance(item, PostprocessStep):
            return item
        if isinstance(item, dict):
            op = str(item.get("op") or item.get("name") or "")
            return cls(
                op=op,
                param=float(item["param"])
                if "param" in item and item["param"] is not None
                else None,
                window=int(item["window"])
                if "window" in item and item["window"] is not None
                else None,
                field=str(item["field"]) if "field" in item and item["field"] is not None else None,
                fields=list(item["fields"])
                if "fields" in item and item["fields"] is not None
                else None,
                group_by=str(item["group_by"])
                if "group_by" in item and item["group_by"] is not None
                else None,
            )
        if isinstance(item, str):
            s = item.strip()
            if ":" in s:
                op_part, arg_part = s.split(":", 1)
                op_part = op_part.strip()
                arg_part = arg_part.strip()
                if op_part in ("MAD", "winsor"):
                    return cls(op=op_part, param=float(arg_part))
                if op_part in ("dropNewStock", "decay", "emadecay"):
                    return cls(op=op_part, window=int(float(arg_part)))
                if op_part == "indNeut":
                    return cls(op=op_part, group_by=arg_part)
                if op_part == "riskNeut":
                    return cls(op=op_part, field=arg_part)
                if op_part == "riskNeutMulti":
                    fields = [x.strip() for x in arg_part.replace("+", ",").split(",") if x.strip()]
                    return cls(op=op_part, fields=fields)
            return cls(op=s)
        raise TypeError(f"无法解析后处理步骤: {item!r}")


def parse_postprocess_pipeline(
    spec: str | Sequence[str | dict[str, Any] | PostprocessStep] | None,
) -> list[PostprocessStep]:
    """将预设名、JSON 字符串、逗号分隔简写或字典列表统一解析为 PostprocessStep 列表。"""
    if spec is None:
        return []
    if isinstance(spec, str):
        s = spec.strip()
        if not s or s.lower() == "none":
            return []
        if s in RECOMMENDED_POSTPROCESS_PRESETS:
            return [PostprocessStep.from_any(d) for d in RECOMMENDED_POSTPROCESS_PRESETS[s]]
        if s.startswith("["):
            loaded = json.loads(s)
            return [PostprocessStep.from_any(d) for d in loaded]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [PostprocessStep.from_any(p) for p in parts]
    return [PostprocessStep.from_any(it) for it in spec]


@dataclass
class FactorDefinition:
    """统一因子定义模型（支持 DSL 表达式或 Python `compute(context)` 代码 + 声明式后处理管线）。"""

    name: str = "unnamed_factor"
    factor_type: str = "expression"  # "expression" | "code"
    expression: str = ""
    code: str = ""
    postprocess: list[PostprocessStep] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        self.factor_type = self.factor_type.strip().lower()
        if self.factor_type not in ("expression", "code"):
            raise ValueError(
                f"factor_type 必须为 'expression' 或 'code', 收到: {self.factor_type!r}"
            )
        self.postprocess = parse_postprocess_pipeline(self.postprocess)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "factor_type": self.factor_type,
            "expression": self.expression,
            "code": self.code,
            "postprocess": [s.to_dict() for s in self.postprocess],
            "description": self.description,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactorDefinition:
        f_type = str(data.get("factor_type") or ("code" if data.get("code") else "expression"))
        return cls(
            name=str(data.get("name", "unnamed_factor")),
            factor_type=f_type,
            expression=str(data.get("expression") or data.get("expr") or ""),
            code=str(data.get("code") or ""),
            postprocess=parse_postprocess_pipeline(data.get("postprocess")),
            description=str(data.get("description") or ""),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> FactorDefinition:
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".py":
            return cls(
                name=p.stem,
                factor_type="code",
                code=content,
            )
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(content)
            return cls.from_dict(data)
        data = json.loads(content)
        return cls.from_dict(data)


# ==============================================================================
# Stage 3: 静态探针 (probe_factor)
# ==============================================================================


def _extract_python_code_dependencies(code: str) -> tuple[list[tuple[str, int]], list[str]]:
    """通过 Python AST 静态扫描 `def compute(context):` 中的 `context.history(field, window)` 调用与安全性。"""
    errors: list[str] = []
    field_windows: dict[str, int] = {}

    try:
        tree = py_ast.parse(code)
    except SyntaxError as exc:
        return [], [f"Python 语法错误 (line {exc.lineno}): {exc.msg}"]

    forbidden_io_attrs = {
        "load",
        "save",
        "savez",
        "savez_compressed",
        "fromfile",
        "tofile",
        "loadtxt",
        "genfromtxt",
        "savetxt",
        "memmap",
        "ctypeslib",
        "DataSource",
    }

    has_compute = False
    for node in py_ast.walk(tree):
        if isinstance(node, py_ast.FunctionDef) and node.name == "compute":
            has_compute = True
            if len(node.args.args) < 1:
                errors.append("函数 `def compute(context):` 必须接受至少 1 个参数 `context`")
        # 阻断危险系统调用与沙箱逃逸 (P0-08)
        if isinstance(node, (py_ast.Import, py_ast.ImportFrom)):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in {"numpy", "np", "math"}:
                    errors.append(f"因子沙箱禁止导入外部模块: {alias.name!r} (仅允许 numpy / math)")
        if isinstance(node, py_ast.Attribute):
            if node.attr.startswith("_"):
                errors.append(f"因子沙箱禁止访问私有或双下划线属性: {node.attr!r}")
            elif node.attr in forbidden_io_attrs:
                errors.append(f"因子沙箱禁止访问文件或内存 I/O 接口: {node.attr!r}")
        if isinstance(node, py_ast.Name) and node.id.startswith("__"):
            errors.append(f"因子沙箱禁止引用双下划线内置名称: {node.id!r}")
        if isinstance(node, py_ast.Call) and isinstance(node.func, py_ast.Attribute):
            if node.func.attr == "history" and len(node.args) >= 1:
                arg0 = node.args[0]
                f_name = (
                    arg0.value
                    if isinstance(arg0, py_ast.Constant) and isinstance(arg0.value, str)
                    else None
                )
                win = 1
                if len(node.args) >= 2:
                    arg1 = node.args[1]
                    if isinstance(arg1, py_ast.Constant) and isinstance(arg1.value, (int, float)):
                        win = int(arg1.value)
                for kw in node.keywords:
                    if kw.arg == "window" and isinstance(kw.value, py_ast.Constant):
                        win = int(kw.value.value)
                if win <= 0:
                    errors.append(f"context.history 窗口参数必须 > 0, 收到: {win}")
                if f_name:
                    field_windows[f_name] = max(field_windows.get(f_name, 0), max(1, win))

    if not has_compute:
        errors.append("Python 因子代码必须定义入口函数 `def compute(context):`")

    return sorted(field_windows.items()), errors


def probe_factor(
    definition: FactorDefinition | dict[str, Any] | str,
    panel: Panel | None = None,
    available_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """静态探针：校验因子语法、提取显式与隐式依赖、计算 4 级预热回看窗口与代数指纹。

    严格遵循 FactorLab Schema v1 探针输出格式，并在毫秒级内完成零样本接触校验。
    """
    if isinstance(definition, str):
        defn = FactorDefinition(name="inline_expr", factor_type="expression", expression=definition)
    elif isinstance(definition, dict):
        defn = FactorDefinition.from_dict(definition)
    else:
        defn = definition

    # 汇总可用字段集合
    known_fields: set[str] = set()
    if panel is not None:
        known_fields.update(panel.fields.keys())
    if available_fields is not None:
        known_fields.update(available_fields)
    if not known_fields:
        from factorlib.core.catalog import FactorCatalog

        cat = FactorCatalog()
        known_fields.update(v["name"] for v in cat.variables)
    known_fields.update({"in_universe_mask", "limit_mask", "sector", "listed_days"})
    known_fields.update(STYLE_NAMES)

    errors: list[str] = []
    stage_warnings: list[str] = []
    explicit_vars: list[dict[str, Any]] = []
    used_functions: list[str] = []
    expr_lookback = 1
    base_canon_expr = ""
    base_canon_hash = ""
    n_nodes = 1
    depth = 1

    # 1. 解析主因子表达式或 Python 代码
    if defn.factor_type == "expression":
        if not defn.expression.strip():
            errors.append("DSL 表达式不能为空")
        else:
            try:
                v_info = factorlib_core.check_expr(defn.expression, sorted(known_fields))
                if not v_info.get("valid", True):
                    errors.append(str(v_info.get("error", "DSL 表达式静态检查未通过")))
                else:
                    base_canon_expr = str(v_info["canon_expr"])
                    base_canon_hash = str(v_info["canon_hash"])
                    n_nodes = int(v_info["n_nodes"])
                    depth = int(v_info["depth"])
                    expr_lookback = max(1, int(v_info.get("lookback", 1)))
                    used_functions = list(v_info.get("operators", []))
                    fw_raw = v_info.get("field_windows", {})
                    fw_items = fw_raw.items() if isinstance(fw_raw, dict) else fw_raw
                    for f_name, f_win in sorted(fw_items):
                        explicit_vars.append({"name": str(f_name), "window": int(f_win)})
            except Exception as exc:
                errors.append(str(exc))
    else:
        fw_list, code_errs = _extract_python_code_dependencies(defn.code)
        errors.extend(code_errs)
        for f_name, f_win in fw_list:
            if f_name not in known_fields:
                errors.append(f"未知的面板字段依赖: {f_name!r}")
            explicit_vars.append({"name": f_name, "window": int(f_win)})
            expr_lookback = max(expr_lookback, int(f_win))
        normalized_code = "\n".join(
            line.rstrip() for line in defn.code.strip().splitlines() if line.strip()
        )
        base_canon_expr = (
            f"py_compute({hashlib.sha256(normalized_code.encode('utf-8')).hexdigest()[:12]})"
        )
        base_canon_hash = hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()[:16]
        n_nodes = max(1, len(normalized_code.splitlines()))
        depth = 2

    # 2. 分析后处理管线：阶段顺序检查、时序平滑回看窗口与隐式依赖窗口放大
    prev_stage = 0
    for idx, step in enumerate(defn.postprocess):
        spec = VALID_POSTPROCESS_OPS[step.op]
        st_order = int(spec["stage_order"])
        if st_order < prev_stage:
            stage_warnings.append(
                f"后处理步骤 #{idx + 1} `{step.op}` (阶段 {st_order}) 位于阶段 {prev_stage} 之后，"
                "推荐顺序为: mask(1) -> outlier(2) -> neutralize(3) -> normalize(4) -> smooth(5)"
            )
        prev_stage = max(prev_stage, st_order)

    # 逆向扫描后处理链，计算每个步骤之后的累积平滑窗口放大增量
    n_steps = len(defn.postprocess)
    downstream_smooth_extra: list[int] = [0] * n_steps
    acc_smooth = 0
    for i in range(n_steps - 1, -1, -1):
        downstream_smooth_extra[i] = acc_smooth
        step = defn.postprocess[i]
        if step.op in ("decay", "emadecay"):
            w = max(1, int(step.window or 1))
            acc_smooth += max(0, w - 1)

    postprocess_lookback = 1 + acc_smooth if acc_smooth > 0 else 0

    postprocess_vars: list[dict[str, Any]] = []
    max_style_extra = 0
    for i, step in enumerate(defn.postprocess):
        amp_extra = downstream_smooth_extra[i]
        if step.op == "baseUniMask":
            postprocess_vars.append(
                {
                    "name": "in_universe_mask",
                    "window": 1 + amp_extra,
                    "source_op": "baseUniMask",
                }
            )
        elif step.op == "trdMask":
            postprocess_vars.append(
                {
                    "name": "limit_mask",
                    "window": 1 + amp_extra,
                    "source_op": "trdMask",
                }
            )
        elif step.op == "dropNewStock":
            postprocess_vars.append(
                {
                    "name": "listed_days",
                    "window": 1 + amp_extra,
                    "source_op": "dropNewStock",
                }
            )
        elif step.op == "indNeut":
            g_col = step.group_by or "sector"
            postprocess_vars.append(
                {
                    "name": g_col,
                    "window": 1 + amp_extra,
                    "source_op": "indNeut",
                }
            )
        elif step.op == "riskNeut":
            r_field = step.field or "SIZE"
            if r_field not in known_fields:
                errors.append(f"riskNeut 引用了未知的风险/风格变量: {r_field!r}")
            style_lb = STYLE_LOOKBACK_MAP.get(r_field, 1)
            max_style_extra = max(max_style_extra, style_lb + amp_extra)
            postprocess_vars.append(
                {
                    "name": r_field,
                    "window": style_lb + amp_extra,
                    "source_op": "riskNeut",
                }
            )
        elif step.op == "riskNeutMulti":
            for r_field in step.fields or []:
                if r_field not in known_fields:
                    errors.append(f"riskNeutMulti 引用了未知的风险/风格变量: {r_field!r}")
                style_lb = STYLE_LOOKBACK_MAP.get(r_field, 1)
                max_style_extra = max(max_style_extra, style_lb + amp_extra)
                postprocess_vars.append(
                    {
                        "name": r_field,
                        "window": style_lb + amp_extra,
                        "source_op": "riskNeutMulti",
                    }
                )

    # 3. 合成完整 canon_expr 与 canon_hash
    if defn.postprocess:
        pp_chain_str = " |> ".join(s.canonical_str() for s in defn.postprocess)
        full_canon_expr = f"{base_canon_expr} |> {pp_chain_str}"
        full_canon_hash = hashlib.sha256(full_canon_expr.encode("utf-8")).hexdigest()[:16]
    else:
        full_canon_expr = base_canon_expr
        full_canon_hash = base_canon_hash

    total_required_lookback = max(expr_lookback + acc_smooth, max_style_extra)
    warmup_bars = max(0, total_required_lookback - 1)

    return {
        "schema_version": "v1",
        "status": "ok" if not errors else "error",
        "name": defn.name,
        "factor_type": defn.factor_type,
        "canon_expr": full_canon_expr,
        "canon_hash": full_canon_hash,
        "n_nodes": n_nodes,
        "depth": depth,
        "dependencies": {
            "variables": explicit_vars,
            "functions": used_functions,
        },
        "postprocess_dependencies": {
            "variables": postprocess_vars,
            "steps": [s.to_dict() for s in defn.postprocess],
        },
        "execution_requirements": {
            "expression_lookback": int(expr_lookback),
            "postprocess_lookback": int(postprocess_lookback),
            "total_required_lookback": int(total_required_lookback),
            "warmup_bars": int(warmup_bars),
        },
        "stage_order_warnings": stage_warnings,
        "errors": errors,
    }


# ==============================================================================
# Stage 4: 因子双模态求值与声明式后处理执行 (evaluate_factor_definition)
# ==============================================================================


class PythonFactorContext:
    """严格因果的 Python 自定义因子运行时上下文 (`def compute(context):`)。

    在每个交易日截面步 `t` 上，`context.history(field, window)` 仅能读取 `[max(0, t-window+1) .. t]`
    的历史切片（不足 `window` 行时顶部填充 NaN 保持 `(window, N)` 形状），从物理内存视图上彻底阻断未来数据访问。
    """

    __slots__ = ("_dates", "_symbols", "_t", "_n_symbols", "_history_fn")

    def __init__(
        self,
        fields: dict[str, np.ndarray],
        dates: Sequence[Any],
        symbols: Sequence[str],
        current_t: int,
    ) -> None:
        self._dates = dates
        self._symbols = list(symbols)
        self._t = int(current_t)
        self._n_symbols = len(self._symbols)

        # P0-08: 通过闭包隔离底层全量 fields 引用，不在实例属性上暴露原字典，彻底阻断未来行读取
        t_idx = self._t
        n_sym = self._n_symbols

        def _fetch_history(field_name: str, window: int = 1) -> np.ndarray:
            if window <= 0:
                raise ValueError(f"history window 必须为正整数, 收到: {window}")
            if field_name not in fields:
                raise KeyError(f"面板中不存在字段: {field_name!r}")

            mat = fields[field_name]
            end_idx = t_idx + 1
            start_idx = max(0, end_idx - window)
            avail = mat[start_idx:end_idx, :]

            if avail.shape[0] < window:
                pad_rows = window - avail.shape[0]
                out = np.full((window, n_sym), np.nan, dtype=np.float64)
                out[pad_rows:, :] = avail
            else:
                out = avail.copy()

            out.flags.writeable = False
            return out

        self._history_fn = _fetch_history

    @property
    def t(self) -> int:
        return self._t

    @property
    def current_date(self) -> str:
        return str(self._dates[self._t])

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def history(self, field_name: str, window: int = 1) -> np.ndarray:
        """获取截至当前交易日 t 的过去 `window` 日面板切片，返回形状固定为 `(window, N)` 的只读数组。"""
        return self._history_fn(field_name, window)


def _execute_python_factor(code: str, panel: Panel) -> np.ndarray:
    """在因果沙箱中按时间步执行 `def compute(context):` 并组装 `(T, N)` 因子矩阵。"""
    import math

    _, ast_errors = _extract_python_code_dependencies(code)
    if ast_errors:
        raise ValueError(f"Python 因子未通过静态沙箱安全检查: {'; '.join(ast_errors)}")

    safe_globals: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "ValueError": ValueError,
            "KeyError": KeyError,
        },
        "np": np,
        "numpy": np,
        "math": math,
    }
    local_ns: dict[str, Any] = {}
    exec(compile(code, "<factor_code>", "exec"), safe_globals, local_ns)

    compute_fn = local_ns.get("compute") or safe_globals.get("compute")
    if not callable(compute_fn):
        raise ValueError("Python 因子代码未定义可调用的 `def compute(context):` 函数")

    T, N = panel.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")

    for t in range(T):
        ctx = PythonFactorContext(panel.fields, panel.dates, panel.symbols, current_t=t)
        row_val = compute_fn(ctx)
        if row_val is None:
            continue
        arr = np.asarray(row_val, dtype=np.float64)
        if arr.ndim == 2 and arr.shape == (1, N):
            arr = arr[0]
        if arr.shape != (N,):
            raise ValueError(
                f"`compute(context)` 在 t={t} 必须返回长度为 N={N} 的一维截面向量，实际形状: {arr.shape}"
            )
        out[t, :] = arr

    return out


def _resolve_implicit_field(
    field_name: str,
    panel: Panel,
    styles_cache: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """解析后处理隐式依赖矩阵（支持面板字段、隐式掩码与 6 大期货风格因子）。"""
    T, N = panel.shape
    if field_name in panel.fields:
        return np.asfortranarray(panel.fields[field_name], dtype=np.float64), styles_cache

    if field_name == "in_universe_mask":
        mask_arr = np.where(panel.valid_mask, 1.0, np.nan)
        return np.asfortranarray(mask_arr, dtype=np.float64), styles_cache

    if field_name == "limit_mask":
        mask_arr = np.where(panel.valid_mask, 1.0, np.nan)
        return np.asfortranarray(mask_arr, dtype=np.float64), styles_cache

    if field_name == "listed_days":
        # 若合成测试面板无 listed_days，默认视为已过上市期 (999.0)
        return np.asfortranarray(np.full((T, N), 999.0, dtype=np.float64)), styles_cache

    if field_name == "sector":
        # 若合成测试面板无 sector，默认赋同一板块编码 1.0
        return np.asfortranarray(np.ones((T, N), dtype=np.float64)), styles_cache

    if field_name in STYLE_NAMES:
        if styles_cache is None:
            styles_cache = compute_futures_styles(panel)
        return np.asfortranarray(styles_cache[field_name], dtype=np.float64), styles_cache

    raise KeyError(f"无法解析后处理依赖字段或风格因子: {field_name!r}")


def apply_postprocess_pipeline(
    raw_factor: np.ndarray,
    steps: Sequence[PostprocessStep | dict[str, Any] | str],
    panel: Panel,
) -> np.ndarray:
    """在 Rust 内核中按序执行声明式后处理算子链。"""
    parsed_steps = parse_postprocess_pipeline(steps)
    if not parsed_steps:
        return np.asfortranarray(raw_factor, dtype=np.float64)

    curr = np.asfortranarray(raw_factor, dtype=np.float64)
    styles_cache: dict[str, np.ndarray] | None = None

    for step in parsed_steps:
        op = step.op
        if op == "baseUniMask":
            mask_mat, styles_cache = _resolve_implicit_field(
                "in_universe_mask", panel, styles_cache
            )
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op("baseUniMask", curr, x2=mask_mat)
            )
        elif op == "trdMask":
            mask_mat, styles_cache = _resolve_implicit_field("limit_mask", panel, styles_cache)
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op("trdMask", curr, x2=mask_mat)
            )
        elif op == "dropNewStock":
            ld_mat, styles_cache = _resolve_implicit_field("listed_days", panel, styles_cache)
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op(
                    "dropNewStock", curr, x2=ld_mat, param=float(step.window or 60)
                )
            )
        elif op in ("MAD", "winsor"):
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op(op, curr, param=float(step.param or 3.0))
            )
        elif op == "indNeut":
            grp_col = step.group_by or "sector"
            grp_mat, styles_cache = _resolve_implicit_field(grp_col, panel, styles_cache)
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op("indNeut", curr, x2=grp_mat)
            )
        elif op == "riskNeut":
            r_col = step.field or "SIZE"
            r_mat, styles_cache = _resolve_implicit_field(r_col, panel, styles_cache)
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op("riskNeut", curr, x2=r_mat)
            )
        elif op == "riskNeutMulti":
            exp_list: list[np.ndarray] = []
            for r_col in step.fields or ["SIZE", "VOLATILITY", "CARRY"]:
                r_mat, styles_cache = _resolve_implicit_field(r_col, panel, styles_cache)
                exp_list.append(r_mat)
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op("riskNeutMulti", curr, exposures=exp_list)
            )
        elif op in ("zScore", "rank"):
            curr = np.asfortranarray(factorlib_core.apply_postprocess_op(op, curr))
        elif op in ("decay", "emadecay"):
            curr = np.asfortranarray(
                factorlib_core.apply_postprocess_op(op, curr, param=float(step.window or 5))
            )
        else:
            raise ValueError(f"不支持的后处理算子: {op!r}")

    return curr


def evaluate_factor_definition(
    definition: FactorDefinition | dict[str, Any] | str,
    panel: Panel,
    engine: factorlib_core.Engine | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行静态探针校验、主因子求值与声明式后处理管线，返回 `(factor_matrix, probe_report)`。"""
    if isinstance(definition, str):
        defn = FactorDefinition(name="inline_expr", factor_type="expression", expression=definition)
    elif isinstance(definition, dict):
        defn = FactorDefinition.from_dict(definition)
    else:
        defn = definition

    probe_rep = probe_factor(defn, panel=panel)
    if probe_rep["status"] != "ok":
        err_msg = "; ".join(probe_rep["errors"])
        raise ValueError(f"因子静态探针校验未通过 ({defn.name}): {err_msg}")

    if defn.factor_type == "expression":
        rust_engine = engine if engine is not None else panel.to_rust_engine()
        raw_mat = np.asfortranarray(rust_engine.eval(defn.expression), dtype=np.float64)
    else:
        raw_mat = _execute_python_factor(defn.code, panel)

    final_mat = apply_postprocess_pipeline(raw_mat, defn.postprocess, panel)
    return final_mat, probe_rep
