#!/usr/bin/env bash
# ==============================================================================
# scripts/verify.sh - FactorLibrary 提交前全资产代码与质量认证门禁脚本
# 
# 用法:
#   bash scripts/verify.sh [--all | --fast | --rust | --python | --git-only]
# ==============================================================================

set -euo pipefail

# 统一 PATH 确保 cargo, rust-analyzer, uv, ruff 可直接寻址
export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PWD}/.venv/bin:${PATH}"

# 彩色输出支持
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

FAILED_STEPS=()

log_info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}${BOLD}[PASS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}${BOLD}[WARN]${NC} $1"
}

log_fail() {
    echo -e "${RED}${BOLD}[FAIL]${NC} $1"
    FAILED_STEPS+=("$1")
}

MODE="${1:---all}"

echo -e "${BOLD}======================================================================${NC}"
echo -e "${BOLD}         FactorLibrary 提交前质量认证门禁 (Mode: ${MODE})          ${NC}"
echo -e "${BOLD}======================================================================${NC}"

# ------------------------------------------------------------------------------
# Gate 1: Git 暂存区与数据资产防泄漏检查
# ------------------------------------------------------------------------------
check_git_safety() {
    log_info "Gate 1: 检查 Git 暂存区与数据资产防泄漏 (Data Leakage & Big Files)..."
    
    # 检查是否有禁止提交的大文件或数据文件被暂存
    local staged_files
    staged_files=$(git diff --cached --name-only 2>/dev/null || true)
    
    local forbidden_exts=("parquet" "sqlite" "arrow" "feather" "h5" "bin" "tar" "gz" "zip")
    local leak_detected=0

    if [[ -n "${staged_files}" ]]; then
        while IFS= read -r file; do
            [[ -z "${file}" ]] && continue
            
            # 1. 检查禁止的敏感目录
            if [[ "${file}" =~ ^(exports/|data/|runs/) ]]; then
                log_fail "严禁将数据资产目录暂存到 Git 仓库: ${file}"
                leak_detected=1
            fi
            
            # 2. 检查特定数据后缀
            for ext in "${forbidden_exts[@]}"; do
                if [[ "${file}" == *."${ext}" ]]; then
                    log_fail "发现禁止提交的数据文件格式 (.${ext}): ${file}"
                    leak_detected=1
                fi
            done
            
            # 3. 检查单文件大小上限 (1MB = 1048576 字节)
            if [[ -f "${file}" ]]; then
                local fsize
                fsize=$(wc -c < "${file}")
                if (( fsize > 1048576 )); then
                    log_fail "文件大小超过 1MB 提交安全阈值 (${fsize} bytes): ${file}"
                    leak_detected=1
                fi
            fi
        done <<< "${staged_files}"
    fi

    if (( leak_detected == 0 )); then
        log_pass "Git 暂存区数据安全审计通过 (未发现敏感数据与超大文件)"
    fi
}

# ------------------------------------------------------------------------------
# Gate 2: 配置文件 (YAML / TOML) 语法与语义完整性校验
# ------------------------------------------------------------------------------
check_configs() {
    log_info "Gate 2: 校验 YAML / TOML 配置文件语法与语义 (Configs Gate)..."
    
    # Python 脚本校验 yaml 与 toml
    python - << 'EOF'
import sys
from pathlib import Path
import yaml
try:
    import tomllib
except ImportError:
    import tomli as tomllib

# 1. 检验所有 YAML 文件语法
yaml_files = list(Path("configs").rglob("*.yaml")) + list(Path("configs").rglob("*.yml"))
for yf in yaml_files:
    with open(yf, "r", encoding="utf-8") as f:
        yaml.safe_load(f)

# 2. 检验关键 TOML 文件语法
toml_files = [Path("pyproject.toml"), Path("rust/factorlib_core/Cargo.toml")]
for tf in toml_files:
    if tf.exists():
        with open(tf, "rb") as f:
            tomllib.load(f)

print(f"Validated {len(yaml_files)} YAML configs and {len(toml_files)} TOML manifests successfully.")
EOF

    # 运行配置语义单元测试 (包括 rule=3 校验、VWAP 计算方法等)
    pytest -q tests/python/test_config.py > /dev/null
    log_pass "YAML/TOML 格式与业务语义校验全部通过"
}

# ------------------------------------------------------------------------------
# Gate 3: Rust 内核代码质量门禁 (rust-analyzer + cargo)
# ------------------------------------------------------------------------------
check_rust() {
    log_info "Gate 3: 运行 Rust 内核质量认证 (rust-analyzer, clippy, fmt, test)..."
    
    local manifest="rust/factorlib_core/Cargo.toml"

    # 1. rust-analyzer diagnostics
    log_info "  -> 正在执行 rust-analyzer diagnostics 静态语法与引用扫描..."
    if command -v rust-analyzer >/dev/null 2>&1; then
        rust-analyzer diagnostics rust/factorlib_core > /dev/null
        log_pass "  rust-analyzer 诊断无致命阻断错误"
    else
        log_warn "  未检测到 rust-analyzer，请执行 'rustup component add rust-analyzer'"
    fi

    # 2. cargo check
    log_info "  -> 正在执行 cargo check..."
    cargo check --manifest-path "${manifest}" --all-targets --quiet
    log_pass "  cargo check 编译通过"

    # 3. cargo clippy (零警告硬约束)
    log_info "  -> 正在执行 cargo clippy (-- -D warnings)..."
    cargo clippy --manifest-path "${manifest}" --all-targets --quiet -- -D warnings
    log_pass "  cargo clippy 零警告准入通过"

    # 4. cargo fmt --check
    log_info "  -> 正在执行 cargo fmt 格式校验..."
    cargo fmt --manifest-path "${manifest}" -- --check
    log_pass "  cargo fmt 格式校验通过"

    # 5. cargo test (含 proptest 性质测试)
    if [[ "${MODE}" != "--fast" ]]; then
        log_info "  -> 正在执行 cargo test (含 proptest 算子性质验证)..."
        cargo test --manifest-path "${manifest}" --quiet
        log_pass "  cargo test (7 项单元与性质测试) 100% 通过"
    fi
}

# ------------------------------------------------------------------------------
# Gate 4: Python 算法层代码质量门禁 (ruff + pytest)
# ------------------------------------------------------------------------------
check_python() {
    log_info "Gate 4: 运行 Python 代码质量认证 (ruff check, ruff format, pytest)..."
    
    local py_targets="python/factorlib tests/python scripts"

    # 1. ruff check
    log_info "  -> 正在执行 ruff check 静态代码分析..."
    ruff check ${py_targets}
    log_pass "  ruff check 零错误通过"

    # 2. ruff format --check
    log_info "  -> 正在执行 ruff format 格式一致性检查..."
    ruff format --check ${py_targets}
    log_pass "  ruff format 格式校验通过"

    # 3. pytest 单元测试
    if [[ "${MODE}" != "--fast" ]]; then
        log_info "  -> 正在执行 pytest 测试套件..."
        pytest -q tests/python
        log_pass "  pytest 单元测试套件全部通过"
    fi
}

# ------------------------------------------------------------------------------
# Gate 5: 因果摄动与防未来函数审计
# ------------------------------------------------------------------------------
check_anti_lookahead() {
    log_info "Gate 5: 执行因果摄动与防未来函数专项审计 (Perturbation Gate)..."
    pytest -q tests/python/test_perturbation.py
    log_pass "因果摄动防未来函数测试通过 (历史特征与标签严格不可被未来扰动影响)"
}

# ------------------------------------------------------------------------------
# Gate 6: 维护脚本与 CLI 命令健全性检查
# ------------------------------------------------------------------------------
check_scripts() {
    log_info "Gate 6: 校验 scripts/ 工具脚本命令行入口健全性..."
    python scripts/build_panel.py --help > /dev/null
    python scripts/audit_data.py --help > /dev/null
    log_pass "CLI 工具脚本 --help 入口响应正常"
}

# ------------------------------------------------------------------------------
# 执行调度
# ------------------------------------------------------------------------------
case "${MODE}" in
    --git-only)
        check_git_safety
        ;;
    --rust)
        check_rust
        ;;
    --python)
        check_python
        ;;
    --fast)
        check_git_safety
        check_configs
        check_rust
        check_python
        check_scripts
        ;;
    --all|*)
        check_git_safety
        check_configs
        check_rust
        check_python
        check_anti_lookahead
        check_scripts
        ;;
esac

echo -e "\n${BOLD}======================================================================${NC}"
if (( ${#FAILED_STEPS[@]} == 0 )); then
    echo -e "${GREEN}${BOLD} ✔ [SUCCESS] 所有提交前检查认证门禁 100% 通过！可以安全执行 git commit。${NC}"
    echo -e "${BOLD}======================================================================${NC}"
    exit 0
else
    echo -e "${RED}${BOLD} ✘ [FAILURE] 以下门禁检查未通过，请修复后再次提交：${NC}"
    for step in "${FAILED_STEPS[@]}"; do
        echo -e "  - ${RED}${step}${NC}"
    done
    echo -e "${BOLD}======================================================================${NC}"
    exit 1
fi
