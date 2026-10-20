"""
Unit test for data audit script and generated audit report.
"""

from pathlib import Path
import pytest


def test_audit_script_exists():
    script_path = Path("scripts/audit_data.py")
    assert script_path.exists(), "scripts/audit_data.py must exist"


def test_audit_report_content():
    report_path = Path("data/derived/futures/audit_report.md")
    assert report_path.exists(), "data/derived/futures/audit_report.md must exist"

    content = report_path.read_text(encoding="utf-8")
    
    # 1. Check all 12 datasets are listed
    expected_datasets = [
        "calendar",
        "instruments_info",
        "industry_mapping",
        "dominant_mapping",
        "bar_1d",
        "dominant_bar_1m",
        "bar_1m",
        "dominant_ex_factor",
        "ex_factor",
        "member_rank",
        "roll_yield",
        "warehouse_stocks"
    ]
    for ds in expected_datasets:
        assert f"`{ds}`" in content, f"Dataset {ds} should be documented in the audit report"

    # 2. Check all 7 risks (R1 to R7) are verified
    for r_idx in range(1, 8):
        r_tag = f"R{r_idx}"
        assert f"### {r_tag}：" in content or f"### {r_tag}:" in content, f"Risk {r_tag} should be verified in the report"

    # 3. Check critical findings
    assert "rule=3" in content
    assert "PASS (已原生完备支持)" in content
    assert "VERIFIED (无未来函数，无需向后平移)" in content
    assert "CONFIRMED & MITIGATED" in content
    assert "min_names >= 10" in content
    assert "1,143,713,208" in content or "11.4 亿" in content
