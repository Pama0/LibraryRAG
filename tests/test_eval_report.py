"""eval/harness/report.py 纯展示+落盘单测（compare 与 run_eval 共用）。"""
import csv
from datetime import datetime

from eval.harness.report import (
    default_result_paths,
    render_delta_table,
    write_detail_csv,
)


# ── render_delta_table：单行（run_eval 单系统场景，自作 baseline，无 delta）──
def test_render_delta_table_single_row_no_delta():
    variants = [
        {"name": "当前系统", "report": {"classification": {"accuracy": 0.7},
            "metric_means": {"faithfulness": 0.9}}},
    ]
    md = render_delta_table(variants, baseline="当前系统")
    assert "| 当前系统 |" in md
    assert "0.70" in md and "0.90" in md
    assert "(+0" not in md and "(-0" not in md   # 单行=baseline 自身，无 delta


# ── default_result_paths：prefix 决定文件名前缀，时间戳防覆盖 ──
def test_default_result_paths_prefix_and_stamp():
    now = datetime(2026, 6, 18, 13, 0, 0)
    md, detail = default_result_paths(prefix="run_eval", now=now)
    assert md.endswith("run_eval_20260618_130000.md")
    assert detail.endswith("run_eval_20260618_130000_detail.csv")


def test_default_result_paths_defaults_to_compare_prefix():
    now = datetime(2026, 6, 18, 13, 0, 0)
    md, _ = default_result_paths(now=now)
    assert md.endswith("compare_20260618_130000.md")


# ── write_detail_csv：写出 variant + match 列（match=实判 vs 金标准一致）──
def test_write_detail_csv_writes_variant_and_match(tmp_path):
    detail = [
        {"variant": "当前系统", "user_input": "Q", "expected_category": "retrievable",
         "category": "retrievable", "outcome": "answered", "response": "A",
         "num_contexts": 2},
        {"variant": "当前系统", "user_input": "Q2", "expected_category": "other",
         "category": "retrievable", "outcome": "answered", "response": "B",
         "num_contexts": 1},
    ]
    path = tmp_path / "out_detail.csv"
    write_detail_csv(detail, str(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["variant"] == "当前系统"
    assert rows[0]["match"] == "1"     # category == expected_category
    assert rows[1]["match"] == "0"     # 不一致
