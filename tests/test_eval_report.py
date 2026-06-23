"""eval/harness/report.py 纯展示+落盘单测。"""
import csv
import os
from datetime import datetime

from eval.harness.report import (
    default_result_paths,
    render_delta_table,
    write_detail_csv,
)


def test_render_delta_table_single_row_no_delta():
    variants = [{"name": "workflow", "report": {"metric_means": {"faithfulness": 0.9}}}]
    md = render_delta_table(variants, baseline="workflow")
    assert "| workflow |" in md
    assert "0.90" in md
    assert "(+0" not in md and "(-0" not in md


def test_render_delta_table_no_classification_column():
    variants = [{"name": "workflow", "report": {"metric_means": {"faithfulness": 0.9}}}]
    md = render_delta_table(variants, baseline="workflow")
    assert "分类准确率" not in md
    assert "faithfulness" in md


def test_render_delta_table_shows_cost_columns():
    variants = [{"name": "S", "report": {
        "metric_means": {"faithfulness": 0.9},
        "cost": {"mean_latency_s": 2.35, "mean_total_tokens": 1200.0, "total_tokens": 2400},
    }}]
    md = render_delta_table(variants, baseline="S")
    assert "时延(s/条)" in md and "tokens/条" in md
    assert "2.35" in md and "1200.00" in md


def test_render_delta_table_cost_missing_shows_dash():
    variants = [{"name": "S", "report": {"metric_means": {}}}]
    md = render_delta_table(variants, baseline="S")
    assert "时延(s/条)" in md and "tokens/条" in md
    assert "—" in md


def test_render_delta_table_raises_on_missing_baseline():
    import pytest
    variants = [{"name": "workflow", "report": {"metric_means": {}}}]
    with pytest.raises(ValueError):
        render_delta_table(variants, baseline="不存在")


def test_default_result_paths_defaults_to_compare_prefix():
    now = datetime(2026, 6, 18, 13, 0, 0)
    md, detail = default_result_paths(now=now)
    assert md.endswith(os.path.join("20260618_130000", "compare.md"))
    assert detail.endswith(os.path.join("20260618_130000", "compare_detail.csv"))


def test_write_detail_csv_includes_cost_columns(tmp_path):
    detail = [{"variant": "S", "user_input": "Q", "expected_category": "retrievable",
               "outcome": "answered", "response": "A", "num_contexts": 2,
               "latency_s": 1.2, "prompt_tokens": 100, "completion_tokens": 20,
               "total_tokens": 120}]
    path = tmp_path / "cost_detail.csv"
    write_detail_csv(detail, str(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["latency_s"] == "1.2"
    assert rows[0]["total_tokens"] == "120"
    assert rows[0]["expected_category"] == "retrievable"


def test_write_detail_csv_has_no_category_or_match_columns(tmp_path):
    detail = [{"variant": "S", "user_input": "Q", "expected_category": "x",
               "outcome": "answered", "response": "A", "num_contexts": 1}]
    path = tmp_path / "no_match.csv"
    write_detail_csv(detail, str(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert "match" not in header
    assert "category" not in header
    assert "expected_category" in header
