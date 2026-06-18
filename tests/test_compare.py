"""eval/harness/compare.py 的缺省落盘路径：eval/results 下带时间戳防覆盖。"""
import os
from datetime import datetime

from eval.harness.compare import default_result_paths


def test_default_paths_under_eval_results():
    md, csv = default_result_paths()
    assert os.path.normpath(os.path.dirname(md)) == os.path.join("eval", "results")
    assert os.path.normpath(os.path.dirname(csv)) == os.path.join("eval", "results")


def test_default_paths_carry_timestamp_and_pair_by_run():
    now = datetime(2026, 6, 17, 14, 30, 22)
    md, csv = default_result_paths(now=now)
    assert md.endswith("compare_20260617_143022.md")
    assert csv.endswith("compare_20260617_143022_detail.csv")  # 同戳配对、csv 带 _detail


def test_default_paths_differ_across_runs():
    a = default_result_paths(now=datetime(2026, 6, 17, 14, 30, 22))
    b = default_result_paths(now=datetime(2026, 6, 17, 14, 30, 23))
    assert a[0] != b[0] and a[1] != b[1]  # 时间戳不同 → 不覆盖
