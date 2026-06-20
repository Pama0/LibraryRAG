"""eval/harness/compare.py 的缺省落盘路径：eval/results 下按时间戳文件夹防覆盖。"""
import os
from datetime import datetime

from eval.harness.compare import default_result_paths


def test_default_paths_under_eval_results_timestamp_folder():
    md, csv = default_result_paths()
    md_dir = os.path.normpath(os.path.dirname(md))
    assert os.path.dirname(md_dir) == os.path.join("eval", "results")  # 上一级是 eval/results
    assert os.path.dirname(csv) == os.path.dirname(md)                 # md 与 csv 同一时间戳文件夹


def test_default_paths_timestamp_folder_and_pair_by_run():
    now = datetime(2026, 6, 17, 14, 30, 22)
    md, csv = default_result_paths(now=now)
    run_dir = os.path.join("eval", "results", "20260617_143022")
    assert os.path.normpath(md) == os.path.join(run_dir, "compare.md")
    assert os.path.normpath(csv) == os.path.join(run_dir, "compare_detail.csv")


def test_default_paths_differ_across_runs():
    a = default_result_paths(now=datetime(2026, 6, 17, 14, 30, 22))
    b = default_result_paths(now=datetime(2026, 6, 17, 14, 30, 23))
    assert a[0] != b[0] and a[1] != b[1]  # 时间戳文件夹不同 → 不覆盖
