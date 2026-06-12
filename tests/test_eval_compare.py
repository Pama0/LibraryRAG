"""对比表渲染纯逻辑单测（render_delta_table）。"""
from eval.compare import render_delta_table


def test_render_delta_table_marks_improvement():
    variants = [
        {"name": "baseline", "report": {"classification": {"accuracy": 0.6},
            "metric_means": {"context_recall": 0.62}}},
        {"name": "+probe", "report": {"classification": {"accuracy": 0.9},
            "metric_means": {"context_recall": 0.78}}},
    ]
    md = render_delta_table(variants, baseline="baseline")
    assert "| baseline |" in md
    assert "| +probe |" in md
    assert "+0.30" in md or "+0.3" in md   # 分类准确率 delta（0.9-0.6）
    assert "0.78" in md                     # context_recall 提升后的值


def test_render_delta_table_baseline_row_has_no_delta():
    variants = [
        {"name": "base", "report": {"classification": {"accuracy": 0.5}, "metric_means": {}}},
    ]
    md = render_delta_table(variants, baseline="base")
    assert "0.50" in md
    assert "(+0" not in md and "(-0" not in md   # baseline 自身不带 delta


def test_render_delta_table_none_metric_shows_dash():
    variants = [
        {"name": "base", "report": {"classification": {"accuracy": None}, "metric_means": {}}},
    ]
    md = render_delta_table(variants, baseline="base")
    assert "—" in md   # 无值列显示破折号
