"""对比表渲染纯逻辑单测（render_delta_table）。"""
from eval.harness.compare import render_delta_table


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


# ── build_sut 工厂与 agent 哨兵变体 ──────────────────────────────
from eval.harness.compare import build_sut, AGENT_VARIANT, VARIANTS
from eval.harness.sut import AgentSystem, DocQueryWorkflowSystem


def test_agent_variant_registered_as_sentinel():
    assert AGENT_VARIANT in VARIANTS
    assert VARIANTS[AGENT_VARIANT] is None   # 哨兵：非 flags dict


def test_build_sut_agent_variant_returns_agent_system():
    sut = build_sut(AGENT_VARIANT, index_manager=object(), llm=object())
    assert isinstance(sut, AgentSystem)


def test_build_sut_workflow_variant_returns_workflow_system():
    sut = build_sut("baseline", index_manager=object(), llm=object())
    assert isinstance(sut, DocQueryWorkflowSystem)


def test_build_sut_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        build_sut("不存在的变体", index_manager=object(), llm=object())


# ── baseline 回退（文档命令场景：默认 baseline 不在所选 --variants 子集里）──
from eval.harness.compare import resolve_baseline


def test_resolve_baseline_present_returns_it():
    assert resolve_baseline("全开", ["全开", "agent(自主规划)"]) == "全开"


def test_resolve_baseline_absent_falls_back_to_first():
    # 文档命令场景：默认 baseline 不在所选变体里 → 回退到首个变体
    assert resolve_baseline("baseline(全单轮)", ["全开", "agent(自主规划)"]) == "全开"


def test_render_delta_table_raises_on_missing_baseline():
    import pytest
    variants = [{"name": "全开", "report": {"classification": {"accuracy": 0.7}, "metric_means": {}}}]
    with pytest.raises(ValueError):
        render_delta_table(variants, baseline="不存在的baseline")
