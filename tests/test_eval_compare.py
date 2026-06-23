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


# ── build_sut 工厂与两路线 VARIANTS ──────────────────────────────
from eval.harness.compare import build_sut, AGENT_VARIANT, WORKFLOW_VARIANT, VARIANTS
from eval.harness.sut import AgentSystem, DocQueryWorkflowSystem


def test_variants_are_exactly_two_routes():
    assert set(VARIANTS) == {WORKFLOW_VARIANT, AGENT_VARIANT}
    assert VARIANTS[WORKFLOW_VARIANT] == {}
    assert VARIANTS[AGENT_VARIANT] is None


def test_build_sut_agent_returns_agent_system():
    sut = build_sut(AGENT_VARIANT, index_manager=object(), llm=object())
    assert isinstance(sut, AgentSystem)


def test_build_sut_workflow_returns_workflow_system():
    sut = build_sut(WORKFLOW_VARIANT, index_manager=object(), llm=object())
    assert isinstance(sut, DocQueryWorkflowSystem)


def test_build_sut_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        build_sut("不存在的变体", index_manager=object(), llm=object())


# ── baseline 回退（默认 baseline 不在所选 --variants 子集里时回退首个）──
from eval.harness.compare import resolve_baseline


def test_resolve_baseline_present_returns_it():
    assert resolve_baseline("workflow", ["workflow", "agent"]) == "workflow"


def test_resolve_baseline_absent_falls_back_to_first():
    assert resolve_baseline("不存在", ["agent", "workflow"]) == "agent"


def test_render_delta_table_raises_on_missing_baseline():
    import pytest
    variants = [{"name": "全开", "report": {"classification": {"accuracy": 0.7}, "metric_means": {}}}]
    with pytest.raises(ValueError):
        render_delta_table(variants, baseline="不存在的baseline")


# ── 打分函数（原 run_eval，已搬入 compare.py）────────────────────────
from dataclasses import dataclass

from eval.harness.compare import aggregate, score_row, _row_to_dict, load_testset
from eval.harness.metrics import MetricSpec
from eval.harness.sut import RagOutput


class _AttrRow:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_row_to_dict_from_dict():
    assert _row_to_dict({"user_input": "Q", "reference": "R"})["user_input"] == "Q"


def test_row_to_dict_from_attr_object():
    d = _row_to_dict(_AttrRow(user_input="Q", reference="R"))
    assert d["user_input"] == "Q" and d["reference"] == "R"


def test_aggregate_means_only_over_answered():
    rows = [
        {"outcome": "answered", "faithfulness": 1.0, "answer_relevancy": 0.8,
         "context_precision": 1.0, "context_recall": 0.5, "factual_correctness": 0.6},
        {"outcome": "answered", "faithfulness": 0.0, "answer_relevancy": 0.6,
         "context_precision": 0.0, "context_recall": 0.5, "factual_correctness": 0.4},
        {"outcome": "empty"},
    ]
    rep = aggregate(rows)
    assert rep["total"] == 3
    assert rep["answered"] == 2
    assert rep["outcome_distribution"] == {"answered": 2, "empty": 1}
    assert rep["metric_means"]["faithfulness"] == 0.5
    assert rep["metric_means"]["answer_relevancy"] == 0.7
    assert "classification" not in rep
    assert "category_distribution" not in rep


def test_aggregate_ignores_none_scores():
    rows = [{"outcome": "answered", "faithfulness": None, "answer_relevancy": 0.4,
             "context_precision": None, "context_recall": None, "factual_correctness": None}]
    rep = aggregate(rows)
    assert rep["metric_means"]["faithfulness"] is None
    assert rep["metric_means"]["answer_relevancy"] == 0.4


def test_aggregate_cost_block_means_and_total():
    rows = [
        {"outcome": "answered", "latency_s": 1.0, "total_tokens": 100},
        {"outcome": "answered", "latency_s": 3.0, "total_tokens": 300},
    ]
    rep = aggregate(rows)
    assert rep["cost"]["mean_latency_s"] == 2.0
    assert rep["cost"]["mean_total_tokens"] == 200
    assert rep["cost"]["total_tokens"] == 400


def test_aggregate_cost_no_tokens_gives_none():
    rep = aggregate([{"outcome": "answered", "latency_s": 1.5}])
    assert rep["cost"]["mean_latency_s"] == 1.5
    assert rep["cost"]["mean_total_tokens"] is None
    assert rep["cost"]["total_tokens"] is None


@dataclass
class _MetricResult:
    value: float


class _FakeMetric:
    def __init__(self, value):
        self._v = value

    async def ascore(self, **kw):
        return _MetricResult(self._v)


class _FakeSUT:
    def __init__(self, out):
        self._out = out

    async def answer(self, query):
        return self._out


class _FakeMeter:
    def __init__(self, tokens):
        self._tokens = tokens
        self.reset_called = 0

    def reset(self):
        self.reset_called += 1

    def read(self):
        return self._tokens


async def test_score_row_answered_scores_all_metrics():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "answered"
    assert res["faithfulness"] == 0.9
    assert res["response"] == "A"
    assert "category" not in res


async def test_score_row_non_answered_skips_metrics():
    out = RagOutput(response="", retrieved_contexts=[], outcome="empty")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "empty"
    assert "faithfulness" not in res


async def test_score_row_refuse_category_skips_metrics_even_if_answered():
    out = RagOutput(response="编造的答案", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.0), lambda r, o: {})]
    res = await score_row(
        {"user_input": "Q", "reference": "", "category": "out_of_scope"}, _FakeSUT(out), specs
    )
    assert res["outcome"] == "answered"
    assert res["expected_category"] == "out_of_scope"
    assert "faithfulness" not in res
    assert res["latency_s"] >= 0


async def test_score_row_missing_info_category_skips_metrics():
    out = RagOutput(response="瞎猜的答案", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("answer_relevancy", _FakeMetric(0.4), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "category": "missing_info"}, _FakeSUT(out), specs)
    assert "answer_relevancy" not in res


async def test_score_row_explain_row_scores_and_is_not_refuse_skipped():
    out = RagOutput(response="教学体答案", retrieved_contexts=["c"], outcome="answered")
    specs = [
        MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {}),
        MetricSpec("answer_relevancy", _FakeMetric(0.8), lambda r, o: {}),
    ]
    res = await score_row({"user_input": "讲讲MVCC", "reference": ""}, _FakeSUT(out), specs)
    assert res["expected_category"] == ""
    assert res["faithfulness"] == 0.9
    assert res["answer_relevancy"] == 0.8


async def test_score_row_records_latency_without_meter():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert isinstance(res["latency_s"], float) and res["latency_s"] >= 0
    assert "total_tokens" not in res


async def test_score_row_with_meter_records_tokens_and_resets():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    meter = _FakeMeter({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), [], meter=meter)
    assert res["prompt_tokens"] == 100
    assert res["completion_tokens"] == 20
    assert res["total_tokens"] == 120
    assert meter.reset_called == 1


async def test_score_row_non_answered_still_records_latency_and_tokens():
    out = RagOutput(response="", retrieved_contexts=[], outcome="error")
    meter = _FakeMeter({"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5})
    res = await score_row({"user_input": "Q"}, _FakeSUT(out), [], meter=meter)
    assert res["latency_s"] >= 0
    assert res["total_tokens"] == 5
