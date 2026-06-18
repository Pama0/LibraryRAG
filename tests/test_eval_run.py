from dataclasses import dataclass

from eval.harness.metrics import MetricSpec
from eval.harness.run_eval import aggregate, build_single_report, score_row, _row_to_dict
from eval.harness.sut import RagOutput


# ── _row_to_dict ──
class _AttrRow:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


def test_row_to_dict_from_dict():
    assert _row_to_dict({"user_input": "Q", "reference": "R"})["user_input"] == "Q"


def test_row_to_dict_from_attr_object():
    row = _AttrRow(user_input="Q", reference="R")
    d = _row_to_dict(row)
    assert d["user_input"] == "Q" and d["reference"] == "R"


# ── aggregate ──
def test_aggregate_means_only_over_answered():
    rows = [
        {"outcome": "answered", "faithfulness": 1.0, "answer_relevancy": 0.8,
         "context_precision": 1.0, "context_recall": 0.5, "factual_correctness": 0.6},
        {"outcome": "answered", "faithfulness": 0.0, "answer_relevancy": 0.6,
         "context_precision": 0.0, "context_recall": 0.5, "factual_correctness": 0.4},
        {"outcome": "clarify"},
    ]
    rep = aggregate(rows)
    assert rep["total"] == 3
    assert rep["answered"] == 2
    assert rep["outcome_distribution"] == {"answered": 2, "clarify": 1}
    assert rep["metric_means"]["faithfulness"] == 0.5
    assert rep["metric_means"]["answer_relevancy"] == 0.7


def test_aggregate_ignores_none_scores():
    rows = [
        {"outcome": "answered", "faithfulness": None, "answer_relevancy": 0.4,
         "context_precision": None, "context_recall": None, "factual_correctness": None},
    ]
    rep = aggregate(rows)
    assert rep["metric_means"]["faithfulness"] is None
    assert rep["metric_means"]["answer_relevancy"] == 0.4


# ── score_row ──
@dataclass
class _MetricResult:
    value: float


class _FakeMetric:
    def __init__(self, value): self._v = value
    async def ascore(self, **kw): return _MetricResult(self._v)


class _FakeSUT:
    def __init__(self, out): self._out = out
    async def answer(self, query): return self._out


async def test_score_row_answered_scores_all_metrics():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "answered"
    assert res["faithfulness"] == 0.9
    assert res["response"] == "A"


async def test_score_row_non_answered_skips_metrics():
    out = RagOutput(response="", retrieved_contexts=[], outcome="clarify")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "clarify"
    assert "faithfulness" not in res


# ── 分类准确率 / category 分布 ──
def test_aggregate_classification_accuracy_and_distribution():
    rows = [
        {"outcome": "answered", "category": "retrievable", "expected_category": "retrievable"},
        {"outcome": "answered", "category": "other", "expected_category": "retrievable"},  # 误判
        {"outcome": "empty", "category": "missing_info", "expected_category": "missing_info"},
    ]
    rep = aggregate(rows)
    assert rep["classification"]["accuracy"] == 2 / 3   # 2/3 判对
    assert rep["classification"]["correct"] == 2
    assert rep["category_distribution"]["other"] == 1
    assert rep["category_distribution"]["retrievable"] == 1


async def test_score_row_carries_category_and_expected():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered", category="other")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row(
        {"user_input": "Q", "reference": "R", "category": "retrievable"}, _FakeSUT(out), specs
    )
    assert res["category"] == "other"                  # SUT 实际判的
    assert res["expected_category"] == "retrievable"   # 测试集金标准标注


def test_aggregate_no_category_system_gives_na_accuracy():
    # agent 路线：每行 category 空、但金标准 expected_category 非空
    rows = [
        {"outcome": "answered", "category": "", "expected_category": "retrievable"},
        {"outcome": "answered", "category": "", "expected_category": "other"},
    ]
    rep = aggregate(rows)
    assert rep["classification"]["accuracy"] is None
    assert rep["classification"]["total"] == 0


def test_aggregate_skips_blank_category_but_keeps_others():
    # 混合：一行有 category（计分），一行空（跳过，不算误判）
    rows = [
        {"outcome": "answered", "category": "retrievable", "expected_category": "retrievable"},
        {"outcome": "error", "category": "", "expected_category": "other"},
    ]
    rep = aggregate(rows)
    assert rep["classification"]["total"] == 1
    assert rep["classification"]["accuracy"] == 1.0


# ── build_single_report：单系统 aggregate + 给每条打 variant 标（供 --detail 落盘）──
def test_build_single_report_tags_variant_and_aggregates():
    scored = [
        {"outcome": "answered", "category": "retrievable", "expected_category": "retrievable",
         "faithfulness": 1.0, "user_input": "Q"},
        {"outcome": "answered", "category": "other", "expected_category": "retrievable",
         "faithfulness": 0.5, "user_input": "Q2"},
    ]
    report, detail = build_single_report("当前系统", scored)
    assert report["classification"]["accuracy"] == 0.5      # 1/2 判对
    assert report["metric_means"]["faithfulness"] == 0.75
    assert [d["variant"] for d in detail] == ["当前系统", "当前系统"]
    assert detail[0]["user_input"] == "Q"
    assert len(detail) == 2
