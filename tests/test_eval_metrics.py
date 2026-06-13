from eval.harness.metrics import METRIC_KWARGS, MetricSpec
from eval.harness.sut import RagOutput


def _row():
    return {"user_input": "Q", "reference": "REF"}


def _out():
    return RagOutput(response="ANS", retrieved_contexts=["c1", "c2"], outcome="answered")


def test_faithfulness_kwargs():
    kw = METRIC_KWARGS["faithfulness"](_row(), _out())
    assert kw == {"user_input": "Q", "response": "ANS", "retrieved_contexts": ["c1", "c2"]}


def test_answer_relevancy_kwargs_omits_contexts():
    kw = METRIC_KWARGS["answer_relevancy"](_row(), _out())
    assert kw == {"user_input": "Q", "response": "ANS"}


def test_context_precision_kwargs():
    kw = METRIC_KWARGS["context_precision"](_row(), _out())
    assert kw == {"user_input": "Q", "reference": "REF", "retrieved_contexts": ["c1", "c2"]}


def test_context_recall_kwargs():
    kw = METRIC_KWARGS["context_recall"](_row(), _out())
    assert kw == {"user_input": "Q", "retrieved_contexts": ["c1", "c2"], "reference": "REF"}


def test_factual_correctness_kwargs():
    kw = METRIC_KWARGS["factual_correctness"](_row(), _out())
    assert kw == {"response": "ANS", "reference": "REF"}


def test_metric_spec_dataclass():
    spec = MetricSpec(name="x", metric=object(), kwargs=lambda r, o: {})
    assert spec.name == "x"
    assert callable(spec.kwargs)
