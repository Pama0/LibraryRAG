import pytest

from eval.retrieval.run import aggregate, per_query_metrics


def test_per_query_metrics_keys_and_values():
    row = per_query_metrics(["a", "b", "c"], {"a", "c"}, k_values=(1, 3))
    assert row["recall@1"] == pytest.approx(1 / 2)   # top1 命中 a
    assert row["recall@3"] == pytest.approx(1.0)
    assert row["precision@1"] == pytest.approx(1.0)
    assert row["mrr"] == pytest.approx(1.0)
    assert "ndcg@3" in row


def test_aggregate_ignores_none():
    rows = [
        {"recall@1": 1.0, "mrr": 0.5},
        {"recall@1": None, "mrr": 1.0},   # recall@1=None 不计入
    ]
    agg = aggregate(rows)
    assert agg["recall@1"] == pytest.approx(1.0)     # 只 (1.0)/1
    assert agg["mrr"] == pytest.approx(0.75)


def test_aggregate_empty_is_empty_dict():
    assert aggregate([]) == {}
