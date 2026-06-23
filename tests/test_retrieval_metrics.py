import math

import pytest

from eval.retrieval.metrics import (
    K_VALUES,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_partial_and_full():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "x"}
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)   # 命中 a
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)   # 命中 a,c


def test_recall_empty_relevant_is_none():
    assert recall_at_k(["a"], set(), 5) is None


def test_precision_divides_by_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(1 / 2)
    assert precision_at_k(retrieved, relevant, 0) is None


def test_mrr_rank_and_miss():
    assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
    assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_single_hit_at_rank2():
    # b 在 index1（rank2）：dcg=1/log2(3)；理想命中数 1：idcg=1/log2(2)=1
    val = ndcg_at_k(["a", "b"], {"b"}, 2)
    assert val == pytest.approx(1 / math.log2(3))


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_empty_relevant_is_none():
    assert ndcg_at_k(["a"], set(), 5) is None


def test_k_values_constant():
    assert K_VALUES == (1, 3, 5, 10)
