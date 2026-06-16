"""core/retrieval/rerank.py 单测：工厂映射 + 协议一致性。

真实 bge 模型需下载 ~600MB，不在单测范围；这里只验证名字→对象的解析与边界，
构造真实 BgeReranker 不触发。
"""
import pytest

from core.retrieval.rerank import Reranker, make_reranker


def test_make_reranker_none_returns_none():
    assert make_reranker(None) is None
    assert make_reranker("") is None


def test_make_reranker_unknown_name_raises():
    with pytest.raises(ValueError):
        make_reranker("no-such-reranker")


def test_make_reranker_bge_name_is_registered():
    from core.retrieval.rerank import _REGISTRY
    assert "bge-reranker-v2-m3" in _REGISTRY


class _FakeReranker:
    async def rerank(self, query, nodes, top_n):
        return nodes[:top_n]


def test_protocol_runtime_check_accepts_conforming_object():
    assert isinstance(_FakeReranker(), Reranker)


def test_make_reranker_caches_instance_per_name(monkeypatch):
    import core.retrieval.rerank as mod

    created = []

    class _Cheap:
        async def rerank(self, query, nodes, top_n):
            return nodes[:top_n]

    def _make_cheap():
        inst = _Cheap()
        created.append(inst)
        return inst

    monkeypatch.setattr(mod, "_REGISTRY", {"cheap": _make_cheap})
    monkeypatch.setattr(mod, "_INSTANCES", {})  # 隔离缓存，避免污染其他测试

    first = mod.make_reranker("cheap")
    second = mod.make_reranker("cheap")

    assert first is second           # 同名复用同一实例
    assert len(created) == 1          # 只构造一次
