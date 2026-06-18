"""AutoAgent 单测：timeout 装配 + searched_queries 重置。

retriever/reranker 解析在 _ensure_agent 里，用 monkeypatch stub 掉，避免加载真模型。
timeout 落在 FunctionAgent 私有 _timeout（LlamaIndex 不暴露公有读），仅作装配验证。
"""
from llama_index.core.llms import MockLLM

import core.agent.auto_agent as aa


class _DummyIndexManager:
    chroma_collection = None

    def get_index(self):
        return object()  # 非 None 即可；_ensure_agent 只装配不检索


class _DummyRetriever:
    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        return []


def _stub_resolvers(monkeypatch):
    monkeypatch.setattr(aa, "make_retriever", lambda n: _DummyRetriever())
    monkeypatch.setattr(aa, "make_reranker", lambda n: None)


def test_auto_agent_timeout_default_wired(monkeypatch):
    _stub_resolvers(monkeypatch)
    a = aa.AutoAgent(_DummyIndexManager(), MockLLM())
    assert a.timeout == 120.0
    assert a._ensure_agent()._timeout == 120.0


def test_auto_agent_timeout_custom_wired(monkeypatch):
    _stub_resolvers(monkeypatch)
    a = aa.AutoAgent(_DummyIndexManager(), MockLLM(), timeout=7.0)
    assert a._ensure_agent()._timeout == 7.0
