"""可插拔 Reranker：装配时注入的检索后处理组件。

不传（None）= 没有重排步骤（基线）；传入实现 = 过召回后重新打分截断。
名字→对象的解析住在本模块（core），eval 只传名字字符串，评测概念不漏进 core。
"""
import asyncio
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """对召回候选重新打分排序，返回前 top_n 个。"""

    async def rerank(self, query: str, nodes: list, top_n: int) -> list: ...


class BgeReranker:
    """本地交叉编码器，包 LlamaIndex SentenceTransformerRerank（默认 bge-reranker-v2-m3）。

    模型同步推理，用 asyncio.to_thread 卸到线程，不堵事件循环。首次使用下载模型。
    """

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        from llama_index.core.postprocessor import SentenceTransformerRerank

        # 设大 top_n：让 postprocess 返回「全部按分排序」，截断交给本地 _postprocess，
        # 不再 per-call mutate 共享状态 → 实例可并发复用、无竞态。
        self._pp = SentenceTransformerRerank(model=model, top_n=10_000)

    async def rerank(self, query: str, nodes: list, top_n: int) -> list:
        if not nodes:
            return nodes
        return await asyncio.to_thread(self._postprocess, query, nodes, top_n)

    def _postprocess(self, query: str, nodes: list, top_n: int) -> list:
        from llama_index.core import QueryBundle

        ranked = self._pp.postprocess_nodes(nodes, query_bundle=QueryBundle(query))
        return ranked[:top_n]


# 名字 → 构造器。新增实现在此登记一行即可。
_REGISTRY = {
    "bge-reranker-v2-m3": lambda: BgeReranker("BAAI/bge-reranker-v2-m3"),
}

# 名字 → 已构造实例的缓存（一进程一次模型加载；instance 设计为可并发复用）。
_INSTANCES: dict = {}


def make_reranker(name: str | None) -> "Reranker | None":
    """名字 → 实例（按名缓存，模型只加载一次）。None/"" → None；未知名字 → ValueError。"""
    if not name:
        return None
    if name not in _REGISTRY:
        raise ValueError(
            f"未知 reranker 名字：{name!r}，可选：{list(_REGISTRY)}"
        )
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]
