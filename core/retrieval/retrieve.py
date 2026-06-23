"""可插拔 Retriever：装配时注入的检索数据源策略。

不传 / "vector" = 基线（向量检索）；"hybrid" = dense + BM25（Task 2 加）。
名字→对象在本模块（core）解析，eval 只传名字。Retriever 是数据源（不像 reranker 是
变换），依赖在 retrieve() 调用时传入，策略对象自身无依赖、由 make_retriever 零参构造。
"""
import asyncio
import re
from typing import Protocol, runtime_checkable

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

RRF_K = 60  # RRF 平滑常数（经验值）


@runtime_checkable
class Retriever(Protocol):
    """检索数据源：query → 候选 NodeWithScore 列表（已按相关度排序）。"""

    async def retrieve(
        self, query: str, *, index_manager, book_titles, top_k: int
    ) -> list: ...


def build_book_filters(book_titles):
    """scope 硬约束 → chroma 元数据过滤器；空范围 → None（全库）。"""
    if not book_titles:
        return None
    return MetadataFilters(filters=[
        MetadataFilter(
            key="book_title",
            operator=FilterOperator.IN,
            value=list(book_titles),
        ),
    ])


def bm25_tokenize(text: str) -> list[str]:
    """中文 BM25 分词：jieba 切词 + 小写 + 丢空白/纯标点 + 停用词（否则噪声毁排序）。

    停用词过滤对 query 与语料同表同序生效，把检索收紧到内容词；只滤真功能词，
    不碰「方法/机制」等中频实义词（IDF 已折损，删了反而误伤召回，见 stopwords.py）。
    """
    import jieba

    from core.retrieval.stopwords import STOPWORDS

    out: list[str] = []
    for t in jieba.lcut(text.lower()):
        t = t.strip()
        if not t or re.fullmatch(r"[\W_]+", t) or t in STOPWORDS:
            continue
        out.append(t)
    return out


def rrf_fuse(ranked_lists: list, top_k: int) -> list:
    """Reciprocal Rank Fusion：score(node)=Σ 1/(RRF_K+rank)，按 node id 去重排序截 top_k。

    ranked_lists：每个是已排序的 NodeWithScore 列表。返回新 NodeWithScore（score=RRF 分）。
    """
    scores: dict = {}
    keep: dict = {}
    for nodes in ranked_lists:
        for rank, nws in enumerate(nodes):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank)
            keep.setdefault(nid, nws)
    ordered = sorted(scores, key=lambda nid: -scores[nid])
    return [
        NodeWithScore(node=keep[nid].node, score=scores[nid])
        for nid in ordered[:top_k]
    ]


class VectorRetriever:
    """基线：当前向量检索（dense），等价改造前的 as_retriever 路径。"""

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        retriever = index_manager.get_index().as_retriever(
            similarity_top_k=top_k,
            filters=build_book_filters(book_titles),
        )
        return await retriever.aretrieve(query)


class HybridRetriever:
    """dense + BM25 的混合检索，RRF 融合。

    BM25 语料从 chroma 全量重建（懒构造 + 缓存 + 并发守卫，一进程只建一次）；
    dense 用 chroma 元数据过滤，BM25 对全库打分后按 scope 后过滤。
    """

    def __init__(self):
        self._bm25 = None
        self._nodes = None       # list[TextNode]，与 BM25 语料同序
        self._lock = asyncio.Lock()

    async def _ensure_bm25(self, index_manager):
        if self._bm25 is not None:
            return
        async with self._lock:
            if self._bm25 is not None:      # 双检：等锁期间别人已建好
                return
            data = index_manager.chroma_collection.get(
                include=["documents", "metadatas"])
            # 重建 + 分词 + 建索引是 CPU 活，卸到线程不堵事件循环
            self._nodes, self._bm25 = await asyncio.to_thread(self._build_bm25, data)

    @staticmethod
    def _build_bm25(data):
        from rank_bm25 import BM25Okapi

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        nodes = [
            TextNode(text=docs[i], id_=ids[i], metadata=metas[i] or {})
            for i in range(len(ids))
        ]
        corpus = [bm25_tokenize(n.text) for n in nodes]
        return nodes, BM25Okapi(corpus)

    def _bm25_search(self, query, book_titles, top_k):
        # 同步打分（O(N) over 全库）：学习项目规模下耗时可忽略，不卸线程；
        # 只有一次性的 BM25 build（_build_bm25）才走 to_thread。
        scores = self._bm25.get_scores(bm25_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            node = self._nodes[i]
            if book_titles and (node.metadata or {}).get("book_title") not in book_titles:
                continue
            out.append(NodeWithScore(node=node, score=float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        await self._ensure_bm25(index_manager)
        dense_retriever = index_manager.get_index().as_retriever(
            similarity_top_k=top_k,
            filters=build_book_filters(book_titles),
        )
        dense = await dense_retriever.aretrieve(query)
        sparse = self._bm25_search(query, book_titles, top_k)
        return rrf_fuse([dense, sparse], top_k)


# 名字 → 构造器。新增策略在此登记一行。
_REGISTRY = {
    "vector": VectorRetriever,
    "hybrid": HybridRetriever,
}

# 名字 → 已构造实例缓存（一进程一次；HybridRetriever 的 BM25 索引只建一次）。
_INSTANCES: dict = {}


def make_retriever(name):
    """名字 → 实例（按名缓存）。None/"vector" → VectorRetriever；未知 → ValueError。"""
    key = name or "vector"
    if key not in _REGISTRY:
        raise ValueError(f"未知 retriever 名字：{name!r}，可选：{list(_REGISTRY)}")
    if key not in _INSTANCES:
        _INSTANCES[key] = _REGISTRY[key]()
    return _INSTANCES[key]
