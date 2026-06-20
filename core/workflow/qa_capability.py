"""QA capability：文档库问答的检索 + 合成实质逻辑（从 DocQueryWorkflow 抽出）。

`DocQueryWorkflow` 是顶层 Router 编排，把 `intent=qa` dispatch 到这里。本类是
注入式协作单元（与 `IntentRouter` / `QueryPreprocessor` / `QueryDecomposer` 同一
模式），【不依赖 LlamaIndex Workflow step 机制】，便于独立单测。

持有：
- 检索（`index_manager`）+ 合成（`llm`）。
- QA 内预处理 `QueryPreprocessor`（降噪 + 难度/明确性分类）。
- 拆解 `QueryDecomposer`（宽问题 → ≤N 子查询）。

对外暴露（均接收 query/book_titles，返回 (answer, source_nodes)，不碰流程事件）：
- `classify(clean_query)` → 预处理结果（category / rewritten_query / reason）。
- `retrieve(ctx, query, book_titles)` → 单轮检索 + 流式合成。
- `split(ctx, query, book_titles)` → 拆解-检索-map-reduce 汇总（失败降级单轮）。
- `assume(ctx, query, book_titles)` → 角度不定：归纳评判维度 → 声明所选角度 → 逐维度检索分节（失败降级单轮）。

流式：检索/合成进度通过 `ctx.write_event_to_stream` 推【流式专用事件】（本模块定义，
`doc_workflow` re-export、api 层映射成前端 SSE）。这些事件不参与 workflow step 图。
"""
import asyncio
import logging
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context, Event

from core.retrieval.rerank import Reranker
from core.retrieval.retrieve import Retriever, VectorRetriever
from core.workflow.chapter_tree import children, dominant_prefix, unique_chapters
from core.workflow.query_decompose import QueryDecomposer
from core.workflow.query_dimension import DimensionExtractor
from core.workflow.query_preprocess import QueryPreprocessor

logger = logging.getLogger(__name__)


# ── 流式专用事件（仅 write_event_to_stream，不参与 step 图）──────────
class RetrievalStartEvent(Event):
    """开始检索（→ 前端 tool_call）。"""

    query: str


class RetrievalDoneEvent(Event):
    """检索完成（→ 前端 tool_result，触发答案阶段）。"""

    count: int


class AnswerDeltaEvent(Event):
    """合成阶段逐 token（→ 前端 delta）。"""

    delta: str


class QaCapability:
    """文档库问答能力：降噪分类 → 检索 → 流式合成 / 拆解汇总。注入式，独立可测。"""

    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        reranker: "Reranker | None" = None,
        rerank_candidate_k: int = 20,
        retriever: "Retriever | None" = None,
        probe_retriever: "Retriever | None" = None,
        probe_reranker: "Reranker | None" = None,
    ):
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_sub_queries = max_sub_queries
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k
        # 检索不可跳过，基线=具体 VectorRetriever（不传即基线）
        self.retriever = retriever or VectorRetriever()
        # probe（探测召回判类）的检索与答案检索解耦：probe 求覆盖信号、答案求精排。
        # 默认 vector + 不重排（rerank 会收敛召回、压扁 pending_split 依赖的章节 spread 信号）。
        self.probe_retriever = probe_retriever or VectorRetriever()
        self.probe_reranker = probe_reranker
        self.preprocessor = QueryPreprocessor(llm)
        self.decomposer = QueryDecomposer(llm)
        self.dimensioner = DimensionExtractor(llm)
        self._retrieve_concurrency = 4  # 扇出检索并发上限，防 embedding/BM25/rerank 打爆

    # ── 预处理：降噪 + 难度/明确性分类（不再消指代）──────────────────
    async def classify(
        self,
        clean_query: str,
        book_titles: Optional[list[str]] = None,
        probe: bool = True,
    ):
        """先用 clean_query 探测召回，把召回信号喂给 judge，堵住「盲判」。

        probe=False（ablation baseline）→ 不探测、纯文本判定；probe 失败亦容错为空。
        """
        retrieval_context = ""
        if probe:
            try:
                located = await self._probe_retrieve(clean_query, book_titles)
                retrieval_context = self._format_probe(located, book_titles)
            except Exception as exc:
                logger.warning("classify probe 探测失败，退回纯文本判定：%s", exc)
        return await self.preprocessor.run(clean_query, retrieval_context)

    def _format_probe(self, nodes: list, book_titles) -> str:
        """探测召回 → 喂 judge 的信号：命中数 + 章节分布 + top 截断片段。"""
        if not nodes:
            return "知识库未召回到任何相关内容。"
        dist: list[str] = []
        seen: set = set()
        for n in nodes:
            meta = getattr(n, "metadata", None) or {}
            tag = f"《{meta.get('book_title', '?')}》{meta.get('chapter', '')}".strip()
            if tag not in seen:
                seen.add(tag)
                dist.append(tag)
        lines: list[str] = []
        for i, n in enumerate(nodes[:5], 1):
            meta = getattr(n, "metadata", None) or {}
            tag = f"《{meta.get('book_title', '?')}》{meta.get('chapter', '')}".strip()
            content = (
                n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            )[:150]
            lines.append(f"{i}. [{tag}] {content}")
        return (
            f"共命中 {len(nodes)} 段，跨 {len(dist)} 个章节：{'、'.join(dist)}\n"
            + "\n".join(lines)
        )

    # ── 分支：单轮检索 + 流式合成 ────────────────────────────────────
    async def retrieve(
        self,
        ctx: Context,
        query: str,
        book_titles: Optional[list[str]],
        preamble: str = "",
    ) -> tuple[str, list]:
        """直接检索 + 流式合成（绕开 agent/工具）。返回 (答案文本, source_nodes)。

        preamble 非空 → 进入答案阶段后先推一个 AnswerDeltaEvent，并拼在答案最前
        （供 missing_info 预算耗尽降级时声明"按最可能解读作答"）。空命中不带声明。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        nodes = await self._retrieve_nodes(query, book_titles)
        logger.info("retrieve: 命中 %d 段 scope=%s", len(nodes), book_titles)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(nodes)))
        if not nodes:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。", []
        if preamble:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=preamble))
        answer = await self._synthesize_stream(ctx, query, nodes)
        return (preamble + answer if preamble else answer), nodes

    async def assume(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """角度不定：定位 → LLM 归纳评判维度 → 声明所选角度 → 逐维度检索分节合成。

        归纳不出维度 → 降级为单轮合成（复用已定位结果，绝不阻塞）。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))

        # 1) 定位：一轮宽召回，拿正文供归纳维度
        located = await self._retrieve_nodes(query, book_titles)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]

        # 2) 归纳维度：从「问题 + 召回正文」产 (label, query) 维度对
        dimensions = await self.dimensioner.run(query, passages, self.max_sub_queries)

        # 降级：归纳不出维度 → 整句单轮合成
        if not dimensions:
            logger.info("assume: 无维度，降级单轮检索")
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return f"在{scope}没有检索到与「{query}」相关的内容。", []
            answer = await self._synthesize_stream(ctx, query, located)
            return answer, located

        # 3) 声明所选角度（透明 + 可纠偏）
        labels = "、".join(d.label for d in dimensions)
        preamble = f"「{query}」可以从以下角度来看：{labels}。下面分别说明——\n"

        # 4) 逐维度检索 + 分节合成（与 split 共用 helper）
        sections = [(d.label, d.query) for d in dimensions]
        return await self._retrieve_and_concat(ctx, sections, book_titles, preamble)

    async def split(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """定位 → 建骨架(结构主+内容辅) → 逐项检索 → map-reduce 汇总。

        拆解失败/空 → 降级为单轮检索+合成（等同 retrieve），绝不阻塞。
        """
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))

        # 1) 定位：一轮宽召回，拿命中 chunk 的 chapter
        located = await self._retrieve_nodes(query, book_titles)

        # 2) 建骨架：章节子树标题（结构）+ 召回正文（内容）→ 子查询
        all_chapters = self._book_chapters(book_titles)
        hit_chapters = [(n.metadata or {}).get("chapter", "") for n in located]
        prefix = dominant_prefix(hit_chapters)
        headings = children(all_chapters, prefix)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]
        sub_queries, mode = await self.decomposer.run(
            query, headings, passages, self.max_sub_queries
        )

        # 降级：拆不出子查询 → 整句单轮合成
        if not sub_queries:
            logger.info("split: 无子查询，降级单轮检索")
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return f"在{scope}没有检索到与「{query}」相关的内容。", []
            answer = await self._synthesize_stream(ctx, query, located)
            return answer, located

        # 综合型：扇出检索 → 去重合并 → 对原始问题一次整合合成（单子查询天然退化单轮）
        if mode == "synthesize":
            logger.info("进入整合路线")
            return await self._retrieve_and_synthesize(ctx, query, sub_queries, book_titles)

        # 罗列型：逐项检索 + map-reduce 汇总（与 assume 共用同一 helper）
        logger.info("进入罗列路线")
        sections = [(sq, sq) for sq in sub_queries]
        return await self._retrieve_and_concat(ctx, sections, book_titles)

    @staticmethod
    def _node_id(n) -> object:
        """稳定去重键：优先 NodeWithScore.node.node_id，退回 node_id，再退回对象 id。"""
        node = getattr(n, "node", None)
        return getattr(node, "node_id", None) or getattr(n, "node_id", None) or id(n)

    def _merge_pool(self, lists: list[list]) -> list:
        """多路检索结果按 node_id 去重合并，保首次出现顺序。"""
        seen: set = set()
        out: list = []
        for ns in lists:
            for n in ns:
                k = self._node_id(n)
                if k in seen:
                    continue
                seen.add(k)
                out.append(n)
        return out

    async def _retrieve_and_synthesize(
        self,
        ctx: Context,
        original_query: str,
        sub_queries: list[str],
        book_titles: Optional[list[str]],
    ) -> tuple[str, list]:
        """synthesize 模式：扇出检索（并发）→ 去重合并 → 对原始问题一次整合合成。

        子查询只为拓宽召回面；合成用【原始问题】，让 LLM 同时看到所有子项原始片段去比较/讲关系。
        """
        retrieved = await self._retrieve_all(sub_queries, book_titles)
        pool = self._merge_pool(retrieved)
        if self.reranker:
            # 拿原始问题（非子查询）对合并池重排，截到上下文预算
            pool = await self.reranker.rerank(original_query, pool, self.rerank_candidate_k)
        else:
            pool = sorted(pool, key=lambda n: getattr(n, "score", 0) or 0, reverse=True)[
                : self.rerank_candidate_k
            ]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(pool)))
        if not pool:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{original_query}」相关的内容。", []
        answer = await self._synthesize_stream(ctx, original_query, pool)
        return answer, pool

    # ── 公共流水线：逐项检索 → 一次 RetrievalDone →（可选声明）→ 逐节合成拼接 ──
    async def _retrieve_and_concat(
        self,
        ctx: Context,
        sections: list[tuple[str, str]],
        book_titles: Optional[list[str]],
        preamble: str = "",
    ) -> tuple[str, list]:
        """sections: [(分节标题, 检索/合成用子查询)]。list 模式：逐节裸拼（split / assume 共用）。

        - 扇出检索并发；逐节合成仍串行（保分节流式顺序）。
        - 先全检索（只发一次 RetrievalDone）。preamble 非空 → 答案阶段先推一个 AnswerDeltaEvent。
        """
        retrieved = await self._retrieve_all([sq for _h, sq in sections], book_titles)
        all_nodes: list = [n for ns in retrieved for n in ns]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(all_nodes)))

        parts: list[str] = []
        if preamble:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=preamble))
            parts.append(preamble)
        for (heading, sub_query), ns in zip(sections, retrieved):
            h = f"\n## {heading}\n"
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=h))
            body = (
                await self._synthesize_stream(ctx, sub_query, ns)
                if ns
                else "（未检索到相关内容）"
            )
            parts.append(h + body)
        return "".join(parts).strip(), all_nodes

    # ── helpers：章节 / 检索 / 流式合成 ──────────────────────────────
    def _book_chapters(self, book_titles: Optional[list[str]]) -> list[str]:
        """取单一选定书的去重 chapter 列表；未选或多选 → []（结构缺失，倒向内容主导）。"""
        if not book_titles or len(book_titles) != 1:
            return []
        data = self.index_manager.chroma_collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        return unique_chapters(metas, book_titles[0])

    async def _retrieve_with(self, query, book_titles, retriever, reranker):
        # 检索策略可插拔（默认 VectorRetriever=基线）；有 reranker 时过召回候选池再重排截断
        fetch_k = self.rerank_candidate_k if reranker else self.similarity_top_k
        nodes = await retriever.retrieve(
            query, index_manager=self.index_manager,
            book_titles=book_titles, top_k=fetch_k,
        )
        if reranker:
            nodes = await reranker.rerank(query, nodes, self.similarity_top_k)
        return nodes

    async def _retrieve_nodes(self, query: str, book_titles: Optional[list[str]]):
        """答案检索：用注入的 retriever/reranker。"""
        return await self._retrieve_with(
            query, book_titles, self.retriever, self.reranker
        )

    async def _probe_retrieve(self, query: str, book_titles: Optional[list[str]]):
        """probe 探测召回：用独立的 probe_retriever/probe_reranker（默认 vector / 不重排）。"""
        return await self._retrieve_with(
            query, book_titles, self.probe_retriever, self.probe_reranker
        )

    async def _retrieve_all(
        self, sub_queries: list[str], book_titles: Optional[list[str]]
    ) -> list[list]:
        """并发扇出检索：对每个子查询各检索一次，返回与入参同序的 node 列表的列表。"""
        sem = asyncio.Semaphore(self._retrieve_concurrency)

        async def _one(q: str):
            async with sem:
                return await self._retrieve_nodes(q, book_titles)

        return await asyncio.gather(*(_one(q) for q in sub_queries))

    async def _stream_tokens(self, query: str, nodes: list):
        """流式合成的 token 源。单独成方法便于单测替身。"""
        synthesizer = get_response_synthesizer(llm=self.llm, streaming=True)
        resp = await synthesizer.asynthesize(query=query, nodes=nodes)
        async for token in resp.async_response_gen():
            yield token

    async def _synthesize_stream(self, ctx: Context, query: str, nodes: list) -> str:
        """逐 token 合成：每 token 推一个 AnswerDeltaEvent，最后拼成完整答案。"""
        parts: list[str] = []
        async for token in self._stream_tokens(query, nodes):
            parts.append(token)
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=token))
        return "".join(parts)
