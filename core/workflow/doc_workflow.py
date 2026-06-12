"""文档知识库 query workflow（顶层编排版）

与 book_rag.BookRagWorkflow 的本质区别：
- 本 workflow 升为【顶层编排器】，持有并贯穿同一套会话 memory。
- 门口 Router（intent_router）：读会话记忆消指代 + 规范化 → clean_query，再意图
  分类 → qa / study_plan / ...，确定性 dispatch。横切的"干净自包含 query"在此产出。
- QA 内部 preprocess：拿 clean_query 做【降噪 + 难度分类】（检索专属，不再消指代），
  按 category 路由到各分支。
- 分支【直接检索 + 流式合成】（绕开 agent/工具）：本 workflow 自持 index_manager，
  retrieve → synthesize 一气呵成，source_nodes 直接随结果带出。

记忆分两层（关键，别混成一锅）：
- 会话记忆：真·用户 turn + 最终答案。门口只【读】它消指代；仅在 finalize 写。
- 本轮工作态：clean_query、改写 query、category、中间产物——只走 ctx.store，
  【绝不】写进会话记忆，否则下一轮指代消解会读到污染历史。

流式：检索/合成进度通过 ctx.write_event_to_stream 推到 handler.stream_events()，
由 api 层映射成前端 SSE payload（RetrievalStart→tool_call、RetrievalDone→tool_result、
AnswerDelta→delta）。这些是【流式专用事件】，不参与 workflow step 图。

未完成处以 TODO 标注（split/assume 分支真实逻辑、study_plan 能力）。
"""
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from core.workflow.chapter_tree import children, dominant_prefix, unique_chapters
from core.workflow.intent_router import IntentRouter
from core.workflow.query_decompose import QueryDecomposer
from core.workflow.query_preprocess import QueryPreprocessor


# ── 流程事件（驱动 workflow step 图）─────────────────────────────────
class RouteEvent(Event):
    """start → 门口 Router（净化 + 意图分类）。纯信号；query 从 ctx 取。"""


class StudyPlanEvent(Event):
    """intent=study_plan → 占位分支（v1 仅验证 dispatch 缝，能力后续实现）。"""


class PreprocessEvent(Event):
    """intent=qa → QA 内部预处理（降噪 + 难度分类）。纯信号；clean_query 从 ctx 取。"""


class RetrieveAgentEvent(Event):
    """retrievable / other / 降级 → 直接检索 + 合成。"""

    rewritten_query: str


class SplitEvent(Event):
    """pending_split → 拆分（v1 先按整句直接检索）。"""

    rewritten_query: str
    split_reason: str = ""


class AssumeEvent(Event):
    """ambiguous → 先声明所选角度再回答（v1 先按整句直接检索）。"""

    rewritten_query: str


class ClarifyEvent(Event):
    """missing_info → 反问用户，本轮终止等补充。"""

    rewritten_query: str
    clarify_reason: str = ""


class FinalizeEvent(Event):
    """各分支汇流到此，统一收尾 + 写会话记忆。"""

    answer: str
    source_nodes: list = []


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


class DocQueryWorkflow(Workflow):
    """文档库顶层 query 编排：route(门口) → preprocess(QA 降噪+难度) → 分支检索合成 → finalize。"""

    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        **kw,
    ):
        super().__init__(**kw)
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_sub_queries = max_sub_queries
        # 门口 Router、QA 预处理、拆解器各自独立、各自可测。
        self.router = IntentRouter(llm)
        self.preprocessor = QueryPreprocessor(llm)
        self.decomposer = QueryDecomposer(llm)

    # ── 入口：把 memory + 原始 query + scope 全塞进 ctx，贯穿全程 ──
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> RouteEvent:
        # memory 是调用方（API handler）从 DB 重建的 ChatMemoryBuffer
        await ctx.store.set("memory", getattr(ev, "memory", None))
        await ctx.store.set("original_query", ev.query)  # 工作态：用户原话
        await ctx.store.set("book_titles", getattr(ev, "book_titles", None))
        await ctx.store.set("allow_clarify", getattr(ev, "allow_clarify", True))
        return RouteEvent()

    # ── 门口 Router：读会话记忆消指代 + 规范化 → clean_query，再意图分类。 ──
    @step
    async def route(
        self, ctx: Context, ev: RouteEvent
    ) -> "PreprocessEvent | StudyPlanEvent":
        original = await ctx.store.get("original_query")
        memory: Optional[ChatMemoryBuffer] = await ctx.store.get("memory")
        book_titles = await ctx.store.get("book_titles")

        # 选中的书一并喂给门口，用于消解"这本书/本书"类指代
        result = await self.router.run(original, memory, book_titles)

        # 工作态落 ctx：clean_query 是门口的横切产物；【绝不】写进会话记忆。
        await ctx.store.set("clean_query", result.clean_query)
        await ctx.store.set("intent", result.intent)

        if result.intent == "study_plan":
            return StudyPlanEvent()
        return PreprocessEvent()  # qa（含降级）

    # ── QA 内部预处理：拿 clean_query 做降噪 + 难度分类。不再消指代。 ──
    @step
    async def preprocess(
        self, ctx: Context, ev: PreprocessEvent
    ) -> "RetrieveAgentEvent | SplitEvent | AssumeEvent | ClarifyEvent":
        clean_query = await ctx.store.get("clean_query")

        result = await self.preprocessor.run(clean_query)

        await ctx.store.set("rewritten_query", result.rewritten_query)
        await ctx.store.set("category", result.category)

        rewritten = result.rewritten_query
        match result.category:
            case "pending_split":
                return SplitEvent(rewritten_query=rewritten, split_reason=result.reason)
            case "ambiguous":
                return AssumeEvent(rewritten_query=rewritten)
            case "missing_info":
                if await ctx.store.get("allow_clarify"):
                    return ClarifyEvent(
                        rewritten_query=rewritten, clarify_reason=result.reason
                    )
                return RetrieveAgentEvent(rewritten_query=rewritten)  # 预算耗尽降级
            case _:  # retrievable / other / 解析失败 fallback
                return RetrieveAgentEvent(rewritten_query=rewritten)

    # ── 分支：QA 各类直接检索 + 流式合成（v1 拆分/角度逻辑留 TODO）──────
    @step
    async def study_plan_branch(self, ctx: Context, ev: StudyPlanEvent) -> FinalizeEvent:
        # TODO: 接入 StudyPlan capability（拆解→排序→渲染，产 plan 产物落 DB）。
        return FinalizeEvent(
            answer="学习计划能力还在建设中，目前仅支持文档问答。", source_nodes=[]
        )

    @step
    async def retrieve_branch(self, ctx: Context, ev: RetrieveAgentEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        answer, nodes = await self._answer(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)

    @step
    async def split_branch(self, ctx: Context, ev: SplitEvent) -> FinalizeEvent:
        """定位 → 建骨架(结构主+内容辅) → 逐项检索 → map-reduce 汇总。

        拆解失败/空 → 降级为单轮检索+合成（等同 retrieve_branch），绝不阻塞。
        """
        book_titles = await ctx.store.get("book_titles")
        query = ev.rewritten_query

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
        sub_queries = await self.decomposer.run(
            query, headings, passages, self.max_sub_queries
        )

        # 降级：拆不出子查询 → 整句单轮合成
        if not sub_queries:
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return FinalizeEvent(
                    answer=f"在{scope}没有检索到与「{query}」相关的内容。",
                    source_nodes=[],
                )
            answer = await self._synthesize_stream(ctx, query, located)
            return FinalizeEvent(answer=answer, source_nodes=located)

        # 3) 逐项检索（先全检索，便于只发一次 RetrievalDone）
        sections: list[tuple[str, list]] = []
        all_nodes: list = []
        for sq in sub_queries:
            ns = await self._retrieve_nodes(sq, book_titles)
            sections.append((sq, ns))
            all_nodes.extend(ns)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(all_nodes)))

        # 4) 汇总（map-reduce）：每子项各自合成一段，按骨架拼接
        parts: list[str] = []
        for sq, ns in sections:
            heading = f"\n## {sq}\n"
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=heading))
            body = (
                await self._synthesize_stream(ctx, sq, ns)
                if ns
                else "（未检索到相关内容）"
            )
            parts.append(heading + body)
        answer = "".join(parts).strip()
        return FinalizeEvent(answer=answer, source_nodes=all_nodes)

    @step
    async def assume_branch(self, ctx: Context, ev: AssumeEvent) -> FinalizeEvent:
        # TODO: 先让回答声明所选角度（写进答案开头）；v1 先按整句直接检索。
        book_titles = await ctx.store.get("book_titles")
        answer, nodes = await self._answer(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)

    # ── 反问：本轮终止，把反问句作为答案，由 finalize 写回记忆等用户补充 ──
    @step
    async def clarify_branch(self, ctx: Context, ev: ClarifyEvent) -> FinalizeEvent:
        question = f"为了更准确地回答，请补充：{ev.clarify_reason}"
        # 反问句经 finalize 作为 assistant turn 进会话记忆，
        # 下一轮门口才能同时看到「原问题 + 反问 + 用户补充」一起消解。
        return FinalizeEvent(answer=question, source_nodes=[])

    # ── 收尾：唯一写「会话记忆」的地方 = 原始问题 + 最终答案 ──────────
    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> StopEvent:
        memory: Optional[ChatMemoryBuffer] = await ctx.store.get("memory")
        original = await ctx.store.get("original_query")
        if memory is not None:
            # 存【用户原话】，不存改写版：保真，且下轮消指代不被机器措辞带偏
            memory.put(ChatMessage(role=MessageRole.USER, content=original))
            memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=ev.answer))
        return StopEvent(result=Response(response=ev.answer, source_nodes=ev.source_nodes))

    # ── helpers：检索 + 流式合成 ─────────────────────────────────────
    def _book_chapters(self, book_titles: Optional[list[str]]) -> list[str]:
        """取单一选定书的去重 chapter 列表；未选或多选 → []（结构缺失，倒向内容主导）。"""
        if not book_titles or len(book_titles) != 1:
            return []
        data = self.index_manager.chroma_collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        return unique_chapters(metas, book_titles[0])

    def _make_filters(self, book_titles: Optional[list[str]]):
        """scope 硬约束转 metadata 过滤器；空范围返回 None（全库）。"""
        if not book_titles:
            return None
        return MetadataFilters(filters=[
            MetadataFilter(
                key="book_title",
                operator=FilterOperator.IN,
                value=list(book_titles),
            ),
        ])

    async def _retrieve_nodes(self, query: str, book_titles: Optional[list[str]]):
        index = self.index_manager.get_index()
        retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=self._make_filters(book_titles),
        )
        return await retriever.aretrieve(query)

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

    async def _answer(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """直接检索 + 流式合成（绕开 agent/工具）。返回 (答案文本, source_nodes)。"""
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        nodes = await self._retrieve_nodes(query, book_titles)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(nodes)))
        if not nodes:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。", []
        answer = await self._synthesize_stream(ctx, query, nodes)
        return answer, nodes
