"""文档知识库 query workflow（顶层编排版）

与 book_rag.BookRagWorkflow 的本质区别：
- 本 workflow 升为【顶层编排器】，持有并贯穿同一套会话 memory。
- 门口准入节点（front_door.FrontDoorAgent）：读会话记忆消指代 + 规范化 → clean_query，
  再四出口决策（dispatch_qa / dispatch_study_plan / converse / clarify），确定性 dispatch；
  dispatch_qa 额外产【路由计划】（`sub_queries: list[RoutedSubQuery]` + `disable_scope`）。
- QA capability（qa_capability.QaCapability，注入）：消费门口的路由计划，QA 子问题逐个
  判定（probe→admit→classify，含 per-subq scope）+ 按 category 路由到各分支【检索 + 流式
  合成】；converse 子问题直接装饰 reply，不检索。本 workflow 不再自持检索/合成实质逻辑，
  只做 step 图编排 + 薄委托。

记忆分两层（关键，别混成一锅）：
- 会话记忆：真·用户 turn + 最终答案。门口只【读】它消指代；仅在 finalize 写。
- 本轮工作态：clean_query、改写 query、category、中间产物——只走 ctx.store，
  【绝不】写进会话记忆，否则下一轮指代消解会读到污染历史。

流式：检索/合成进度通过 ctx.write_event_to_stream 推到 handler.stream_events()，
由 api 层映射成前端 SSE payload（RetrievalStart→tool_call、RetrievalDone→tool_result、
AnswerDelta→delta）。这些【流式专用事件】定义在 qa_capability，此处 re-export 供
api 层按既有路径 import；它们不参与 workflow step 图。

未完成处以 TODO 标注（study_plan 能力）。
"""
import logging
from typing import Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from core.workflow.front_door import FrontDoorAgent
from core.workflow.qa_capability import (  # noqa: F401  (事件类 re-export 供 api 层 import)
    AnswerDeltaEvent,
    EmptySkeleton,
    MissingInfo,
    OutOfScope,
    QaCapability,
    REFUSAL_FALLBACK,
    REFUSAL_TEXT,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)
from core.agent.qa_agent import QaAgent
from core.retrieval.rerank import make_reranker
from core.retrieval.retrieve import make_retriever

logger = logging.getLogger(__name__)


# ── 流程事件（驱动 workflow step 图）─────────────────────────────────
class RouteEvent(Event):
    """start → 门口 Router（净化 + 意图分类）。纯信号；query 从 ctx 取。"""


class StudyPlanEvent(Event):
    """intent=study_plan → 占位分支（v1 仅验证 dispatch 缝，能力后续实现）。"""


class DirectReplyEvent(Event):
    """converse / clarify → 门口直接回复（不检索/不分类）。"""

    reply: str
    action: str = ""


class SplitAnswerEvent(Event):
    """dispatch_qa → 多子问题拆分 + 编排作答。纯信号；clean_query 从 ctx 取。"""


class FinalizeEvent(Event):
    """各分支汇流到此，统一收尾 + 写会话记忆。"""

    answer: str
    source_nodes: list = []


class DocQueryWorkflow(Workflow):
    """文档库顶层 query 编排：route(门口) → split_answer(QA 拆分+编排作答) → finalize。"""

    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        # 可插拔检索组件（按名字注入；名字→对象在 core 解析）。与下面的布尔决策开关不同：
        # 那是二元开关，这是带多实现的具名组件。
        # reranker：检索后处理，None=基线无重排。retriever：检索数据源，None="vector"=向量基线。
        reranker: str | None = None,
        retriever: str | None = None,
        # probe（探测召回判类）检索与答案检索解耦：默认 vector + 不重排（rerank 会收敛
        # 召回、压扁 pending_split 依赖的章节 spread 信号）。eval 可显式给 probe 开 hybrid。
        probe_retriever: str | None = None,
        probe_reranker: str | None = None,
        probe_then_classify: bool = True,
        other_agent_enabled: bool = True,
        **kw,
    ):
        super().__init__(**kw)
        # 门口 Router（消指代 + 规范化 + 意图分类）与 QA capability（降噪分类 + 检索合成）
        # 各自独立、各自可测。检索/合成实质逻辑全在 qa，本 workflow 只编排 + 委托。
        self.front_door = FrontDoorAgent(llm, index_manager)
        self.qa = QaCapability(
            index_manager, llm, similarity_top_k, max_sub_queries,
            reranker=make_reranker(reranker),
            retriever=make_retriever(retriever),
            probe_retriever=make_retriever(probe_retriever),  # None → VectorRetriever
            probe_reranker=make_reranker(probe_reranker),     # None → None（不重排）
            explain_retriever=make_retriever("hybrid"),       # explain 宽覆盖召回默认 hybrid
            agent_enabled=other_agent_enabled,
        )
        self.qa_agent = QaAgent(index_manager, llm, similarity_top_k, max_iterations=6)
        # 注入有界 agent：simple 证据不足升级 / complex 自由多轮探索（见 qa._execute_subq）。
        self.qa.qa_agent = self.qa_agent
        # 决策开关（评测 ablation 用）：probe_then_classify 经 self._probe 传给 qa.answer；
        # other_agent_enabled 经 qa.agent_enabled 真正门控 _execute_subq 的有界 agent 调用点
        # （complex / simple 升级 / explain EmptySkeleton 兜底，见 qa_capability.py）。
        # 旧的拆分开关/比较开关在新 qa.answer 编排里无对应分支（拆分结构性、
        # always-on；compare 总走 assume），无意义映射，已删除。
        self._probe = probe_then_classify

    # ── 入口：把 memory + 原始 query + scope 全塞进 ctx，贯穿全程 ──
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> RouteEvent:
        # memory 是调用方（API handler）从 DB 重建的 ChatMemoryBuffer
        await ctx.store.set("memory", getattr(ev, "memory", None))
        await ctx.store.set("original_query", ev.query)  # 工作态：用户原话
        await ctx.store.set("book_titles", getattr(ev, "book_titles", None))
        await ctx.store.set("allow_clarify", getattr(ev, "allow_clarify", True))
        return RouteEvent()

    # ── 门口准入决策：读会话记忆做净化 + 四出口决策，确定性 dispatch。 ──
    @step
    async def route(
        self, ctx: Context, ev: RouteEvent
    ) -> "SplitAnswerEvent | StudyPlanEvent | DirectReplyEvent":
        original = await ctx.store.get("original_query")
        memory: Optional[ChatMemoryBuffer] = await ctx.store.get("memory")
        book_titles = await ctx.store.get("book_titles")

        decision = await self.front_door.run(original, memory, book_titles)

        # 工作态落 ctx：action 供观测；clean_query 是门口横切产物，绝不写会话记忆。
        await ctx.store.set("action", decision.action)

        if decision.action == "dispatch_study_plan":
            await ctx.store.set("clean_query", decision.clean_query)
            return StudyPlanEvent()
        if decision.action in ("converse", "clarify"):
            return DirectReplyEvent(reply=decision.reply, action=decision.action)
        # dispatch_qa（含降级）—— memory/book_titles 在 route 顶部已取
        await ctx.store.set("clean_query", decision.clean_query)
        await ctx.store.set("sub_queries", decision.sub_queries)
        await ctx.store.set("disable_scope", decision.disable_scope)
        return SplitAnswerEvent()

    # ── 分支：dispatch 到 QA capability（薄委托），各分支统一收成 FinalizeEvent ──
    @step
    async def study_plan_branch(self, ctx: Context, ev: StudyPlanEvent) -> FinalizeEvent:
        # TODO: 接入 StudyPlan capability（拆解→排序→渲染，产 plan 产物落 DB）。
        return FinalizeEvent(
            answer="学习计划能力还在建设中，目前仅支持文档问答。", source_nodes=[]
        )

    @step
    async def direct_reply_branch(
        self, ctx: Context, ev: DirectReplyEvent
    ) -> FinalizeEvent:
        # converse/clarify：门口已生成面向用户的回复，直接收尾，不进 probe/检索。
        return FinalizeEvent(answer=ev.reply, source_nodes=[])

    @step
    async def split_answer(self, ctx: Context, ev: SplitAnswerEvent) -> FinalizeEvent:
        # dispatch_qa → 委托 QA capability 统一编排：消费门口路由计划 + 判定 + 按序执行 + 合并装饰。
        sub_queries = await ctx.store.get("sub_queries")
        book_titles = await ctx.store.get("book_titles")
        disable_scope = await ctx.store.get("disable_scope", False)
        answer, nodes, meta = await self.qa.answer(
            ctx, sub_queries, book_titles, probe=self._probe, disable_scope=disable_scope
        )
        await ctx.store.set("qa_meta", meta)
        return FinalizeEvent(answer=answer, source_nodes=nodes)

    # ── 收尾：唯一写「会话记忆」的地方 = 原始问题 + 最终答案 ──────────
    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> StopEvent:
        memory: Optional[ChatMemoryBuffer] = await ctx.store.get("memory")
        original = await ctx.store.get("original_query")
        if memory is not None:
            # 存【用户原话】，不存改写版：保真，且下轮消指代不被机器措辞带偏
            memory.put(ChatMessage(role=MessageRole.USER, content=original))
            memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=ev.answer))
        logger.info(
            "finalize: answer_len=%d source_nodes=%d",
            len(ev.answer or ""), len(ev.source_nodes or []),
        )
        # meta 以 qa.answer 产出的 qa_meta（category/categories/sub_count）为主，叠加 action；
        # 非 qa 路径（study_plan/converse/clarify）qa_meta 为空，meta 退化为仅 {"action": ...}。
        qa_meta = await ctx.store.get("qa_meta", {}) or {}
        meta = {**qa_meta, "action": await ctx.store.get("action", None)}
        return StopEvent(
            result=Response(
                response=ev.answer, source_nodes=ev.source_nodes, metadata=meta
            )
        )
