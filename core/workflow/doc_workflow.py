"""文档知识库 query workflow（顶层编排版）

与 book_rag.BookRagWorkflow 的本质区别：
- 本 workflow 升为【顶层编排器】，持有并贯穿同一套会话 memory。
- 门口准入节点（front_door.FrontDoorAgent）：读会话记忆消指代 + 规范化 → clean_query，
  再四出口决策（dispatch_qa / dispatch_study_plan / converse / clarify），确定性 dispatch。
  横切的"干净自包含 query"在此产出。
- QA capability（qa_capability.QaCapability，注入）：拿 clean_query 做【降噪 + 难度
  分类】（检索专属，不再消指代），按 category 路由到各分支【检索 + 流式合成】。
  本 workflow 不再自持检索/合成实质逻辑，只做 step 图编排 + 薄委托。

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
    QaCapability,
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


class OutOfScopeEvent(Event):
    """out_of_scope → 库外问题（探测召回片段与主题无关）。如实告知，不检索/不反问。"""


class PreprocessEvent(Event):
    """intent=qa → QA 内部预处理（降噪 + 难度分类）。纯信号；clean_query 从 ctx 取。"""


class RetrieveAgentEvent(Event):
    """retrievable / other / 降级 → 直接检索 + 合成。

    assumption_note 非空（missing_info 预算耗尽降级）→ 答案前声明所作假设。
    """

    rewritten_query: str
    assumption_note: str = ""


class SplitEvent(Event):
    """pending_split → 拆解-检索-汇总。"""

    rewritten_query: str
    split_reason: str = ""


class AssumeEvent(Event):
    """ambiguous → 归纳评判维度 + 声明所选角度后逐维度回答。"""

    rewritten_query: str


class ClarifyEvent(Event):
    """missing_info → 反问用户，本轮终止等补充。"""

    rewritten_query: str
    clarify_reason: str = ""
    clarify_question: str = ""


class OtherEvent(Event):
    """other → 高难度/开放问题分支。

    第二步将换为「有界 agent（自由调用工具 + 成本/次数边界 + 超界降级）」；
    v1 先从 fallback 独立出来、暂走单轮检索。
    """

    rewritten_query: str


class FinalizeEvent(Event):
    """各分支汇流到此，统一收尾 + 写会话记忆。"""

    answer: str
    source_nodes: list = []


class DocQueryWorkflow(Workflow):
    """文档库顶层 query 编排：route(门口) → preprocess(QA 降噪+难度) → 分支检索合成 → finalize。"""

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
        split_enabled: bool = True,
        assume_enabled: bool = True,
        other_agent_enabled: bool = True,
        **kw,
    ):
        super().__init__(**kw)
        # 门口 Router（消指代 + 规范化 + 意图分类）与 QA capability（降噪分类 + 检索合成）
        # 各自独立、各自可测。检索/合成实质逻辑全在 qa，本 workflow 只编排 + 委托。
        self.front_door = FrontDoorAgent(llm)
        self.qa = QaCapability(
            index_manager, llm, similarity_top_k, max_sub_queries,
            reranker=make_reranker(reranker),
            retriever=make_retriever(retriever),
            probe_retriever=make_retriever(probe_retriever),  # None → VectorRetriever
            probe_reranker=make_reranker(probe_reranker),     # None → None（不重排）
        )
        self.qa_agent = QaAgent(index_manager, llm, similarity_top_k, max_iterations=6)
        # 决策开关（评测 ablation 用；off → 对应分支降级单轮 retrieve、probe 关闭）
        self._probe = probe_then_classify
        self._split_enabled = split_enabled
        self._assume_enabled = assume_enabled
        self._other_agent_enabled = other_agent_enabled

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
    ) -> "PreprocessEvent | StudyPlanEvent | DirectReplyEvent":
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
        # dispatch_qa（含降级）
        await ctx.store.set("clean_query", decision.clean_query)
        return PreprocessEvent()

    # ── QA 内部预处理：委托 QA capability 做降噪 + 难度分类，据 category dispatch。 ──
    @step
    async def preprocess(
        self, ctx: Context, ev: PreprocessEvent
    ) -> "RetrieveAgentEvent | SplitEvent | AssumeEvent | ClarifyEvent | OtherEvent | OutOfScopeEvent":
        clean_query = await ctx.store.get("clean_query")
        book_titles = await ctx.store.get("book_titles")

        result = await self.qa.classify(clean_query, book_titles, probe=self._probe)

        await ctx.store.set("rewritten_query", result.rewritten_query)
        await ctx.store.set("category", result.category)

        rewritten = result.rewritten_query
        match result.category:
            case "out_of_scope":
                return OutOfScopeEvent()
            case "pending_split":
                return SplitEvent(rewritten_query=rewritten, split_reason=result.reason)
            case "ambiguous":
                return AssumeEvent(rewritten_query=rewritten)
            case "missing_info":
                if await ctx.store.get("allow_clarify"):
                    return ClarifyEvent(
                        rewritten_query=rewritten,
                        clarify_reason=result.reason,
                        clarify_question=result.clarify_question,
                    )
                # 预算耗尽降级：不反问，声明假设、尽力答
                note = (
                    f"（注：原问题信息不足（{result.reason}），"
                    f"以下按最可能的解读作答。）\n"
                )
                return RetrieveAgentEvent(
                    rewritten_query=rewritten, assumption_note=note
                )
            case "other":
                return OtherEvent(rewritten_query=rewritten)
            case _:  # retrievable / 解析失败 fallback
                return RetrieveAgentEvent(rewritten_query=rewritten)

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
    async def out_of_scope_branch(self, ctx: Context, ev: OutOfScopeEvent) -> FinalizeEvent:
        # 库外：探测召回片段与问题主题无关 → 对话式转场，友好告知 + 邀请换问法，不检索/不反问。
        return FinalizeEvent(
            answer=(
                "这个问题知识库里暂未收录相关内容，我没法基于现有资料回答。"
                "你可以换个已入库主题问我，或把问题换个角度再试试～"
            ),
            source_nodes=[],
        )

    @step
    async def retrieve_branch(self, ctx: Context, ev: RetrieveAgentEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        answer, nodes = await self.qa.retrieve(
            ctx, ev.rewritten_query, book_titles, ev.assumption_note
        )
        return FinalizeEvent(answer=answer, source_nodes=nodes)

    @step
    async def other_branch(self, ctx: Context, ev: OtherEvent) -> FinalizeEvent:
        """高难度/开放问题 → 有界 agent 自由多轮检索探索。

        agent 异常 → 降级单轮检索，绝不让 other 比单轮更脆。
        """
        book_titles = await ctx.store.get("book_titles")
        if not self._other_agent_enabled:
            answer, nodes = await self.qa.retrieve(ctx, ev.rewritten_query, book_titles)
            return FinalizeEvent(answer=answer, source_nodes=nodes)
        try:
            answer, nodes = await self.qa_agent.run(
                ctx, ev.rewritten_query, book_titles
            )
        except Exception as exc:
            logger.warning("other agent 失败，降级单轮检索：%s", exc)
            answer, nodes = await self.qa.retrieve(
                ctx, ev.rewritten_query, book_titles
            )
        return FinalizeEvent(answer=answer, source_nodes=nodes)

    @step
    async def split_branch(self, ctx: Context, ev: SplitEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        if self._split_enabled:
            answer, nodes = await self.qa.split(ctx, ev.rewritten_query, book_titles)
        else:
            answer, nodes = await self.qa.retrieve(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)

    @step
    async def assume_branch(self, ctx: Context, ev: AssumeEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        if self._assume_enabled:
            answer, nodes = await self.qa.assume(ctx, ev.rewritten_query, book_titles)
        else:
            answer, nodes = await self.qa.retrieve(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)

    # ── 反问：本轮终止，把反问句作为答案，由 finalize 写回记忆等用户补充 ──
    @step
    async def clarify_branch(self, ctx: Context, ev: ClarifyEvent) -> FinalizeEvent:
        # 优先用 LLM 产出的自然反问句；缺失则退回模板拼 reason（绝不阻塞）
        question = ev.clarify_question or f"为了更准确地回答，请补充：{ev.clarify_reason}"
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
        logger.info(
            "finalize: answer_len=%d source_nodes=%d",
            len(ev.answer or ""), len(ev.source_nodes or []),
        )
        # 把 category/intent 附到结果 metadata，供评测算分类准确率/分支分布（api 不受影响）
        meta = {
            "category": await ctx.store.get("category", None),
            "action": await ctx.store.get("action", None),
        }
        return StopEvent(
            result=Response(
                response=ev.answer, source_nodes=ev.source_nodes, metadata=meta
            )
        )
