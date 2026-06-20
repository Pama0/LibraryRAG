"""AutoAgent：评测用的自主规划问答 agent。

eval 侧用它跑 RAG 效果对照——绕过 DocQueryWorkflow 的决策路由，直接让有界
FunctionAgent 自由多轮调用检索工具，与 workflow 路线比对效果。结构镜像 QaAgent：
ToolContext 收口检索依赖与 per-run scope/sources；assemble_tools 按注册表动态产出
工具与 system prompt 的工具清单（不写死）；tool_selection 可按需选工具子集/覆盖 usage。

- 边界：max_iterations + early_stopping_method="generate"（超界基于已收集结果作答）。
- 流式：ToolCall/ToolCallResult 桥接成项目既有 Retrieval 事件（eval 传 no-op ctx）。
- grounding：system prompt 强约束只基于检索片段；source_nodes 由工具收集回传。
"""
import logging
from typing import Optional

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from core.agent.tools import ToolContext, assemble_tools
from core.retrieval.rerank import make_reranker
from core.retrieval.retrieve import make_retriever
from core.workflow.qa_capability import (
    AnswerDeltaEvent,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)

logger = logging.getLogger(__name__)

AUTO_AGENT_SYSTEM_PROMPT = """你是技术书籍知识库的问答 agent，你需要处理用户提出的问题。

铁律（grounding）：
- 【先检索再答】拿到问题必须先调用 book_search 检索，严禁在检索前就用训练知识猜测含义、或直接反问用户"你是指什么"。即便问题里有你不认识的词，也要先检索——它很可能就是知识库里的专有名词。
- 只能基于检索片段作答，严禁用你自己的训练知识或常识脑补事实。
- 复杂问题可多次调用 book_search（换关键词/换角度）逐步收集证据，再综合。
- 【复杂问题先分解】遇到对比、多跳、含多个子项的问题，先把它拆成若干子问题，逐个 book_search 收集证据，最后综合，不要一次检索就下结论。
- 【需要时定向到某本书】跨书对比时，对每本书分别用 book=书名 检索并各自归属；想在某一本里深挖也用 book 收窄。不确定有哪些书时先 list_books。

检索之后，按以下顺序三选一收场（判断一律基于召回片段，不基于你是否认识问题中的词）：
1. 【拒答·库外】若召回片段与问题的主体技术实体明显不相关（即知识库里没有该主题），就如实告知："知识库里暂无与该问题相关的内容。"——不要编造、不要用训练知识硬答、不要反问。
   判据以问题的主体技术实体为准：若问题问的是 A 系统/产品，而召回片段讲的全是另一套系统 B，即便字面共享通用术语（"一致性""分片""架构""事务"等）也属不相关 → 拒答。
2. 【反问·指代不明】若召回到了与问题相关的主题，但问题里有指代不明或缺关键限定、无法定位到具体所指，就用一句自然的反问点明不明之处并引导补充（能列候选就列），不要硬猜作答。
3. 【正常作答】以上都不是 → 基于召回片段作答；若片段只够部分回答，如实说明缺口，不得编造或推断。

工具：
{tools}

回答：中文，结构清晰，必要时引用书名/章节；先给结论再展开。"""


class AutoAgent:
    """评测用自主规划 agent：FunctionAgent + 检索工具 + 流式桥接 + source 收集。"""

    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_iterations: int = 6,
        tool_selection: Optional[list] = None,
        retriever_name: str = "hybrid",
        reranker_name: Optional[str] = "bge-reranker-v2-m3",
        rerank_candidate_k: int = 20,
        timeout: float = 120.0,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.timeout = timeout  # 整轮 wall-clock 上限，挂死时 fail-fast（异常交调用方处理）
        self.tool_selection = tool_selection
        # 默认装更强的检索组合（hybrid + bge 重排），让自主规划 agent 拿到更好证据；
        # 名字→对象的解析推迟到 _ensure_agent（首跑），避免构造期就加载 reranker 模型。
        self._retriever_name = retriever_name
        self._reranker_name = reranker_name
        self.ctx = ToolContext(
            index_manager=index_manager,
            similarity_top_k=similarity_top_k,
            rerank_candidate_k=rerank_candidate_k,
        )
        # 懒构造：FunctionAgent 需合法 LLM 且较重，只在真要跑时才建。
        self.agent = None

    def _ensure_agent(self) -> FunctionAgent:
        if self.agent is None:
            self.ctx.retriever = make_retriever(self._retriever_name)
            self.ctx.reranker = make_reranker(self._reranker_name)
            tools, tools_prompt = assemble_tools(self.ctx, self.tool_selection)
            self.agent = FunctionAgent(
                tools=tools,
                llm=self.llm,
                system_prompt=AUTO_AGENT_SYSTEM_PROMPT.format(tools=tools_prompt),
                early_stopping_method="generate",
                timeout=self.timeout,
            )
        return self.agent

    async def run(
        self, ctx, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """跑有界 agent，桥接流式事件到外层 ctx，返回 (答案, source_nodes)。"""
        self.ctx.scope = book_titles
        self.ctx.sources = []
        self.ctx.searched_queries = set()
        logger.info(
            "auto_agent 启动: query=%r max_iter=%d", query[:80], self.max_iterations
        )

        handler = self._ensure_agent().run(
            user_msg=query, max_iterations=self.max_iterations
        )
        async for ev in handler.stream_events():
            name = ev.__class__.__name__
            if name == "ToolCall":
                tq = (
                    ev.tool_kwargs.get("query", query)
                    if getattr(ev, "tool_name", "") == "book_search"
                    else query
                )
                logger.info(
                    "auto_agent tool_call: %s(%r)",
                    getattr(ev, "tool_name", "?"), tq[:60],
                )
                ctx.write_event_to_stream(RetrievalStartEvent(query=tq))
            elif name == "ToolCallResult":
                ctx.write_event_to_stream(
                    RetrievalDoneEvent(count=len(self.ctx.sources))
                )
        final = await handler
        answer = str(final)
        logger.info("auto_agent 完成: %d sources", len(self.ctx.sources))
        ctx.write_event_to_stream(AnswerDeltaEvent(delta=answer))
        return answer, list(self.ctx.sources)
