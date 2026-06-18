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
- 检索不足以回答时，如实说明缺口，不得编造或推断。

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
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.tool_selection = tool_selection
        self.ctx = ToolContext(
            index_manager=index_manager, similarity_top_k=similarity_top_k
        )
        # 懒构造：FunctionAgent 需合法 LLM 且较重，只在真要跑时才建。
        self.agent = None

    def _ensure_agent(self) -> FunctionAgent:
        if self.agent is None:
            tools, tools_prompt = assemble_tools(self.ctx, self.tool_selection)
            self.agent = FunctionAgent(
                tools=tools,
                llm=self.llm,
                system_prompt=AUTO_AGENT_SYSTEM_PROMPT.format(tools=tools_prompt),
                early_stopping_method="generate",
            )
        return self.agent

    async def run(
        self, ctx, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """跑有界 agent，桥接流式事件到外层 ctx，返回 (答案, source_nodes)。"""
        self.ctx.scope = book_titles
        self.ctx.sources = []
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
