"""QaAgent：other（高难度/开放问题）分支的有界 agent。

DocQueryWorkflow 把 intent=qa & category=other dispatch 到这里。用 LlamaIndex
FunctionAgent 让 LLM 自由多轮调用工具（检索器）探索，设步数边界与超界强制作答，
避免失控（见 ARCHITECTURE.md §2「按可预测性配控制结构」）。

- 工具是检索器：book_search 返回原文片段、list_books 返回书单，agent 多轮综合。
- 边界：max_iterations + early_stopping_method="generate"（超界基于已收集结果作答）。
- 流式：把 agent 的 ToolCall/ToolCallResult 转译成项目既有的 RetrievalStart/Done
  事件推到外层 ctx（前端零改动）；中间 thought 不外露；最终答案推一个 AnswerDelta。
- grounding：system prompt 强约束只基于检索片段；source_nodes 由工具收集回传。
- 每请求随 DocQueryWorkflow 新建，故可用实例变量持 per-run scope/sources（无并发）。
"""
import logging
from typing import Optional

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from core.retrieval.retrieve import build_book_filters
from core.workflow.qa_capability import (
    AnswerDeltaEvent,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)

logger = logging.getLogger(__name__)

QA_AGENT_SYSTEM_PROMPT = """你是技术书籍知识库的高难度问答 agent，处理需要多步推理、跨主题综合或开放权衡的复杂问题。

铁律（grounding）：
- 【先检索再答】拿到问题必须先调用 book_search 检索，严禁在检索前就用训练知识猜测含义、或直接反问用户"你是指什么"。即便问题里有你不认识的词，也要先检索——它很可能就是知识库里的专有名词。
- 只能基于 book_search 返回的检索片段作答，严禁用你自己的训练知识或常识脑补事实。
- 复杂问题可多次调用 book_search（换关键词/换角度）逐步收集证据，再综合。
- 检索不足以回答时，如实说明缺口，不得编造或推断。

工具：
1. book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。
2. list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。

回答：中文，结构清晰，必要时引用书名/章节；先给结论再展开。"""


class QaAgent:
    """other 分支的有界 agent：FunctionAgent + 检索工具 + 流式桥接 + source 收集。"""

    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_iterations: int = 6,
    ):
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_iterations = max_iterations
        self._run_scope: Optional[list[str]] = None
        self._run_sources: list = []
        # 懒构造：FunctionAgent 需合法 LLM 且较重，只在真走 other 分支时才建。
        # 这样 DocQueryWorkflow 每请求构造（含单测替身 LLM）不被 FunctionAgent 校验拖累，
        # 多数不走 other 的请求也省下构造开销。
        self.agent = None

    def _ensure_agent(self) -> FunctionAgent:
        if self.agent is None:
            self.agent = FunctionAgent(
                tools=self._make_tools(),
                llm=self.llm,
                system_prompt=QA_AGENT_SYSTEM_PROMPT,
                early_stopping_method="generate",
            )
        return self.agent

    async def _search(self, query: str) -> str:
        """检索器主体：按 query 取 top-k 原文片段并收集 nodes。供 book_search 工具调用。"""
        if not isinstance(query, str):
            query = str(query)
        query = query.strip()
        if not query:
            return "请提供要检索的问题。"
        index = self.index_manager.get_index()
        if index is None:
            return "知识库为空，请先上传 PDF。"
        retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=build_book_filters(self._run_scope),
        )
        nodes = await retriever.aretrieve(query)
        if not nodes:
            return "（未检索到相关内容）"
        self._run_sources.extend(nodes)
        return "\n---\n".join(
            (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:500]
            for n in nodes
        )

    def _make_tools(self) -> list:
        async def book_search(query: str) -> str:
            """在书籍知识库中检索与 query 相关的原文片段。

            Args:
                query: 检索问题（字符串）。
            Returns:
                拼接的检索片段；无命中返回占位提示。
            """
            return await self._search(query)

        def list_books() -> str:
            """列出当前知识库已入库的书籍清单。"""
            data = self.index_manager.chroma_collection.get(include=["metadatas"])
            counts: dict[str, int] = {}
            for meta in data.get("metadatas", []) or []:
                title = (meta or {}).get("book_title")
                if not title:
                    continue
                counts[title] = counts.get(title, 0) + 1
            if not counts:
                return "知识库当前为空。"
            return "已入库书籍：\n" + "\n".join(
                f"- 《{t}》（{c} 块）" for t, c in sorted(counts.items())
            )

        return [
            FunctionTool.from_defaults(
                fn=book_search,
                name="book_search",
                description="书籍知识库检索：按 query 返回相关原文片段，范围由用户选定。",
            ),
            FunctionTool.from_defaults(
                fn=list_books,
                name="list_books",
                description="列出当前已入库书籍清单。",
            ),
        ]

    async def run(
        self, ctx, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """跑有界 agent，桥接流式事件到外层 ctx，返回 (答案, source_nodes)。

        agent 异常由调用方（other_branch）降级单轮检索。
        """
        self._run_scope = book_titles
        self._run_sources = []
        logger.info(
            "qa_agent 启动: query=%r max_iter=%d", query[:80], self.max_iterations
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
                    "qa_agent tool_call: %s(%r)",
                    getattr(ev, "tool_name", "?"), tq[:60],
                )
                ctx.write_event_to_stream(RetrievalStartEvent(query=tq))
            elif name == "ToolCallResult":
                ctx.write_event_to_stream(
                    RetrievalDoneEvent(count=len(self._run_sources))
                )
        final = await handler
        answer = str(final)
        logger.info("qa_agent 完成: %d sources", len(self._run_sources))
        ctx.write_event_to_stream(AnswerDeltaEvent(delta=answer))
        return answer, list(self._run_sources)
