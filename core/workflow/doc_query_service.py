"""DocQueryWorkflow 的装配层服务封装。

把"会话并发锁 + 从 DB 历史构造记忆 + 每请求起一个 workflow"收口成一个对象，
供 api（chat / sessions 路由）与 CLI 复用。取代原 BookAgent 在装配层的位置：
顶层不再是 agent + 工具，而是 DocQueryWorkflow（门口 Router → QA 检索合成）。

- 每请求新建一个 DocQueryWorkflow（与原 book_search 工具每次新建 workflow 同构，
  workflow 轻量，index_manager 是共享引用，入库后立即可检索）。
- build_memory 鸭子类型读 .role/.content，不反向依赖 persistence 层。
"""
import asyncio
from typing import List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer

from core.workflow.doc_workflow import DocQueryWorkflow
from core.workflow.summarizer import SUMMARY_MARKER


class DocQueryService:
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        memory_token_limit: int = 4000,
        timeout: float = 300.0,
    ):
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.memory_token_limit = memory_token_limit
        self.timeout = timeout
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, session_id: Optional[str]) -> asyncio.Lock:
        """获取 session 锁；session_id 为空返回一次性锁。"""
        if not session_id:
            return asyncio.Lock()
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def build_memory(self, db_messages, summary: Optional[str] = None) -> ChatMemoryBuffer:
        """根据数据库历史消息构造 ChatMemoryBuffer（鸭子类型读 .role / .content）。

        summary 非空时，前置一条带 SUMMARY_MARKER 的消息承载被压缩掉的远期上下文；
        front_door.format_history 会据该标记【永远保留】此头部。db_messages 应只传
        【未摘要的最近消息】（已摘要的部分由 summary 代表），由装配层按水位过滤。
        """
        memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        if summary:
            memory.put(ChatMessage(
                role=MessageRole.USER, content=f"{SUMMARY_MARKER}\n{summary}"
            ))
        role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
        for m in db_messages:
            role = role_map.get(m.role)
            if role is None or not m.content:
                continue
            memory.put(ChatMessage(role=role, content=m.content))
        return memory

    def reset(self, session_id: str) -> bool:
        """清理指定 session 的并发锁（DB 数据由 sessions 路由删除）。"""
        return self._locks.pop(session_id, None) is not None

    def run_handler(
        self,
        query: str,
        memory: Optional[ChatMemoryBuffer],
        book_titles: Optional[List[str]] = None,
        allow_clarify: bool = True,
    ):
        """起一个 workflow run，返回 WorkflowHandler（可 await，可 stream_events）。"""
        workflow = DocQueryWorkflow(
            index_manager=self.index_manager,
            llm=self.llm,
            similarity_top_k=self.similarity_top_k,
            timeout=self.timeout,
        )
        return workflow.run(
            query=query,
            memory=memory,
            book_titles=book_titles,
            allow_clarify=allow_clarify,
        )

    async def ask(
        self,
        query: str,
        memory: Optional[ChatMemoryBuffer],
        book_titles: Optional[List[str]] = None,
    ) -> str:
        """单轮提问（CLI / 简单调用用）：跑完返回最终答案文本。"""
        result = await self.run_handler(query, memory, book_titles)
        return str(getattr(result, "response", result))

    async def chat(self) -> None:
        """交互式 CLI 对话（单会话内存记忆）。"""
        memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        print("=" * 50)
        print("book 知识库助手已启动（DocQueryWorkflow）")
        print("输入 'exit' 退出")
        print("=" * 50)
        while True:
            try:
                user_input = input("\n用户：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("再见！")
                break
            answer = await self.ask(user_input, memory)
            print(f"\n助手：{answer}")
