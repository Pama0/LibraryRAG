"""主 Agent（book 知识库助手）

- 基于 LlamaIndex FunctionAgent（支持流式事件，前端依赖）
- per-session 并发锁，防同会话并发互踩
- 从 DB 历史构造 ChatMemoryBuffer（鸭子类型读 .role/.content，不反向依赖 persistence）
- 保留交互式 CLI chat() 便捷入口
"""
import asyncio
from typing import List, Optional

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool


BOOK_SYSTEM_PROMPT = """你是一个书籍知识库助手，帮助用户从已入库的技术书籍中查找答案。

可用工具：
1. book_search(query, book_title?) — 在书籍知识库中检索内容。
   - 用户问具体技术问题时必须先调用，不要凭记忆作答。
   - 如果用户指明了书名（如"在《MySQL是怎样运行的》里..."），把书名传给 book_title 参数。
   - 留空 book_title 则跨所有书检索。
2. list_books() — 列出当前已入库的书籍清单。
   - 用户问"有哪些书"、"知识库里有什么"时调用。
   - 当 book_search 返回"没有检索到相关内容"时，可调用此工具帮助用户了解可选范围。

回答规则：
- 答案必须基于检索结果。检索为空就如实告知，不要编造。
- 中文回答，简洁清楚，必要时引用书名/章节。
- 不要重复调用同一个工具，除非确实需要换关键词或换书重试。
- 调用 book_search 前：若用户问题含指代词（它/这个/上面说的/前面提到的），先根据会话历史把 query 改写为不依赖上文、能独立成立的句子，再传入 query 参数。"""


class BookAgent:
    """统一主 agent：一个全局 FunctionAgent + per-session 并发锁。"""

    def __init__(
        self,
        tools: List[FunctionTool],
        llm: LLM,
        system_prompt: str = BOOK_SYSTEM_PROMPT,
        memory_token_limit: int = 4000,
        timeout: float = 300.0,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.memory_token_limit = memory_token_limit

        self.agent = FunctionAgent(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            timeout=timeout,
            verbose=True,
        )
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, session_id: Optional[str]) -> asyncio.Lock:
        """获取 session 锁；session_id 为空返回一次性锁"""
        if not session_id:
            return asyncio.Lock()
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def build_memory(self, db_messages) -> ChatMemoryBuffer:
        """根据数据库历史消息构造 ChatMemoryBuffer。

        db_messages: 按 id 升序的消息行，鸭子类型读取 .role / .content。
        """
        memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
        for m in db_messages:
            role = role_map.get(m.role)
            if role is None or not m.content:
                continue
            memory.put(ChatMessage(role=role, content=m.content))
        return memory

    def reset(self, session_id: str) -> bool:
        """清理指定 session 的并发锁（DB 数据由 sessions 路由删除）"""
        return self._locks.pop(session_id, None) is not None

    async def ask_question(self, question: str, memory: ChatMemoryBuffer) -> str:
        """单轮提问（CLI / 简单调用用）"""
        response = await self.agent.run(user_msg=question, memory=memory)
        return str(response)

    async def chat(self) -> None:
        """交互式 CLI 对话（单会话内存记忆）"""
        memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        print("=" * 50)
        print("book 知识库助手已启动")
        print(f"可用工具：{[t.metadata.name for t in self.tools]}")
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
            answer = await self.ask_question(user_input, memory)
            print(f"\n助手：{answer}")
