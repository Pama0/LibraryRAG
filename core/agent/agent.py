import asyncio
from typing import List

from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool


class MyAgent:
    """
    智能 Agent - 纯粹的 Agent 定义

    职责：接收工具列表，执行对话
    工具创建：由外部（app.py）负责
    """

    def __init__(
        self,
        tools: List[FunctionTool],
        llm: LLM,
        system_prompt: str,
    ):
        """
        初始化 Agent

        Args:
            tools: 工具列表（由外部创建并传入）
            llm: 语言模型
            system_prompt: 系统提示词
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools

        # 创建 ReActAgent
        self.agent = ReActAgent(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            verbose=True
        )

    async def ask_question(self, question: str, memory: ChatMemoryBuffer) -> str:
        """提问并获取回答"""
        response = await self.agent.run(user_msg=question, memory=memory)
        return str(response)

    async def chat(self):
        """交互式对话"""
        memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

        print("=" * 50)
        print("智能 Agent 已启动")
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

            response = await self.ask_question(user_input, memory)
            print(f"\n助手：{response}")