from core.agent.agent import MyAgent
from core.tools.tools import create_multi_strategy_rag_tool,create_simple_rag_tool
from configs.embedding import configure_embedding
from configs.llm import configure_llm
from core.rag.data_loader import RAGIndexManager

async def run():
    """应用入口 - 组装各组件"""
    # 1. 初始化配置
    configure_embedding()
    llm = configure_llm()

    # 2. 加载索引，为空则自动初始化
    manager = RAGIndexManager()
    if manager.index is None:
        print("索引为空，开始初始化...")
        manager.add_documents("./data")
    index = manager.get_index()

    # 3. 创建工具（在组装层完成）
    # 简单 RAG 工具
    simple_rag_tool = create_simple_rag_tool(
        index=index,
        llm=llm,
        name="simple_rag",
        description="对本地知识库进行简单快速查询"
    )

    # 多策略 RAG 工具（高质量检索）
    multi_strategy_tool = create_multi_strategy_rag_tool(
        index=index,
        llm=llm,
    )

    tools = [simple_rag_tool]

    # 4. 创建 Agent
    agent = MyAgent(
        tools=tools,
        llm=llm,
        system_prompt="""你是一个可以使用工具的超级智能助手。

可用工具：
1. simple_rag - 快速检索，适合简单问题
2. multi_strategy_search - 高级检索，使用多种策略并行搜索并选择最佳答案

重要规则：
1. 回答任何问题之前，你必须先调用检索工具查询本地知识库
2. 对于简单问题使用 simple_rag，对于复杂或需要高质量答案的问题使用 multi_strategy_search
3. 不要依赖对话历史中的信息，因为历史可能不完整
4. 如果检索工具没有返回相关信息，请明确告知用户知识库中没有相关内容"""
    )

    # 5. 启动对话
    await agent.chat()



if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
