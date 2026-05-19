"""Agent 工具模块 - 依赖 agent/workflow 的工具定义"""
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool


from core.workflow.multi_strategy_rag import MultiStrategyRAGWorkflow

from core.workflow.simple_rag import SimpleRagWorkflow


def create_simple_rag_tool(
    index: VectorStoreIndex,
    llm: LLM,
    name: str = "simple_rag",
    description: str = "对本地知识库进行简单查询"
) -> FunctionTool:
    """
    创建简单 RAG 检索工具

    Args:
        index: 存储索引
        llm: 语言模型
        name: 工具名称
        description: 工具描述

    Returns:
        QueryEngineTool 实例
    """
    workflow = SimpleRagWorkflow(index=index,llm=llm)

    async def simple_rag_search(query: str = "") -> str:
        """对本地知识库进行简单快速查询

        Args:
            query: 要查询的问题，必须是字符串，如"保安服务管理条例第三条是什么"
        """
        # LLM 有时返回 dict 而非字符串，或未传参数
        if not isinstance(query, str):
            query = query.get("title") or query.get("text") or str(query)
        query = str(query).strip()
        if not query:
            return "请提供要查询的问题"
        result = await workflow.run(query=query)
        return str(result)

    return FunctionTool.from_defaults(
        fn=simple_rag_search,
        name=name,
        description=description
    )

def create_multi_strategy_rag_tool(
    index: VectorStoreIndex,
    llm: LLM,
) -> FunctionTool:
    """
    创建多策略 RAG 检索工具

    该工具会：
    1. 判断查询质量，必要时自动改进
    2. 并行执行 3 种检索策略（Naive, HighTopK, Rerank）
    3. 由 LLM 评判选择最佳答案

    Args:
        index: 向量索引
        llm: 语言模型

    Returns:
        FunctionTool 实例
    """
    workflow = MultiStrategyRAGWorkflow(index=index, llm=llm)

    async def multi_strategy_search(query: str) -> str:
        """
        使用多种 RAG 策略搜索文档并返回最佳答案。

        适用场景：需要高质量检索答案、普通检索效果不佳时。

        Args:
            query: 搜索查询

        Returns:
            最佳检索答案
        """
        result = await workflow.run(query=query)
        return str(result)

    return FunctionTool.from_defaults(
        fn=multi_strategy_search,
        name="multi_strategy_search",
        description=(
            "高级文档检索工具：使用多种策略并行搜索文档，"
            "自动评判选择最佳答案。适合需要高质量检索结果的场景。"
        )
    )
