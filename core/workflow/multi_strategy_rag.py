"""多策略 RAG 工作流 - 支持并行策略执行和最佳结果评判"""
import logging
import os
from typing import Optional

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine


# ==================== 事件定义 ====================

class JudgeEvent(Event):
    """评判查询质量事件"""
    query: str


class BadQueryEvent(Event):
    """糟糕查询事件 - 需要改进"""
    query: str


class NaiveRAGEvent(Event):
    """朴素 RAG 策略事件"""
    query: str


class HighTopKEvent(Event):
    """高 TopK 策略事件"""
    query: str


class RerankEvent(Event):
    """重排序策略事件"""
    query: str


class ResponseEvent(Event):
    """响应事件 - 包含策略来源"""
    query: str
    response: str
    source: str = "Unknown"


# ==================== 工作流定义 ====================

class MultiStrategyRAGWorkflow(Workflow):
    """
    多策略 RAG 工作流

    流程：
    1. 判断查询质量 -> 坏查询则改进后重试
    2. 并行执行 3 种 RAG 策略：Naive, HighTopK, Rerank
    3. 收集所有响应，LLM 评判选择最佳结果

    适用场景：需要高质量检索答案的场景
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: LLM,
        data_dir: str = "data",
        persist_dir: str = "storage",
        **kwargs
    ):
        """
        初始化工作流

        Args:
            index: 向量索引
            llm: 语言模型
            data_dir: 数据目录
            persist_dir: 索引持久化目录
        """
        # 设置较长超时，因为有多个策略并行执行
        kwargs.setdefault("timeout", 180)
        kwargs.setdefault("verbose", False)
        super().__init__(**kwargs)
        self.index = index
        self.llm = llm
        self.data_dir = data_dir
        self.persist_dir = persist_dir

    @step
    async def judge_query(
        self, ctx: Context, ev: StartEvent | JudgeEvent
    ) -> BadQueryEvent | NaiveRAGEvent | HighTopKEvent | RerankEvent:
        logging.info("进入复杂查询流程")
        """
        判断查询质量，决定是改进查询还是直接执行检索策略
        """
        # 初始化上下文
        judge_engine = await ctx.store.get("judge", default=None)
        if judge_engine is None:
            await ctx.store.set("llm", self.llm)
            await ctx.store.set("index", self.index)
            await ctx.store.set("judge", SimpleChatEngine.from_defaults(llm=self.llm))

        judge_engine = await ctx.store.get("judge")

        # 判断查询质量
        response = await judge_engine.achat(
            f"""判断以下查询是否能从 RAG 系统获得好的结果。
好的查询包含具体关键词且详细；坏的查询模糊或有歧义。

只需回答 'good' 或 'bad'，不要其他内容。

查询：{ev.query}"""
        )

        if "bad" in str(response).lower():
            return BadQueryEvent(query=ev.query)
        else:
            # 并行发送到 3 个策略
            self.send_event(NaiveRAGEvent(query=ev.query))
            self.send_event(HighTopKEvent(query=ev.query))
            self.send_event(RerankEvent(query=ev.query))

    @step
    async def improve_query(
        self, ctx: Context, ev: BadQueryEvent
    ) -> JudgeEvent:
        """改进糟糕的查询"""
        llm = await ctx.store.get("llm")
        response = await llm.acomplete(
            f"""这是一个 RAG 系统的查询，但太模糊了。

请提供一个更详细的查询版本，包含：
- 具体的关键词
- 明确的搜索目标
- 消除歧义

原查询：{ev.query}

改进后的查询："""
        )
        print(f"[Workflow] 查询已改进: {ev.query} -> {str(response).strip()}")
        return JudgeEvent(query=str(response).strip())

    @step
    async def naive_rag(
        self, ctx: Context, ev: NaiveRAGEvent
    ) -> ResponseEvent:
        """朴素 RAG 策略：基础 top-k 检索"""
        index = await ctx.store.get("index")
        engine = index.as_query_engine(similarity_top_k=5)
        response = await engine.aquery(ev.query)
        print(f"[Naive RAG] 完成")
        return ResponseEvent(
            query=ev.query,
            source="Naive (top_k=5)",
            response=str(response)
        )

    @step
    async def high_top_k(
        self, ctx: Context, ev: HighTopKEvent
    ) -> ResponseEvent:
        """高 TopK 策略：检索更多候选"""
        index = await ctx.store.get("index")
        engine = index.as_query_engine(similarity_top_k=20)
        response = await engine.aquery(ev.query)
        print(f"[High TopK] 完成")
        return ResponseEvent(
            query=ev.query,
            source="HighTopK (top_k=20)",
            response=str(response)
        )

    @step
    async def rerank(
        self, ctx: Context, ev: RerankEvent
    ) -> ResponseEvent:
        """重排序策略：检索后用 LLM 重排序"""
        index = await ctx.store.get("index")
        llm = await ctx.store.get("llm")

        reranker = RankGPTRerank(top_n=5, llm=llm)
        retriever = index.as_retriever(similarity_top_k=20)
        engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
        )
        response = await engine.aquery(ev.query)
        print(f"[Rerank] 完成")
        return ResponseEvent(
            query=ev.query,
            source="Rerank (top_k=20 + rerank)",
            response=str(response)
        )

    @step
    async def judge(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        """
        收集所有策略的响应，选择最佳结果
        """
        # 等待收集所有 3 个响应
        ready = ctx.collect_events(ev, [ResponseEvent] * 3)
        if ready is None:
            return None

        judge_engine = await ctx.get("judge")

        # 让 LLM 评判最佳响应
        response = await judge_engine.achat(
            f"""用户查询：{ev.query}

以下是 3 种不同 RAG 策略的回答，请选择最佳的一个。

策略 1 ({ready[0].source})：
{ready[0].response}

策略 2 ({ready[1].source})：
{ready[1].response}

策略 3 ({ready[2].source})：
{ready[2].response}

请只回答最佳策略的编号（1、2 或 3），不要其他内容。"""
        )

        try:
            best_idx = int(str(response).strip()) - 1
            best_idx = max(0, min(best_idx, 2))  # 确保在有效范围内
        except ValueError:
            best_idx = 0

        best = ready[best_idx]
        print(f"[Judge] 最佳策略: {best.source}")

        return StopEvent(result=best.response)
