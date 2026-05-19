from ragas import evaluate
from ragas.metrics import (
    Faithfulness,  # 忠实度 - 回答是否基于检索内容
    AnswerRelevancy,  # 答案相关性 - 回答是否切题
    ContextPrecision,  # 上下文精确度 - 检索内容是否精准
    ContextRecall,  # 上下文召回率 - 是否检索到所有相关信息
)
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.llms.openai import OpenAI


class RagasEvaluator:
    """Ragas RAG 评估器"""

    def __init__(self, llm=None):
        # 使用 LlamaIndex 的 LLM 包装器
        self.evaluator_llm = LlamaIndexLLMWrapper(
            llm or OpenAI(model="gpt-4o-mini")
        )

        # 初始化评估指标
        self.metrics = [
            Faithfulness(llm=self.evaluator_llm),
            AnswerRelevancy(llm=self.evaluator_llm),
            ContextPrecision(llm=self.evaluator_llm),
            ContextRecall(llm=self.evaluator_llm),
        ]

    def evaluate_response(
            self,
            query: str,
            response: str,
            contexts: list[str],
            reference: str = None
    ):
        """评估单条 RAG 响应"""
        from ragas import EvaluationDataset, SingleTurnSample

        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
            reference=reference
        )

        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=self.metrics
        )
        return result.to_pandas()

    def compare_strategies(self, query: str, strategies_results: dict):
        """比较多个策略的评估结果

        Args:
            query: 查询
            strategies_results: {
                "Naive": {"response": "...", "contexts": [...]},
                "HighTopK": {"response": "...", "contexts": [...]},
                "Rerank": {"response": "...", "contexts": [...]},
            }
        """
        all_results = []
        for strategy_name, data in strategies_results.items():
            result = self.evaluate_response(
                query=query,
                response=data["response"],
                contexts=data["contexts"]
            )
            result["strategy"] = strategy_name
            all_results.append(result)

        import pandas as pd
        return pd.concat(all_results, ignore_index=True)
