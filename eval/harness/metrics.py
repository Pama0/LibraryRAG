"""5 个 ragas collections 指标：字段映射（纯函数）+ MetricSpec 装配。

字段映射与指标构造分离——映射可离线单测，构造需真实 InstructorLLM（集成 smoke）。
"""
from dataclasses import dataclass
from typing import Callable


# ── 字段映射：row(dict) + RagOutput → ascore kwargs ──
def _faithfulness_kwargs(row, out):
    return {"user_input": row["user_input"], "response": out.response,
            "retrieved_contexts": out.retrieved_contexts}


def _answer_relevancy_kwargs(row, out):
    return {"user_input": row["user_input"], "response": out.response}


def _context_precision_kwargs(row, out):
    return {"user_input": row["user_input"], "reference": row["reference"],
            "retrieved_contexts": out.retrieved_contexts}


def _context_recall_kwargs(row, out):
    return {"user_input": row["user_input"], "retrieved_contexts": out.retrieved_contexts,
            "reference": row["reference"]}


def _factual_correctness_kwargs(row, out):
    return {"response": out.response, "reference": row["reference"]}


METRIC_KWARGS: dict[str, Callable] = {
    "faithfulness": _faithfulness_kwargs,
    "answer_relevancy": _answer_relevancy_kwargs,
    "context_precision": _context_precision_kwargs,
    "context_recall": _context_recall_kwargs,
    "factual_correctness": _factual_correctness_kwargs,
}

# 指标均值聚合的固定顺序
METRIC_NAMES = list(METRIC_KWARGS.keys())


@dataclass
class MetricSpec:
    name: str
    metric: object
    kwargs: Callable  # (row: dict, out: RagOutput) -> dict


def build_metric_specs(llm, embeddings) -> list[MetricSpec]:
    """构造 5 个 collections 指标（llm/embeddings 须为真实 ragas 对象）。"""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
        FactualCorrectness,
    )

    return [
        MetricSpec("faithfulness", Faithfulness(llm=llm), METRIC_KWARGS["faithfulness"]),
        MetricSpec("answer_relevancy", AnswerRelevancy(llm=llm, embeddings=embeddings),
                   METRIC_KWARGS["answer_relevancy"]),
        MetricSpec("context_precision", ContextPrecisionWithReference(llm=llm),
                   METRIC_KWARGS["context_precision"]),
        MetricSpec("context_recall", ContextRecall(llm=llm), METRIC_KWARGS["context_recall"]),
        MetricSpec("factual_correctness", FactualCorrectness(llm=llm, mode="f1"),
                   METRIC_KWARGS["factual_correctness"]),
    ]
