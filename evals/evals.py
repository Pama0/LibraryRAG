from ragas.metrics import DiscreteMetric
from llama_index.core.workflow import Workflow
from ragas.metrics.collections import ContextEntityRecall

# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.

Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
    allowed_values=["pass", "fail"],
)

import asyncio
from typing import Dict, Any
from ragas import experiment

@experiment()
async def evaluate_rag(row, rag: Workflow, llm, recall_metric: ContextEntityRecall) -> Dict[str, Any]:
    """
    Run RAG evaluation on a single row.

    Args:
        row: SingleTurnSample object containing user_input and reference
        rag: Pre-initialized RAG instance
        llm: Pre-initialized LLM client for evaluation
        recall_metric: Pre-initialized recall metric instance

    Returns:
        Dictionary with evaluation results
    """
    # 从 SingleTurnSample 对象中提取数据（使用属性访问）
    question = row.user_input
    expected_answer = row.reference

    # Query the RAG system
    rag_response = await rag.run(query=question)
    model_response = str(rag_response)

    # Evaluate correctness asynchronously
    score = await correctness_metric.ascore(
        question=question,
        expected_answer=expected_answer,
        response=model_response,
        llm=llm
    )
    recall = await recall_metric.ascore(
        reference=row.reference_contexts,
        retrieved_contexts=[node.node.text for node in rag_response.source_nodes],
    )
    # Return evaluation results
    result = {
        "user_input": question,
        "reference": expected_answer,
        "model_response": model_response,
        "correctness_score": score.value,
        "correctness_reason": score.reason,
        "recall_score": recall,
    }

    return result
