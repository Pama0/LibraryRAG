import asyncio
import os
from dotenv import load_dotenv
import openai
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextEntityRecall

from configs.embedding import configure_embedding
from configs.llm import configure_llm
from core.rag.data_loader import RAGIndexManager

from core.workflow.simple_rag import SimpleRagWorkflow
from evals.evals import evaluate_rag


async def main():
    load_dotenv()
    api_key = os.getenv('ZHIPU_API_KEY')

    # 1. 配置 RAG 组件
    configure_embedding()
    llm = configure_llm()

    # 2. 加载数据并创建查询引擎
    manager = RAGIndexManager()
    index = manager.get_index()
    if index is None:
        raise ValueError(
            "索引为空，请先运行 python init_index.py 初始化数据"
        )
        # 3. 创建 RAG 工作流实例
    rag_workflow = SimpleRagWorkflow(index,llm)
    # rag_workflow = QueryEngineWorkflow(query_engine)

    # 4. 创建评测用 LLM（与 RAG 分开）
    # 注意：使用 AsyncOpenAI 客户端，因为 ragas 的 ascore 需要异步客户端
    eval_client = openai.AsyncOpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=api_key
    )
    eval_llm = llm_factory("glm-4-flash", client=eval_client)
    recall_metric = ContextEntityRecall(llm=eval_llm)

    # 5. 加载测试数据（使用 JSONL，保留嵌套结构）
    # 注意：手动读取并指定 utf-8 编码，避免 Windows 默认 GBK 编码问题
    import json
    from ragas.dataset_schema import EvaluationDataset
    with open("./evals/dataset/testset.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    dataset = EvaluationDataset.from_list(data)

    # 6. 运行评测
    from datetime import datetime
    from ragas.backends.local_csv import LocalCSVBackend

    exp_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_naiverag"

    results = await evaluate_rag.arun(
        dataset,
        name=exp_name,
        rag=rag_workflow,
        llm=eval_llm,
        recall_metric=recall_metric,
        backend=LocalCSVBackend(root_dir="./evals/results"),  # 直接传入实例
    )

    # 7. 输出结果
    if results:
        pass_count = sum(1 for r in results if r.get("correctness_score") == "pass")
        total = len(results)
        print(f"通过率: {pass_count}/{total} ({pass_count / total * 100:.1f}%)")

    return results


if __name__ == "__main__":
    asyncio.run(main())
