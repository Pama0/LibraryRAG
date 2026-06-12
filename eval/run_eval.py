"""@experiment runner：逐行跑 SUT → 打分 → 聚合。

本文件上半部（load_testset / _row_to_dict / score_row / aggregate）是纯逻辑，
已 TDD；下半部 main 是 @experiment + Dataset 装配，靠集成 smoke 验证。
"""
import json
import argparse
import asyncio
import os

from eval.metrics import METRIC_NAMES, MetricSpec
from eval.sut import RagOutput, RagSystem


def load_testset(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_dict(row) -> dict:
    """把 ragas Dataset 行（可能是 dict / pydantic / 带属性对象）归一成 dict。"""
    if isinstance(row, dict):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    # 兜底：按已知字段取属性
    keys = ["user_input", "reference", "reference_contexts"]
    return {k: getattr(row, k, None) for k in keys}


async def score_row(row: dict, sut: RagSystem, metric_specs: list[MetricSpec]) -> dict:
    out: RagOutput = await sut.answer(row["user_input"])
    base = {
        "user_input": row["user_input"],
        "reference": row.get("reference", ""),
        "response": out.response,
        "outcome": out.outcome,
        "category": out.category,                       # SUT 实际判的 category
        "expected_category": row.get("category", ""),   # 测试集金标准标注
        "num_contexts": len(out.retrieved_contexts),
    }
    if out.outcome != "answered":
        return base
    for spec in metric_specs:
        try:
            result = await spec.metric.ascore(**spec.kwargs(row, out))
            base[spec.name] = result.value
        except Exception as e:  # noqa: BLE001 — 单指标失败不影响其他指标
            base[spec.name] = None
            base[f"{spec.name}_error"] = f"{type(e).__name__}: {e}"
    return base


def aggregate(rows: list[dict]) -> dict:
    """指标均值（仅 answered 行、忽略 None）+ outcome/category 分布 + 分类准确率。"""
    outcomes: dict[str, int] = {}
    cat_dist: dict[str, int] = {}
    cls_total = cls_correct = 0
    for r in rows:
        oc = r.get("outcome", "error")
        outcomes[oc] = outcomes.get(oc, 0) + 1
        cat = r.get("category") or ""
        if cat:
            cat_dist[cat] = cat_dist.get(cat, 0) + 1
        exp = r.get("expected_category")
        if exp:
            cls_total += 1
            cls_correct += int(cat == exp)
    answered = [r for r in rows if r.get("outcome") == "answered"]
    metric_means: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        vals = [r[name] for r in answered if r.get(name) is not None]
        metric_means[name] = (sum(vals) / len(vals)) if vals else None
    return {
        "total": len(rows),
        "answered": len(answered),
        "outcome_distribution": outcomes,
        "category_distribution": cat_dist,
        "classification": {
            "total": cls_total,
            "correct": cls_correct,
            "accuracy": (cls_correct / cls_total) if cls_total else None,
        },
        "metric_means": metric_means,
    }


async def _run(testset_path: str, limit: int | None) -> dict:
    from ragas import Dataset, experiment
    from ragas.backends import LocalCSVBackend

    from eval.config import CHROMA_DIR, RESULTS_DIR, make_eval_embeddings, make_eval_llm
    from eval.metrics import build_metric_specs
    from eval.sut import BookRagWorkflowSystem
    from configs.embedding import configure_embedding
    from configs.llm import configure_llm
    from core.rag.data_loader import RAGIndexManager

    rows = load_testset(testset_path)
    if limit:
        rows = rows[:limit]

    eval_llm = make_eval_llm()
    eval_emb = make_eval_embeddings()
    metric_specs = build_metric_specs(eval_llm, eval_emb)

    # SUT 检索需要全局 Settings.embed_model 与 llm，二者都要先配置
    sut_llm = configure_llm()
    configure_embedding()
    sut = BookRagWorkflowSystem(
        index_manager=RAGIndexManager(persist_dir=CHROMA_DIR), llm=sut_llm
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    backend = LocalCSVBackend(root_dir=RESULTS_DIR)
    dataset = Dataset(name="book_testset", backend=backend, data=rows)

    @experiment()
    async def book_rag_experiment(row, sut, metric_specs):
        return await score_row(_row_to_dict(row), sut, metric_specs)

    exp = await book_rag_experiment.arun(
        dataset, name="book_rag", sut=sut, metric_specs=metric_specs,
    )
    result_rows = [_row_to_dict(r) for r in exp.to_pandas().to_dict("records")]
    return aggregate(result_rows)


def main():
    parser = argparse.ArgumentParser(description="Book RAG ragas 评测")
    parser.add_argument("--testset", default=None, help="测试集 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    args = parser.parse_args()

    from eval.config import TESTSET_PATH
    path = args.testset or TESTSET_PATH
    report = asyncio.run(_run(path, args.limit))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
