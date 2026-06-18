"""单系统 runner：逐行跑一个被测系统 → 打分 → 聚合 → 渲染表 + 落盘。

本文件上半部（load_testset / _row_to_dict / score_row / aggregate /
build_single_report）是纯逻辑，已 TDD；下半部 _run / main 是装配，靠集成 smoke 验证。
与 compare（多变体 ablation）的区别：本入口只跑**一个**系统（默认 flags 的
DocQueryWorkflow），共用 report.py 的渲染与落盘（单行表，自作 baseline，无 delta）。
"""
import json
import argparse
import asyncio
import os

from eval.harness.metrics import METRIC_NAMES, MetricSpec
from eval.harness.report import (
    default_result_paths,
    render_delta_table,
    write_detail_csv,
)
from eval.harness.sut import RagOutput, RagSystem

SINGLE_SYSTEM_LABEL = "当前系统(默认flags)"


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
        if exp and cat:  # 仅在系统确实产出类别时计分；agent(无 category) → N/A，error 行不算误判
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


def build_single_report(label: str, scored: list[dict]) -> tuple[dict, list[dict]]:
    """单系统：aggregate 跑分 + 给每条明细打 variant 标（与 compare detail 同构，供落盘）。"""
    report = aggregate(scored)
    detail = [{"variant": label, **s} for s in scored]
    return report, detail


async def _run(testset_path: str, limit: int | None) -> tuple[dict, list[dict]]:
    from eval.config import CHROMA_DIR, make_eval_embeddings, make_eval_llm
    from eval.harness.metrics import build_metric_specs
    from eval.harness.sut import DocQueryWorkflowSystem
    from configs.embedding import configure_embedding
    from configs.llm import configure_llm
    from core.rag.data_loader import RAGIndexManager

    rows = load_testset(testset_path)
    if limit:
        rows = rows[:limit]

    eval_llm, eval_emb = make_eval_llm(), make_eval_embeddings()
    metric_specs = build_metric_specs(eval_llm, eval_emb)

    # SUT 检索需要全局 Settings.embed_model 与 llm，二者都要先配置
    sut_llm = configure_llm()
    configure_embedding()
    # 单系统：默认 flags 的 DocQueryWorkflow（构造默认决策配置）
    sut = DocQueryWorkflowSystem(
        index_manager=RAGIndexManager(persist_dir=CHROMA_DIR), llm=sut_llm
    )

    scored = [await score_row(_row_to_dict(r), sut, metric_specs) for r in rows]
    return build_single_report(SINGLE_SYSTEM_LABEL, scored)


def main():
    parser = argparse.ArgumentParser(
        description="Book RAG 单系统评测（默认 flags 的 DocQueryWorkflow）"
    )
    parser.add_argument("--testset", default=None, help="测试集 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--out", default=None,
                        help="跑分表 Markdown 落盘路径；缺省 eval/results/run_eval_<时间戳>.md")
    parser.add_argument("--detail", default=None,
                        help="每条明细 CSV 落盘路径；缺省 eval/results/run_eval_<时间戳>_detail.csv")
    args = parser.parse_args()

    from eval.config import TESTSET_PATH
    path = args.testset or TESTSET_PATH
    report, detail = asyncio.run(_run(path, args.limit))

    # 单行表：该系统自作 baseline，无 delta；列与 compare 一致（分类准确率 + 5 ragas）
    table = render_delta_table(
        [{"name": SINGLE_SYSTEM_LABEL, "report": report}], baseline=SINGLE_SYSTEM_LABEL
    )
    print(table)

    default_md, default_csv = default_result_paths("run_eval")
    out_path = args.out or default_md
    detail_path = args.detail or default_csv

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 单系统跑分（{SINGLE_SYSTEM_LABEL}）\n\n")
        f.write(f"测试集：`{path}`" + (f"（前 {args.limit} 条）" if args.limit else "") + "\n\n")
        f.write(table + "\n")
    print(f"\n[已存] {out_path}")

    write_detail_csv(detail, detail_path)
    print(f"[已存明细] {detail_path}（共 {len(detail)} 行）")


if __name__ == "__main__":
    main()
