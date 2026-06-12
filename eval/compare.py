"""决策对比 runner：对变体列表跑同一测试集，渲染 baseline vs 变体的 delta 表。

变体 = 一组决策 flag（probe/split/assume/other 的 on-off）。baseline 通常全单轮，
逐个打开决策，对比表每行一个变体、delta 列即"该决策带来多少提升"。
"""
import argparse
import asyncio

# 对比表展示的列（确定性指标——分类准确率——优先，最适合归因决策）
_COLS = [
    ("分类准确率", lambda rep: rep.get("classification", {}).get("accuracy")),
    ("context_recall", lambda rep: rep.get("metric_means", {}).get("context_recall")),
    ("factual_correctness", lambda rep: rep.get("metric_means", {}).get("factual_correctness")),
    ("faithfulness", lambda rep: rep.get("metric_means", {}).get("faithfulness")),
    ("answer_relevancy", lambda rep: rep.get("metric_means", {}).get("answer_relevancy")),
]


def _fmt(val, base):
    """单元格：值 + 相对 baseline 的 delta（baseline 自身或无值不带 delta）。"""
    if val is None:
        return "—"
    if base is None or val == base:
        return f"{val:.2f}"
    return f"{val:.2f} ({val - base:+.2f})"


def render_delta_table(variants: list[dict], baseline: str) -> str:
    """variants: [{"name", "report"(aggregate 输出)}]。→ Markdown delta 表。"""
    base_rep = next(v["report"] for v in variants if v["name"] == baseline)
    header = "| 配置 | " + " | ".join(c[0] for c in _COLS) + " |"
    sep = "|" + "---|" * (len(_COLS) + 1)
    lines = [header, sep]
    for v in variants:
        cells = [_fmt(getter(v["report"]), getter(base_rep)) for _, getter in _COLS]
        lines.append(f"| {v['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 变体矩阵：baseline 全单轮，逐个打开决策
VARIANTS = {
    "baseline(全单轮)": dict(probe_then_classify=False, split_enabled=False,
                             assume_enabled=False, other_agent_enabled=False),
    "+probe": dict(probe_then_classify=True, split_enabled=False,
                   assume_enabled=False, other_agent_enabled=False),
    "+probe+split": dict(probe_then_classify=True, split_enabled=True,
                         assume_enabled=False, other_agent_enabled=False),
    "全开": dict(probe_then_classify=True, split_enabled=True,
                 assume_enabled=True, other_agent_enabled=True),
}


async def _run_variants(testset_path, limit, names):
    from eval.config import CHROMA_DIR, make_eval_embeddings, make_eval_llm
    from eval.metrics import build_metric_specs
    from eval.run_eval import load_testset, score_row, aggregate
    from eval.sut import DocQueryWorkflowSystem
    from configs.embedding import configure_embedding
    from configs.llm import configure_llm
    from core.rag.data_loader import RAGIndexManager

    rows = load_testset(testset_path)
    if limit:
        rows = rows[:limit]
    eval_llm, eval_emb = make_eval_llm(), make_eval_embeddings()
    metric_specs = build_metric_specs(eval_llm, eval_emb)
    sut_llm = configure_llm()
    configure_embedding()
    index_manager = RAGIndexManager(persist_dir=CHROMA_DIR)

    variants = []
    for name in names:
        sut = DocQueryWorkflowSystem(index_manager, sut_llm, flags=VARIANTS[name])
        scored = [await score_row(r, sut, metric_specs) for r in rows]
        variants.append({"name": name, "report": aggregate(scored)})
    return variants


def main():
    p = argparse.ArgumentParser(description="决策对比评测（ablation）")
    p.add_argument("--testset", required=True, help="测试集 jsonl（建议金标准 golden.jsonl）")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                   help=f"变体名子集，可选：{list(VARIANTS.keys())}")
    p.add_argument("--baseline", default="baseline(全单轮)", help="作为 delta 基准的变体名")
    args = p.parse_args()
    variants = asyncio.run(_run_variants(args.testset, args.limit, args.variants))
    print(render_delta_table(variants, baseline=args.baseline))


if __name__ == "__main__":
    main()
