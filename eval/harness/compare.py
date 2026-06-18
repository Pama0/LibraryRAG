"""决策对比 runner：对变体列表跑同一测试集，渲染 baseline vs 变体的 delta 表。

变体 = 一组决策 flag（probe/split/assume/other 的 on-off）。baseline 通常全单轮，
逐个打开决策，对比表每行一个变体、delta 列即"该决策带来多少提升"。
"""
# agent vs 全开（delta 相对全开）python -m eval.harness.compare --testset eval/dataset/golden.jsonl --variants "全开" "agent(自主规划)"
import argparse
import asyncio
import os

from eval.harness.report import (
    default_result_paths,
    render_delta_table,
    write_detail_csv,
)


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
    "全开+rerank": dict(probe_then_classify=True, split_enabled=True,
                        assume_enabled=True, other_agent_enabled=True,
                        reranker="bge-reranker-v2-m3"),
    "全开+hybrid": dict(probe_then_classify=True, split_enabled=True,
                        assume_enabled=True, other_agent_enabled=True,
                        retriever="hybrid"),
    "全开+hybrid+rerank": dict(probe_then_classify=True, split_enabled=True,
                               assume_enabled=True, other_agent_enabled=True,
                               retriever="hybrid", reranker="bge-reranker-v2-m3"),
}

# agent 自主规划路线：另一个 SUT 类（非 flags 组合）。值置 None 作哨兵，
# 进 CLI 可选名集合（choices）但不进默认 --variants 全集（多轮真实 agent，烧 LLM，需显式选）；
# build_sut 里按 None 分流到 AgentSystem。
AGENT_VARIANT = "agent(自主规划)"
VARIANTS[AGENT_VARIANT] = None


def build_sut(name: str, index_manager, llm):
    """按变体名构造被测系统：哨兵(None) → AgentSystem，否则 DocQueryWorkflowSystem(flags)。"""
    from eval.harness.sut import AgentSystem, DocQueryWorkflowSystem
    if name not in VARIANTS:
        raise KeyError(name)
    flags = VARIANTS[name]
    if flags is None:
        return AgentSystem(index_manager, llm)
    return DocQueryWorkflowSystem(index_manager, llm, flags=flags)


def resolve_baseline(baseline: str, variant_names: list[str]) -> str:
    """baseline 名不在所选变体里时回退到第一个变体（leftmost 作 delta 锚）。"""
    return baseline if baseline in variant_names else (variant_names[0] if variant_names else baseline)


async def _run_variants(testset_path, limit, names):
    from eval.config import CHROMA_DIR, make_eval_embeddings, make_eval_llm
    from eval.harness.metrics import build_metric_specs
    from eval.harness.meter import attach_token_meter
    from eval.harness.run_eval import load_testset, score_row, aggregate
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
    meter = attach_token_meter(sut_llm)  # 挂在 SUT llm 上 → 只数被测系统 token（逐行 reset）
    index_manager = RAGIndexManager(persist_dir=CHROMA_DIR)

    variants = []
    detail = []  # 每条明细（带 variant 列），供 --detail 落盘
    for name in names:
        sut = build_sut(name, index_manager, sut_llm)
        scored = [await score_row(r, sut, metric_specs, meter=meter) for r in rows]
        for s in scored:
            detail.append({"variant": name, **s})
        variants.append({"name": name, "report": aggregate(scored)})
    return variants, detail


def main():
    p = argparse.ArgumentParser(description="决策对比评测（ablation）")
    p.add_argument("--testset", required=True, help="测试集 jsonl（建议金标准 golden.jsonl）")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    p.add_argument("--variants", nargs="+",
                   default=[n for n in VARIANTS if n != AGENT_VARIANT],
                   choices=list(VARIANTS.keys()),
                   help=f"变体名子集，可选：{list(VARIANTS.keys())}；agent(自主规划) 需显式指定")
    p.add_argument("--baseline", default="baseline(全单轮)", help="作为 delta 基准的变体名")
    p.add_argument("--out", default=None,
                   help="对比表 Markdown 落盘路径；缺省 eval/results/compare_<时间戳>.md")
    p.add_argument("--detail", default=None,
                   help="每条明细 CSV 落盘路径；缺省 eval/results/compare_<时间戳>_detail.csv")
    args = p.parse_args()
    args.baseline = resolve_baseline(args.baseline, args.variants)
    variants, detail = asyncio.run(_run_variants(args.testset, args.limit, args.variants))
    table = render_delta_table(variants, baseline=args.baseline)
    print(table)

    # 缺省即落盘到 eval/results（带时间戳防覆盖）；--out/--detail 可显式改路径
    default_md, default_csv = default_result_paths()
    out_path = args.out or default_md
    detail_path = args.detail or default_csv

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 决策对比（baseline={args.baseline}）\n\n")
        f.write(f"测试集：`{args.testset}`" + (f"（前 {args.limit} 条）" if args.limit else "") + "\n\n")
        f.write(table + "\n")
    print(f"\n[已存] {out_path}")

    write_detail_csv(detail, detail_path)
    print(f"[已存明细] {detail_path}（共 {len(detail)} 行 = 条数 × 变体数）")


if __name__ == "__main__":
    main()
