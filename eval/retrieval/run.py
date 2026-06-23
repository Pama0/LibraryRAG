"""阶段二：读冻结标注，逐 retriever 绕过 front_door 检索、算指标、出对比表。
零 LLM（dense 路调 embedding），可反复跑——改 retrieve.py 后只重跑此脚本。

运行（项目根）：python -m eval.retrieval.run --retrievers vector hybrid
"""
import argparse
import asyncio
import csv
import json
import os
from datetime import datetime, timezone

from configs.embedding import configure_embedding
from core.rag.data_loader import RAGIndexManager
from core.retrieval.retrieve import make_retriever
from eval.config import CHROMA_COLLECTION, CHROMA_DIR
from eval.retrieval.label import LABEL_OUT
from eval.retrieval.metrics import (
    K_VALUES,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RESULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results",
    f"retrieval_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
)


def per_query_metrics(
    retrieved: list[str], relevant: set[str], k_values: tuple[int, ...] = K_VALUES
) -> dict[str, float | None]:
    """一条 query 的全部指标（recall/precision/ndcg@k + mrr）。"""
    row: dict[str, float | None] = {"mrr": mrr(retrieved, relevant)}
    for k in k_values:
        row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        row[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return row


def aggregate(rows: list[dict]) -> dict[str, float | None]:
    """逐键求均值，忽略 None。空入参 → {}。"""
    if not rows:
        return {}
    out: dict[str, float | None] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def _metric_cols(k_values: tuple[int, ...] = K_VALUES) -> list[str]:
    cols = ["mrr"]
    for k in k_values:
        cols += [f"recall@{k}", f"precision@{k}", f"ndcg@{k}"]
    return cols


def _load_labels() -> list[dict]:
    if not os.path.exists(LABEL_OUT):
        raise SystemExit(f"标注文件不存在：{LABEL_OUT}\n先跑：python -m eval.retrieval.label")
    with open(LABEL_OUT, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    labeled = [r for r in rows if not r.get("skipped")]
    if not labeled:
        raise SystemExit("标注全为 skipped，无可评测样本；先核对 label 产物。")
    return labeled


async def _eval_retriever(name: str, labels: list[dict], idx) -> tuple[dict, list[dict]]:
    """跑一个 retriever：返回 (聚合均值, 每条明细)。"""
    retriever = make_retriever(name)
    top_k = max(K_VALUES)
    per_rows: list[dict] = []
    detail: list[dict] = []
    for r in labels:
        q = r["user_input"]
        relevant = set(r["relevant_chunk_ids"])
        try:
            nodes = await retriever.retrieve(
                q, index_manager=idx, book_titles=None, top_k=top_k
            )
            retrieved = [n.node.node_id for n in nodes]
        except Exception as exc:  # noqa: BLE001 — 单条异常不计入，不中断
            print(f"[warn] {name} 检索失败：{q[:30]} | {type(exc).__name__}: {exc}")
            continue
        m = per_query_metrics(retrieved, relevant)
        per_rows.append(m)
        detail.append({"variant": name, "user_input": q,
                       "category": r.get("category", ""), **m})
    return aggregate(per_rows), detail


def _render_table(results: list[tuple[str, dict]], sample_count: int) -> str:
    """results: [(name, 聚合均值)]。→ Markdown 对比表，末尾带跨 retriever 总和行。"""
    cols = _metric_cols()
    header = f"| retriever ({sample_count} 样本) | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for name, agg in results:
        cells = [f"{agg.get(c):.3f}" if agg.get(c) is not None else "—" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # 跨 retriever 总和行（对各列所有 variant 的均值再求均值）
    if len(results) > 1:
        total = aggregate([agg for _, agg in results])
        cells = [f"{total.get(c):.3f}" if total.get(c) is not None else "—" for c in cols]
        lines.append(f"| **总计** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_csv(all_detail: list[dict], aggregates: list[tuple[str, dict]], path: str) -> None:
    cols = ["variant", "user_input", "category"] + _metric_cols()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in all_detail:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in d.items()})
        # 每个 variant 的聚合均值追加在明细末尾
        for name, agg in aggregates:
            row = {"variant": name, "user_input": "", "category": ""}
            row.update({k: f"{v:.4f}" for k, v in agg.items() if v is not None})
            w.writerow(row)


async def main(retrievers: list[str]) -> None:
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
    labels = _load_labels()
    print(f"评测 {len(labels)} 条有标注样本 | retrievers={retrievers}\n")

    results: list[tuple[str, dict]] = []
    all_detail: list[dict] = []
    for name in retrievers:
        agg, detail = await _eval_retriever(name, labels, idx)
        results.append((name, agg))
        all_detail += detail

    print(_render_table(results, len(labels)))
    _write_csv(all_detail, results, RESULT_CSV)
    print(f"\n明细已写 {RESULT_CSV}")


def _parse_args():
    p = argparse.ArgumentParser(description="检索层评测：vector vs hybrid 的 Recall@k/nDCG")
    p.add_argument("--retrievers", nargs="+", default=["vector", "hybrid"],
                   help="被测 retriever 名（make_retriever 注册名），默认 vector hybrid")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.retrievers))
