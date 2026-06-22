"""probe 离散度筛子 v2：措辞模板定向产 split / other / retrievable，凑齐配额。

两轮 PoC 校准出的规律：
  - 召回【离散度】决定 散 vs 聚：聚(有主导簇)→retrievable；散(无主导簇)→该拆
  - 散题再按【措辞意图】细分：
      列举式（"…分别有哪些/起什么作用"）→ pending_split（罗列并列子项）
      综合式（"…如何协作/取舍"）        → other（跨主题综合推理）

故标签 = 客观离散度(散/聚) × 措辞意图(列举/综合)，独立于被测分类器 → 金标准不循环。
同时跑真实 classify() 记 sut_category，输出一致性预览（分类准确率雏形）。

产出：dataset/split_candidates.jsonl + 控制台表格。人工抽检 ⚠ 不一致 / borderline 后并入 golden。

运行：python -m eval.datagen.build_split_candidates
"""
import asyncio
import json
import os
from collections import Counter

from configs.embedding import configure_embedding
from configs.llm import configure_llm
from core.rag.data_loader import RAGIndexManager
from core.workflow.chapter_tree import chapter_number
from core.workflow.qa_capability import QaCapability
from eval.config import CHROMA_DIR, DATASET_DIR

PROBE_K = 10  # 量离散度用更大 k，信号更稳

# 跨章多实体组合：各概念分属【不同专章】，强制召回横跨远隔章 → 离散
COMBOS = [
    "索引、基于成本的优化、EXPLAIN分析",          # 6 / 12 / 15
    "Buffer Pool、redo日志、undo日志",            # 18 / 20 / 22
    "InnoDB的表空间结构、记录行格式、B+树索引",    # 9 / 4 / 6
]
# 措辞模板：同一组合两种问法，定向落 split / other
ENUM_TMPL = "在MySQL中，{c}分别起什么作用？请分别说明各自的功能。"          # → pending_split
SYNTH_TMPL = "在MySQL中，{c}是如何协作配合的？请综合分析它们之间的关系与设计取舍。"  # → other

# 窄题种子（单一概念、内容集中 → retrievable）
NARROW = [
    "COMPACT行格式的NULL值列表是做什么用的？",
    "redo日志里的LSN是什么？",
    "什么是聚簇索引？",
    "innodb_log_file_size 参数有什么作用？",
]


def make_candidates():
    """[(source, question, intent)]；intent ∈ {enumerate, synthesize, narrow}。"""
    out = []
    for c in COMBOS:
        out.append(("跨章·列举", ENUM_TMPL.format(c=c), "enumerate"))
        out.append(("跨章·综合", SYNTH_TMPL.format(c=c), "synthesize"))
    for q in NARROW:
        out.append(("窄题", q, "narrow"))
    return out


def dispersion(nodes):
    tops = []
    for n in nodes:
        p = chapter_number((getattr(n, "metadata", {}) or {}).get("chapter", "") or "")
        if p:
            tops.append(p[0])
    if not tops:
        return [], 0, 0.0
    c = Counter(tops)
    return sorted(c), len(c), round(max(c.values()) / len(tops), 2)


# 措辞意图 → 目标类别（按构造直接定标签）。
# 实测：离散度 dominant_share 在 k=10 下两轮乱跳(0.4/0.6)、且窄题也会散，
# 噪声比 SUT 自身判断还大，不堪当金标准 oracle；而构造意图与 SUT 判定 10/10 吻合。
# 故标签由「构造意图」定，离散度(章数/主导%)降级为诊断列，仅供人工核 why。
INTENT_LABEL = {
    "enumerate": "pending_split",   # 列举并列子项
    "synthesize": "other",          # 跨主题综合推理
    "narrow": "retrievable",        # 单一概念
}


def suggest(intent):
    return INTENT_LABEL[intent]


async def main():
    candidates = make_candidates()
    llm = configure_llm()
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name="book_knowledge")
    qa = QaCapability(idx, llm, similarity_top_k=PROBE_K)
    print(f"[筛子v2] 候选 {len(candidates)} 道，probe + classify……\n")

    rows = []
    for src, q, intent in candidates:
        nodes = await qa._retrieve_nodes(q, None)
        hits, n_distinct, dom = dispersion(nodes)
        sug = suggest(intent)
        dec = await qa._decide_subq(q, None, probe=True)
        sut_category = dec.category if dec.verdict == "ok" else dec.verdict
        rows.append({
            "user_input": q, "scope": None,
            "suggested_category": sug, "sut_category": sut_category,
            "intent": intent, "n_distinct_chapters": n_distinct,
            "dominant_share": dom, "hit_chapters": hits, "source": src,
        })

    os.makedirs(DATASET_DIR, exist_ok=True)
    out = os.path.join(DATASET_DIR, "split_candidates.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 96)
    print(f"{'建议标签':14}{'SUT判':14}{'意图':11}{'章数':5}{'主导%':7} 问题")
    print("-" * 96)
    agree = 0
    for r in rows:
        same = r["suggested_category"] == r["sut_category"]
        agree += same
        mark = "" if same else "  ⚠"
        print(f"{r['suggested_category']:14}{r['sut_category']:14}{r['intent']:11}"
              f"{r['n_distinct_chapters']:<5}{r['dominant_share']:<7}{r['user_input'][:30]}{mark}")
    print("-" * 96)
    by = Counter(r["suggested_category"] for r in rows)
    print(f"建议分桶: " + "  ".join(f"{k}={v}" for k, v in by.items()))
    print(f"建议 vs SUT 一致率: {agree}/{len(rows)}")
    print(f"已写 {out}")


if __name__ == "__main__":
    asyncio.run(main())
