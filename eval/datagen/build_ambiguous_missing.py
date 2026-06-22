"""补 ambiguous / missing_info 两类金标准候选（闭环 classify 验证）。

这两类机制与 split/other/retrievable 不同——不看召回离散度，看【问句语义形态】：
  - ambiguous   ：真·在库的具体概念 + 缺评判维度的开放问法（probe 命中相关且集中，但缺角度）
  - missing_info：probe 落空/全无关
      · 子型A 指代不明：悬空指代，单轮无上文可补（直接喂 QA judge，不经 Layer1 消解）
      · 子型B 库外内容：两本书都没有的话题 → probe 捞不到相关内容

标签按【构造意图】定，跑真实 classify() 记 sut_category，⚠不一致交人工裁定。
诊断列换成 probe 命中数（ambiguous 应命中相关；missing_info 应落空/无关）。

产出：dataset/ambiguous_missing_candidates.jsonl + 控制台表。

运行：python -m eval.datagen.build_ambiguous_missing
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

PROBE_K = 5

# ── ambiguous：真·在库概念 + 缺维度问法 ──────────────────────────
AMBIGUOUS = [
    "InnoDB的自适应哈希索引好用吗？",
    "建表用InnoDB还是MyISAM好？",
    "MySQL大表查询慢怎么优化？",          # checklist 自带例：场景具体但多角度
    "给表加联合索引好不好？",
    "用COMPACT行格式还是Dynamic行格式好？",
]

# ── missing_info 子型A：悬空指代（单轮无上文）──────────────────
MISSING_REF = [
    "上面说的那个参数怎么配？",
    "它和前面提到的那个有什么区别？",
    "刚才那个怎么用？",
    "这个的适用场景是什么？",
]

# ── missing_info 子型B：库外内容（两本书都没有）────────────────
MISSING_OOB = [
    "PostgreSQL的MVCC是怎么实现的？",
    "MongoDB的分片机制是怎样的？",
    "Oracle的RAC架构是什么？",
    "Cassandra的一致性级别有哪些？",
]

INTENT_LABEL = {
    "ambiguous": "ambiguous",
    "missing_ref": "missing_info",
    "missing_oob": "missing_info",
}


def make_candidates():
    out = []
    for q in AMBIGUOUS:
        out.append(("缺维度", q, "ambiguous"))
    for q in MISSING_REF:
        out.append(("悬空指代", q, "missing_ref"))
    for q in MISSING_OOB:
        out.append(("库外", q, "missing_oob"))
    return out


def probe_summary(nodes):
    """命中数 + top 命中章（ambiguous 应相关集中；missing 应落空/无关）。"""
    tops = []
    for n in nodes:
        p = chapter_number((getattr(n, "metadata", {}) or {}).get("chapter", "") or "")
        if p:
            tops.append(p[0])
    c = Counter(tops)
    top = c.most_common(1)[0][0] if c else "-"
    return len(nodes), top


async def main():
    candidates = make_candidates()
    llm = configure_llm()
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name="book_knowledge")
    qa = QaCapability(idx, llm, similarity_top_k=PROBE_K)
    print(f"[补类] 候选 {len(candidates)} 道（ambiguous/missing），probe + classify……\n")

    rows = []
    for src, q, intent in candidates:
        nodes = await qa._retrieve_nodes(q, None)
        n_hit, top = probe_summary(nodes)
        sug = INTENT_LABEL[intent]
        dec = await qa._decide_subq(q, None, probe=True)
        sut_category = dec.category if dec.verdict == "ok" else dec.verdict
        rows.append({
            "user_input": q, "scope": None,
            "suggested_category": sug, "sut_category": sut_category,
            "intent": intent, "probe_hits": n_hit, "top_chapter": top,
            "reason": dec.reason, "clarify": dec.clarify_question, "source": src,
        })

    os.makedirs(DATASET_DIR, exist_ok=True)
    out = os.path.join(DATASET_DIR, "ambiguous_missing_candidates.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 96)
    print(f"{'建议标签':14}{'SUT判':14}{'意图':13}{'命中':5} 问题")
    print("-" * 96)
    agree = 0
    for r in rows:
        same = r["suggested_category"] == r["sut_category"]
        agree += same
        mark = "" if same else "  ⚠"
        print(f"{r['suggested_category']:14}{r['sut_category']:14}{r['intent']:13}"
              f"{r['probe_hits']:<5}{r['user_input'][:26]}{mark}")
    print("-" * 96)
    by = Counter(r["suggested_category"] for r in rows)
    print(f"建议分桶: " + "  ".join(f"{k}={v}" for k, v in by.items()))
    print(f"建议 vs SUT 一致率: {agree}/{len(rows)}")
    print(f"已写 {out}")


if __name__ == "__main__":
    asyncio.run(main())
