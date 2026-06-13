"""给 golden.jsonl 补 reference（context_recall / factual_correctness 解锁）。

策略：
  - missing_info  → reference = ""（checklist 规定）
  - 其余类别      → 慷慨检索 top-K（K 大于 SUT 的 5，避免循环：reference 应代表"理想全集"）
                    → DeepSeek 基于召回片段、按【类别形态】生成要点式参考答案 → 写回

按类别定 reference 形态：
  retrievable   单概念几句要点
  pending_split 逐子项罗列（有 A/B/C，分别…）
  other         多概念关系/取舍的综合要点
  ambiguous     列出评判维度 + 每维度要点

安全起见写到 golden.with_ref.jsonl（不覆盖原文件）；人工抽检后再替换 golden.jsonl。

运行：python -m eval.datagen.fill_reference
"""
import asyncio
import json
import os

from openai import AsyncOpenAI

from configs.embedding import configure_embedding
from configs.llm import configure_llm, deepseek_api_key
from core.rag.data_loader import RAGIndexManager
from core.workflow.qa_capability import QaCapability
from eval.config import DATASET_DIR, CHROMA_DIR

GOLDEN = os.path.join(DATASET_DIR, "golden.jsonl")
OUT = os.path.join(DATASET_DIR, "golden.with_ref.jsonl")
REF_K = 15  # 慷慨检索：比 SUT 的 5 多，reference 代表更全的"金标准全集"

# 按类别给生成指令（reference 形态对齐该类应有的答案）
CATEGORY_SHAPE = {
    "retrievable": "针对单一概念，给几句准确要点即可。",
    "pending_split": "逐个子项罗列（如'有 A、B、C，分别…'），覆盖问题涉及的各个实体/子主题。",
    "other": "给跨多个概念的综合要点，点明它们的关系、协作或取舍。",
    "ambiguous": "先列出可选的评判维度/角度，再给每个维度的要点（契合'分维度作答'）。",
}

_PROMPT = """下面是从知识库检索到的片段，请据此为问题写【参考答案要点】，用于评测时和系统答案做语义比对。

要求：
- **只依据片段，不许编造**；片段没有的不要写。
- 要点对即可，不必辞藻完美，但该覆盖的点要全。
- 形态：{shape}
- 中文，专名（MySQL/InnoDB 等）保留英文。

问题：{question}

检索片段：
{context}

直接输出参考答案正文，不要任何前缀说明。"""


def chunk_text(n):
    return (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")) or ""


async def gen_reference(gen, qa, question, category):
    nodes = await qa._retrieve_nodes(question, None)
    context = "\n---\n".join(chunk_text(n)[:600] for n in nodes[:REF_K])
    prompt = (_PROMPT.replace("{shape}", CATEGORY_SHAPE.get(category, "要点式作答。"))
              .replace("{question}", question).replace("{context}", context))
    resp = await gen.chat.completions.create(
        model="deepseek-v4-flash", messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}}, max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()


async def main():
    with open(GOLDEN, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    llm = configure_llm()
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name="book_knowledge")
    qa = QaCapability(idx, llm, similarity_top_k=REF_K)
    gen = AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=deepseek_api_key)

    print(f"[补 reference] {len(rows)} 条，慷慨检索 K={REF_K}……\n")
    for r in rows:
        cat = r.get("category", "")
        if cat == "missing_info":
            r["reference"] = ""
            print(f"[missing_info] {r['user_input'][:30]} → reference=\"\"")
            continue
        r["reference"] = await gen_reference(gen, qa, r["user_input"], cat)
        print("=" * 72)
        print(f"[{cat}] {r['user_input']}")
        print(f"  → {r['reference'][:140]}{'…' if len(r['reference']) > 140 else ''}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n" + "=" * 72)
    print(f"已写 {OUT}")
    print("人工抽检后，确认无误即可：copy golden.with_ref.jsonl → golden.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
