"""章节摘要法【完整闭环】PoC：整章锚定生成广题 → 喂真实 _decide_subq → 看是否触发 split。

相比 poc_chapter_summary（只生成）+ poc_classify_check（只分类），这里串成一条：
  整章聚合 → DeepSeek 生成广题 → QaCapability._decide_subq() → 打印 category。

锚定"横切主题"的整章（如第6章 索引），其内容在全库分散，召回天然离散 → 期望 pending_split。

运行：python -m eval.poc.poc_chapter_loop
"""
import asyncio
import json

import chromadb
from openai import AsyncOpenAI

from configs.embedding import configure_embedding
from configs.llm import configure_llm, deepseek_api_key
from core.rag.data_loader import RAGIndexManager
from core.workflow.chapter_tree import chapter_number
from core.workflow.qa_capability import QaCapability
from eval.config import CHROMA_COLLECTION, CHROMA_DIR

BOOK = "MySQL是怎样运行的：从根儿上理解MySQL"
PREFIX = (6,)  # 整个第6章「B+树索引」——横切主题，全库分散
CHAPTER_NAME = "第6章 快速查询的秘籍-B+树索引"


def load_chapter(prefix):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(CHROMA_COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    rows = []
    for doc, m in zip(data["documents"], data["metadatas"]):
        if not m or m.get("book_title") != BOOK:
            continue
        p = chapter_number(m.get("chapter") or "")
        if p and p[: len(prefix)] == prefix:
            rows.append(((m.get("chapter") or "").strip(), doc))
    rows.sort(key=lambda r: chapter_number(r[0]) or ())
    return rows


def build_context(rows):
    by_sec = {}
    for chap, doc in rows:
        by_sec.setdefault(chap, []).append(doc)
    return "\n\n".join(f"## {c}\n" + "\n".join(d) for c, d in by_sec.items())


_GEN_PROMPT = """下面是技术书《{book}》整章「{chapter}」的正文，按子节用 ## 标出。

基于【整章】产出**概览/综述式**的宽泛大问题，用于测试 RAG 的"宽问题拆解"能力。

【关键】要的是**笼统、扫一整片**的问题，不是聚焦某两个概念的细节对比。
- 好的（要这种）：「讲讲MySQL的索引体系」「InnoDB的索引都有哪些类型和方案」「索引这块整体是怎么设计的」「怎么用好索引」
- 不要的（太具体会被判窄）：「聚簇索引和二级索引在页分裂上的区别」这种锁定2个概念的细节题

输出 JSON（无多余内容）：
{
  "questions": ["3 道中文概览式宽问题，每道都笼统覆盖整章主题、答全需罗列多个子项"]
}
专名（MySQL/InnoDB/B+树）保留英文，整体中文。

正文：
{context}"""


async def main():
    rows = load_chapter(PREFIX)
    context = build_context(rows)
    print(f"[闭环PoC] 锚定={CHAPTER_NAME} chunk={len(rows)} 字数={len(context)}\n")

    # ① 整章生成广题（DeepSeek）
    gen = AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=deepseek_api_key)
    prompt = (_GEN_PROMPT.replace("{book}", BOOK)
              .replace("{chapter}", CHAPTER_NAME).replace("{context}", context))
    resp = await gen.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}}, max_tokens=1500,
    )
    questions = json.loads(resp.choices[0].message.content).get("questions", [])

    # ② 装配真实 QA，逐题闭环分类
    llm = configure_llm()
    configure_embedding()
    idx = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name="book_knowledge")
    qa = QaCapability(idx, llm)

    print("=" * 72)
    for i, q in enumerate(questions, 1):
        nodes = await qa._retrieve_nodes(q, None)
        chaps = {chapter_number(getattr(n, "metadata", {}).get("chapter", "") or "")[:1]
                 for n in nodes if chapter_number(getattr(n, "metadata", {}).get("chapter", "") or "")}
        dec = await qa._decide_subq(q, None, probe=True)
        sut_category = dec.category if dec.verdict == "ok" else dec.verdict
        flag = "✓触发split" if sut_category == "pending_split" else "·"
        print(f"[广题{i}] {q}")
        print(f"   召回命中章: {sorted(c[0] for c in chaps)}  →  category={sut_category} {flag}")
        if dec.reason:
            print(f"   reason: {dec.reason}")
        print("-" * 72)


if __name__ == "__main__":
    asyncio.run(main())
