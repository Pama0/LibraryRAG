"""单章 PoC：章节摘要法生成「广题 + 多子节参考答案」。

验证思路（不接入主流程，纯打印给人看质量）：
  ① 按 (书, 章前缀) 聚合 chunk —— 锚定整章而非单 chunk
  ② 按子节组织正文（保留子主题结构，这是"广"的信号）
  ③ 一次 LLM 调用产出：子主题骨架 + 1~2 道广题 + 多子节参考答案

运行：python -m eval.poc.poc_chapter_summary
"""
import asyncio
import json

import chromadb
from openai import AsyncOpenAI

from configs.llm import deepseek_api_key
from core.workflow.chapter_tree import chapter_number
from eval.config import CHROMA_COLLECTION, CHROMA_DIR

BOOK = "MySQL是怎样运行的：从根儿上理解MySQL"
PREFIX = (4, 3)  # 4.3 InnoDB行格式（5 子节，~20 chunk）


def load_chapter(prefix: tuple[int, ...]) -> list[tuple[str, str]]:
    """取该书在 prefix 下的所有 chunk，返回 [(子节标题, 正文)]，按章节序。"""
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


def build_structured_context(rows: list[tuple[str, str]]) -> str:
    """按子节聚合正文，保留 ## 子节标题结构，喂 LLM 出题。"""
    by_sec: dict[str, list[str]] = {}
    for chap, doc in rows:
        by_sec.setdefault(chap, []).append(doc)
    parts = []
    for chap, docs in by_sec.items():
        parts.append(f"## {chap}\n" + "\n".join(docs))
    return "\n\n".join(parts)


_GEN_PROMPT = """下面是技术书《{book}》某一章「{chapter}」的完整正文，已按子节用 ## 标出结构。

你的任务：基于整章内容，产出"需要综合多个子节才能答全"的广覆盖问题，用于测试 RAG 系统的跨子节综合能力。严禁出只看单个子节就能答的窄问题。

请输出 JSON（不要任何多余内容）：
{
  "subtopics": ["该章涵盖的子主题清单，逐条，对应各子节核心点"],
  "questions": [
    {
      "user_input": "一道中文广题，答全需覆盖上面多个子主题（如'X有哪几种？各自特点和区别？'）",
      "reference": "参考答案：基于正文，逐个子主题给要点，覆盖面要全。专有名词保留英文，整体中文。",
      "covered_subtopics": ["这道题答案覆盖了哪些子主题"]
    }
  ]
}

要求：
- 出 2 道广题，难度一中一难。
- reference 必须来自正文、不要编；要点对即可，覆盖多个子节。
- 全部用中文（MySQL/InnoDB/COMPACT 等专名保留英文）。

正文：
{context}"""


async def main():
    rows = load_chapter(PREFIX)
    context = build_structured_context(rows)
    chapter_name = f"{PREFIX[0]}.{PREFIX[1]} InnoDB行格式"
    prompt = (
        _GEN_PROMPT.replace("{book}", BOOK)
        .replace("{chapter}", chapter_name)
        .replace("{context}", context)
    )

    client = AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=deepseek_api_key)
    print(f"[PoC] 章={chapter_name} chunk={len(rows)} 正文字数={len(context)}")
    print("[PoC] 调 DeepSeek 生成广题中……\n")
    resp = await client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=3000,
    )
    text = resp.choices[0].message.content
    data = json.loads(text)

    print("=" * 70)
    print("【子主题骨架】（该章应被覆盖的点）")
    for i, s in enumerate(data.get("subtopics", []), 1):
        print(f"  {i}. {s}")
    for i, q in enumerate(data.get("questions", []), 1):
        print("=" * 70)
        print(f"【广题 {i}】{q['user_input']}")
        print(f"  覆盖子主题: {q.get('covered_subtopics')}")
        print(f"  参考答案: {q['reference']}")


if __name__ == "__main__":
    asyncio.run(main())
