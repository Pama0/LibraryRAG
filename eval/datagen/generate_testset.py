"""从 book chroma 切片用 ragas TestsetGenerator 生成测试集草稿。

走"复用已切块"路线：把 chroma 片段包成 LangChain Document，喂 generate_with_chunks。
产出 dataset/testset.draft.jsonl，人工校验后另存为 testset.jsonl 供 compare 使用。
运行（项目根目录）：python -m eval.datagen.generate_testset --size 50
"""
import argparse
import asyncio
import json
import os
import random

from langchain_core.documents import Document as LCDocument


def sample_chunks(chunks: list, max_chunks: int | None, seed: int = 42) -> list:
    """chunk 数超过 max_chunks 时固定种子随机抽样，否则原样返回。

    ragas 建知识图谱的成本随喂入 chunk 数线性增长（与 testset_size 无关），
    小规模生成时必须抽样，否则会对全量 chunk 跑多遍 LLM。
    """
    if max_chunks is None or len(chunks) <= max_chunks:
        return chunks
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(chunks)), max_chunks))
    return [chunks[i] for i in idx]


def filter_by_book(documents: list[str], metadatas: list[dict],
                   book_filter: str | None) -> tuple[list[str], list[dict]]:
    """按 book_title 子串（不区分大小写）筛选并行的 (正文, 元数据) 列表。

    book_filter 为 None 时原样返回；book_knowledge 可能混多本书，用它只取目标书。
    """
    if not book_filter:
        return documents, metadatas
    needle = book_filter.lower()
    docs, metas = [], []
    for d, m in zip(documents, metadatas):
        title = str((m or {}).get("book_title", "")).lower()
        if needle in title:
            docs.append(d)
            metas.append(m)
    return docs, metas


# B: prompt 末尾追加的中文输出约束
_CHINESE_CONSTRAINT = (
    "\n\n【输出语言】你必须用中文生成问题。"
    "专有名词（如MySQL、InnoDB、B+树）保留英文，但整个句子必须是中文。"
    "用户是中文读者，所有 query 必须用中文表达。"
)


def _force_chinese_output(synthesizer) -> None:
    """在 synthesizer 所有 prompt 模板末尾追加中文输出约束。

    优先走 ragas 标准接口 get_prompts/set_prompts；
    回退遍历实例属性中的长字符串（prompt 模板识别），防重复追加。
    """
    # 标准接口
    if hasattr(synthesizer, "get_prompts") and hasattr(synthesizer, "set_prompts"):
        prompts = synthesizer.get_prompts()
        modified = {}
        for key, value in prompts.items():
            if isinstance(value, str) and _CHINESE_CONSTRAINT not in value:
                modified[key] = value + _CHINESE_CONSTRAINT
            else:
                modified[key] = value
        synthesizer.set_prompts(**modified)
        return
    # 回退：遍历实例属性
    for attr, val in list(synthesizer.__dict__.items()):
        if attr.startswith("_"):
            continue
        if isinstance(val, str) and len(val) > 50 and _CHINESE_CONSTRAINT not in val:
            try:
                setattr(synthesizer, attr, val + _CHINESE_CONSTRAINT)
            except AttributeError:
                pass


def chunks_to_langchain(documents: list[str], metadatas: list[dict]) -> list[LCDocument]:
    """把 chroma 的 (正文, 元数据) 逐条包成 LangChain Document，跳过空文本。"""
    out: list[LCDocument] = []
    for text, meta in zip(documents, metadatas):
        if not text or not text.strip():
            continue
        out.append(LCDocument(
            page_content=text,
            metadata={
                "book_title": (meta or {}).get("book_title", ""),
                "chapter": (meta or {}).get("chapter", ""),
                "page": (meta or {}).get("page", ""),
                "file_path": (meta or {}).get("file_path", ""),
            },
        ))
    return out


def load_book_chunks(max_chunks: int | None = None, seed: int = 42,
                     book_filter: str | None = None) -> list[LCDocument]:
    """从项目 chroma 取 book 切片并转 LangChain Document；可按书名过滤、限喂入量。

    直接用 chromadb 读原始文本+元数据，绕开 RAGIndexManager——后者在 collection
    有数据时会急切构建 VectorStoreIndex（需要全局 embed_model），而此处只读不检索。
    """
    import chromadb

    from eval.config import CHROMA_COLLECTION, CHROMA_DIR

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    docs, metas = filter_by_book(data["documents"], data["metadatas"], book_filter)
    chunks = chunks_to_langchain(docs, metas)
    sampled = sample_chunks(chunks, max_chunks, seed)
    print(f"从 chroma 加载 {len(chunks)} 条 book 切片"
          f"（book_filter={book_filter!r}），喂入 {len(sampled)} 条（max_chunks={max_chunks}）")
    return sampled


async def generate(size: int, max_chunks: int | None = 60, seed: int = 42,
                   book_filter: str | None = None) -> None:
    from ragas.testset import TestsetGenerator
    from ragas.testset.persona import Persona
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

    from eval.config import DATASET_DIR, TESTSET_DRAFT_PATH, make_eval_embeddings, make_eval_llm

    chunks = load_book_chunks(max_chunks=max_chunks, seed=seed, book_filter=book_filter)
    if not chunks:
        raise SystemExit("chroma 无 book 切片，先入库（python main.py 入库流程）再生成测试集")

    gen_llm = make_eval_llm()
    gen_emb = make_eval_embeddings()

    # A: Persona 强约束中文输出
    personas = [
        Persona(
            name="chinese_tech_reader",
            role_description=(
                "正在阅读中文技术书籍的中国工程师。"
                "你提出的所有问题都必须使用中文。"
                "问题中出现的专有名词（如MySQL、InnoDB、B+树、Buffer Pool）保留原文，但整个句子必须是中文。"
                "针对书中具体的技术概念、机制、章节提出有据可查的问题。"
            ),
        ),
    ]

    generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb, persona_list=personas)

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=gen_llm), 0.6),
        (MultiHopSpecificQuerySynthesizer(llm=gen_llm), 0.4),
    ]
    # 中文 prompt 适配 + B: prompt 末尾追加输出语言约束
    for query, _ in distribution:
        prompts = await query.adapt_prompts("chinese", llm=gen_llm)
        query.set_prompts(**prompts)
        _force_chinese_output(query)

    print(f"开始生成测试集（{size} 条）……")
    dataset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=size,
        query_distribution=distribution,
    )
    eval_dataset = dataset.to_evaluation_dataset()

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(TESTSET_DRAFT_PATH, "w", encoding="utf-8") as f:
        for sample in eval_dataset.to_list():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"草稿已写入 {TESTSET_DRAFT_PATH}（共 {len(eval_dataset.to_list())} 条）")
    print("[注意] 人工校验后另存为 testset.jsonl，再跑 compare。")


def main():
    parser = argparse.ArgumentParser(description="生成 book RAG 测试集草稿")
    parser.add_argument("--size", type=int, default=50, help="测试集条数")
    parser.add_argument("--max-chunks", type=int, default=60,
                        help="喂入 ragas 的 chunk 上限（建图成本随此线性增长）；传 0 表示全量")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--book", default=None,
                        help="按 book_title 子串过滤（不区分大小写），如 MySQL；缺省用全部书")
    args = parser.parse_args()
    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    asyncio.run(generate(args.size, max_chunks=max_chunks, seed=args.seed, book_filter=args.book))


if __name__ == "__main__":
    main()
