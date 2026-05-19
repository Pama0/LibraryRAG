import asyncio
import os
import random
from dotenv import load_dotenv

load_dotenv()

import chromadb
from langchain_core.documents import Document as LCDocument
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona

from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)

# ── 配置 ──────────────────────────────────────────────
CHROMA_PERSIST_DIR = "../chroma_db"
CHROMA_COLLECTION = "documents"
TESTSET_SIZE = 50
TARGET_FILES = 50  # 随机选取的法规文件数
SAMPLES_PER_FILE = 3  # 每个文件取 3 条，50 × 3 = 150 条输入
RANDOM_SEED = 42


def load_chunks_from_chroma() -> list[LCDocument]:
    """从 ChromaDB 加载已切好的法律条文，转为 LangChain Document"""
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = db.get_or_create_collection(CHROMA_COLLECTION)

    # 全量取出
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    # 按 file_name 分组
    file_groups: dict[str, list[int]] = {}
    for i, meta in enumerate(metas):
        fname = meta.get("file_name", "unknown")
        file_groups.setdefault(fname, []).append(i)

    # 先随机选 TARGET_FILES 个法规文件，每个取 SAMPLES_PER_FILE 条
    random.seed(RANDOM_SEED)
    selected_files = random.sample(list(file_groups.keys()), min(TARGET_FILES, len(file_groups)))
    sampled_indices: list[int] = []
    for fname in selected_files:
        indices = file_groups[fname]
        sampled_indices.extend(random.sample(indices, min(SAMPLES_PER_FILE, len(indices))))

    # 上下文前缀已在切片时拼入文本，直接使用
    chunks: list[LCDocument] = []
    for i in sampled_indices:
        meta = metas[i]
        chunks.append(LCDocument(
            page_content=docs[i],
            metadata={
                "law_name": meta.get("file_name", ""),
                "article_no": meta.get("article_no", ""),
                "chapter": meta.get("chapter", ""),
                "chapter_title": meta.get("chapter_title", ""),
            },
        ))

    print(f"从 ChromaDB 加载 {len(sampled_indices)} 条法律条文（来自 {len(file_groups)} 个法规文件）")
    return chunks


async def generate_testset():
    # 1. 从 ChromaDB 加载已切好的数据
    chunks = load_chunks_from_chroma()

    # 2. 配置 LLM（评测用，不复用 RAG 的 LLM）
    api_key = os.getenv('ZHIPU_API_KEY')
    import openai
    client = openai.AsyncOpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=api_key,
    )
    generator_llm = llm_factory("glm-4-flash", client=client)

    # 3. Embedding
    generator_embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-zh-v1.5")

    # 4. Persona：法律领域测试
    personas = [
        Persona(
            name="legal_practitioner",
            role_description="针对具体法律条文提出实务问题，测试RAG系统对法律规定的检索准确性",
        ),
    ]

    # 5. Transforms：150 条输入量可接受默认 transform（NER/Summary/Theme）
    transforms = None

    # 6. 生成器
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
    )

    # 7. 问题分布：60% 单跳 + 40% 多跳
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.6),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.4),
    ]

    # 适配中文 prompt
    for query, _ in distribution:
        prompts = await query.adapt_prompts("chinese", llm=generator_llm)
        query.set_prompts(**prompts)

    # 8. 生成测试集（使用 generate_with_chunks，直接喂已切好的块）
    print(f"开始生成测试集（{TESTSET_SIZE} 条）")
    dataset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=TESTSET_SIZE,
        transforms=transforms,
        query_distribution=distribution,
    )
    eval_dataset = dataset.to_evaluation_dataset()

    # 9. 导出
    df = eval_dataset.to_pandas()
    df.to_csv("./dataset/testset.csv", index=False, encoding='utf-8-sig')
    print("CSV 导出完成")

    import json
    with open("./dataset/testset.jsonl", "w", encoding="utf-8") as f:
        for sample in eval_dataset.to_list():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("JSONL 导出完成")


if __name__ == "__main__":
    asyncio.run(generate_testset())
