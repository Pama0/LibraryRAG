"""
单文件 RAG 测试脚本
- 使用独立的 chroma_db_single_test 数据库，不影响主流程
- 读取 test_data/ 中的单文件
- 支持交互式查询和一次性查询
"""
import asyncio
import sys

from configs.llm import configure_llm
from configs.embedding import configure_embedding
from core.rag.data_loader import RAGIndexManager
from core.workflow.simple_rag import SimpleRagWorkflow

# 独立数据库路径，与主流程隔离
TEST_CHROMA_DIR = "./chroma_db_single_test"
TEST_DATA_DIR = "./test_data"
COLLECTION_NAME = "single_file_test"


def setup():
    """初始化配置和索引"""
    llm = configure_llm()
    configure_embedding()

    manager = RAGIndexManager(
        persist_dir=TEST_CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    index = manager.add_documents(data_dir=TEST_DATA_DIR)
    if index is None:
        print("索引为空，请检查 test_data/ 目录是否有文件")
        sys.exit(1)

    return llm, index, manager


async def run_queries(llm, index, queries: list[str]):
    """执行一组查询"""
    workflow = SimpleRagWorkflow(index=index, llm=llm)

    for q in queries:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print('-'*60)
        result = await workflow.run(query=q)
        print(f"回答: {result}")
    print(f"\n{'='*60}")


async def interactive_mode(llm, index):
    """交互式查询"""
    workflow = SimpleRagWorkflow(index=index, llm=llm)
    print("\n单文件 RAG 测试 (输入 q 退出)")
    print("="*60)

    while True:
        q = input("\n问题: ").strip()
        if q.lower() in ('q', 'quit', 'exit'):
            break
        if not q:
            continue

        result = await workflow.run(query=q)
        print(f"\n回答: {result}")


def cleanup():
    """删除测试数据库"""
    import shutil
    import os
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR)
        print(f"已清理测试数据库: {TEST_CHROMA_DIR}")


if __name__ == "__main__":
    # 解析命令行参数
    if "--cleanup" in sys.argv:
        cleanup()
        sys.exit(0)

    llm, index, manager = setup()

    # 查看索引状态
    print(f"\n索引节点数: {manager.chroma_collection.count()}")
    all_docs = manager.chroma_collection.get(include=["metadatas"])
    files = set(m.get("file_path", "") for m in all_docs["metadatas"])
    print(f"索引文件: {files}")


    asyncio.run(interactive_mode(llm, index))
