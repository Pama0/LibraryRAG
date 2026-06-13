"""闭环验证：把章节摘要法产出的广题喂进真实 QaCapability.classify()，
看 probe 探测 + 最终 category 是否如预期路由到 pending_split。

对照组放 1 道窄题（应为 retrievable），验证分类器不是"逢题都判 split"。

运行：python -m eval.poc.poc_classify_check
"""
import asyncio

from configs.embedding import configure_embedding
from configs.llm import configure_llm
from core.rag.data_loader import RAGIndexManager
from core.workflow.qa_capability import QaCapability
from eval.config import CHROMA_DIR

# 二轮：对比"单节聚集型广题"(4.3内) vs "跨章分散型广题" vs 经典对照
CASES = [
    ("①单节广题(4.3行格式)",
     "InnoDB的行格式分哪几种？它们在存储CHAR(M)列和处理NULL值时有何不同？"),
    ("②跨章分散广题(优化是一整片)",
     "怎么优化MySQL？"),
    ("③跨章分散广题(Buffer Pool/redo/undo分属不同章)",
     "InnoDB的Buffer Pool、redo日志、undo日志分别起什么作用？"),
    ("④经典pending_split例(讲讲X)",
     "讲讲MySQL的索引"),
    ("⑤GBK对照(我之前断言会判split)",
     "GBK和GB2312字符集在编码方式上有什么区别？"),
    ("⑥窄题对照(期望retrievable)",
     "COMPACT行格式的NULL值列表是做什么用的？"),
]


async def main():
    llm = configure_llm()
    configure_embedding()
    index_manager = RAGIndexManager(persist_dir=CHROMA_DIR, collection_name="book_knowledge")
    qa = QaCapability(index_manager, llm)

    book_titles = None  # 全库判定（更贴近真实、更难）
    for tag, q in CASES:
        # 先单独探测，把 probe 召回信号打出来看
        nodes = await qa._retrieve_nodes(q, book_titles)
        probe = qa._format_probe(nodes, book_titles)
        result = await qa.classify(q, book_titles=book_titles, probe=True)
        print("=" * 72)
        print(f"[{tag}]")
        print(f"  Q: {q}")
        print(f"  probe召回: {len(nodes)} 段 | {probe.splitlines()[0]}")
        print(f"  ▶ category = {result.category}")
        print(f"  reason    = {result.reason}")


if __name__ == "__main__":
    asyncio.run(main())
