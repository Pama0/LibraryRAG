"""
初始化/更新索引脚本
将 data 目录下的文档增量导入 ChromaDB：
- 新文件：自动添加
- 变更文件：删除旧节点后重新索引
- 未变更文件：跳过
"""

from configs.embedding import configure_embedding
from core.rag.data_loader import RAGIndexManager


def init_index():
    """增量更新文档索引"""
    configure_embedding()

    manager = RAGIndexManager()
    manager.add_documents("./data")
    print("更新完成！")


if __name__ == "__main__":
    init_index()
