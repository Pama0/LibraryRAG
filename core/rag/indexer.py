from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
import os

def build_index(nodes, persist_dir: str = "./storage"):
	"""构建并持久化索引"""
	# 确保存储目录存在
	os.makedirs(persist_dir, exist_ok=True)

	# 创建带存储上下文的索引
	storage_context = StorageContext.from_defaults()  # 创建新上下文
	index = VectorStoreIndex(
		nodes,
		storage_context=storage_context  # 关键：绑定上下文
	)

	# 持久化到指定目录
	storage_context.persist(persist_dir=persist_dir)
	return index

def load_index(persist_dir: str = "./storage"):
    """加载已有索引"""
    # 必须重建存储上下文
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)
