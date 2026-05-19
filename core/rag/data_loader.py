import os

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex

from llama_index.vector_stores.chroma import ChromaVectorStore

from core.rag.parser import ArticleSplitter


class RAGIndexManager:
    """使用 iter_data() 的大规模文档索引管理"""

    def __init__(
            self,
            persist_dir: str = "./chroma_db",
            collection_name: str = "documents",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.db = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.splitter = ArticleSplitter(
            include_chapter_context=False
        )

        if self.chroma_collection.count() > 0:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
        else:
            self.index = None

    def _get_indexed_file_info(self) -> dict:
        """从 ChromaDB 中提取已索引文件的指纹信息

        Returns:
            {file_path: {file_size: str, last_modified_date: str, node_ids: [str]}}
        """
        all_docs = self.chroma_collection.get(include=["metadatas"])
        file_info = {}
        for i, meta in enumerate(all_docs["metadatas"]):
            fpath = meta.get("file_path")
            if not fpath:
                continue
            if fpath not in file_info:
                file_info[fpath] = {
                    "file_size": str(meta.get("file_size", "")),
                    "last_modified_date": str(meta.get("last_modified_date", "")),
                    "node_ids": [],
                }
            file_info[fpath]["node_ids"].append(all_docs["ids"][i])
        return file_info

    def _delete_nodes_by_file(self, file_name: str, node_ids: list):
        """物理删除指定文件的所有节点"""
        self.chroma_collection.delete(ids=node_ids)
        print(f"  删除旧节点: {file_name} ({len(node_ids)} 个)")


    def add_documents(self, data_dir: str, recursive: bool = True):
        """增量添加文档 - 按文件变更检测"""
        indexed_info = self._get_indexed_file_info()

        # 反向清理：数据库中存在但文件已删除的，直接删除
        deleted_count = 0
        for fpath, info in indexed_info.items():
            if not os.path.exists(fpath):
                self._delete_nodes_by_file(fpath, info["node_ids"])
                deleted_count += 1

        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            recursive=recursive,
            exclude=[
                ".venv", "venv", "node_modules", "__pycache__",
                ".git", ".idea", ".claude",
            ],
        )

        file_count = 0
        skip_count = 0
        update_count = 0

        for documents in reader.iter_data(show_progress=True):
            if not documents:
                continue

            # 从第一个 document 的 metadata 获取文件信息
            doc_meta = documents[0].metadata
            fpath = doc_meta.get("file_path", "")
            fsize = str(doc_meta.get("file_size", ""))
            fmodified = doc_meta.get("last_modified_date", "")

            # 对比指纹：file_size + last_modified_date
            existing = indexed_info.get(fpath)
            if existing and existing["file_size"] == fsize and existing["last_modified_date"] == fmodified:
                skip_count += 1
                continue

            # 文件变更：先删旧节点
            if existing:
                self._delete_nodes_by_file(fpath, existing["node_ids"])
                update_count += 1
            else:
                print(f"  新增文件: {fpath}")

            # 切片
            nodes = self.splitter.get_nodes_from_documents(documents)

            # 索引
            if self.index is None:
                self.index = VectorStoreIndex(
                    nodes,
                    storage_context=self.storage_context,
                )
            else:
                self.index.insert_nodes(nodes)

            file_count += 1

        print(f"处理完成: 新增/更新 {file_count} 个文件, 跳过 {skip_count} 个未变更, 清理 {deleted_count} 个已删除, {self.chroma_collection.count()} 个向量")
        return self.index

    def get_query_engine(self, llm, similarity_top_k: int = 3):
        """获取查询引擎"""
        if self.index is None:
            raise ValueError("索引为空，请先添加文档")
        return self.index.as_query_engine(llm=llm, similarity_top_k=similarity_top_k)

    def get_index(self):
        """获取索引对象"""
        return self.index
