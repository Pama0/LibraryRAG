import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
from tqdm import tqdm


class RAGIndexManager:
    """大规模文档增量索引管理"""

    def __init__(
            self,
            persist_dir: str = "./chroma_db",
            collection_name: str = "documents",
            chunk_size: int = 512,
            chunk_overlap: int = 50,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.db = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 加载已有索引
        if self.chroma_collection.count() > 0:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
        else:
            self.index = None

    def _get_supported_files(self, data_dir: str) -> list[str]:
        """获取支持的文件列表"""
        supported_extensions = {
            '.txt', '.pdf', '.docx', '.doc', '.md',
            '.html', '.csv', '.json', '.xml'
        }
        files = []
        for f in os.listdir(data_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_extensions:
                files.append(os.path.join(data_dir, f))
        return sorted(files)

    def add_documents(self, data_dir: str, batch_size: int = 50):
        """分批增量添加文档"""
        files = self._get_supported_files(data_dir)
        total_files = len(files)

        if total_files == 0:
            print(f"目录 {data_dir} 中没有支持的文档")
            return

        print(f"发现 {total_files} 个文档，分 {(total_files + batch_size - 1) // batch_size} 批处理")

        # 分批处理文件
        for batch_idx in tqdm(
                range(0, total_files, batch_size),
                desc="Processing batches"
        ):
            batch_files = files[batch_idx:batch_idx + batch_size]

            # 只加载当前批次的文档
            documents = SimpleDirectoryReader(input_files=batch_files).load_data()

            if len(documents) == 0:
                continue

            # 切片
            nodes = self.splitter.get_nodes_from_documents(documents)

            # 建立或更新索引
            if self.index is None:
                self.index = VectorStoreIndex(
                    nodes,
                    storage_context=self.storage_context,
                    show_progress=True
                )
            else:
                # 增量插入
                self.index.insert_nodes(nodes)

            # 释放内存
            del documents
            del nodes

        print(f"处理完成，共索引 {self.chroma_collection.count()} 个向量")
        return self.index


def get_query_engine(self, llm, similarity_top_k: int = 3, doc_type: str = None):
    """获取查询引擎"""
    if self.index is None:
        raise ValueError("索引为空，请先添加文档")

    if doc_type:
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
        filters = MetadataFilters(
            filters=[MetadataFilter(key="doc_type", value=doc_type)]
        )
        return self.index.as_query_engine(
            llm=llm,
            similarity_top_k=similarity_top_k,
            filters=filters
        )

    return self.index.as_query_engine(llm=llm, similarity_top_k=similarity_top_k)
