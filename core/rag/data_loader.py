import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.rag.pdf_parser import BookPDFParser


class RAGIndexManager:
    """使用 iter_data() 的大规模文档索引管理"""

    def __init__(
            self,
            persist_dir: str = "./chroma_db",
            collection_name: str = "book_knowledge",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.db = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        if self.chroma_collection.count() > 0:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
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


    def add_book(
        self,
        pdf_path: str,
        book_title: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        """入库单本技术书籍 PDF

        Args:
            pdf_path: PDF 文件路径
            book_title: 书名，如《深入理解MySQL核心技术》
            chunk_size: 分块大小（字符数）
            chunk_overlap: 块间重叠字符数

        Returns:
            index 对象，已创建或更新
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        # 检查是否已入库（基于文件路径）
        indexed_info = self._get_indexed_file_info()
        existing = indexed_info.get(pdf_path)
        if existing:
            self._delete_nodes_by_file(pdf_path, existing["node_ids"])
            print(f"  {book_title} 已有索引，重建中...")

        print(f"  解析 PDF: {book_title} ...")
        parser = BookPDFParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = parser.parse(pdf_path, book_title)
        print(f"    检测到 {len(documents)} 个文档块")

        # 二次切分
        text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        nodes = text_splitter.get_nodes_from_documents(documents)
        print(f"    切分为 {len(nodes)} 个节点")

        # 入库
        if self.index is None:
            self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        else:
            self.index.insert_nodes(nodes)

        print(f"  {book_title} 入库完成，向量总数: {self.chroma_collection.count()}")
        return self.index

    def add_book_quick(
        self,
        pdf_path: str,
        book_title: str,
    ):
        """快速入库：跳过章节检测，仅按页分块（作为降级方案）

        适用于章节字体特征不明显或扫描版 PDF（已 OCR）。
        """
        import fitz

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        # 检查重复
        indexed_info = self._get_indexed_file_info()
        existing = indexed_info.get(pdf_path)
        if existing:
            self._delete_nodes_by_file(pdf_path, existing["node_ids"])

        print(f"  快速解析 PDF: {book_title} ...")
        doc = fitz.open(pdf_path)
        documents = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text.strip():
                continue
            documents.append(Document(
                text=text,
                metadata={
                    "book_title": book_title,
                    "chapter": "",
                    "page": page_num + 1,
                    "content_type": "text",
                    "file_path": pdf_path,
                    "chunk_type": "book_chunk",
                },
            ))
        doc.close()

        text_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
        nodes = text_splitter.get_nodes_from_documents(documents)
        print(f"    切分为 {len(nodes)} 个节点 ({len(documents)} 页)")

        if self.index is None:
            self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
        else:
            self.index.insert_nodes(nodes)

        print(f"  {book_title} 入库完成，向量总数: {self.chroma_collection.count()}")
        return self.index

    def get_query_engine(self, llm, similarity_top_k: int = 3):
        """获取查询引擎"""
        if self.index is None:
            raise ValueError("索引为空，请先添加文档")
        return self.index.as_query_engine(llm=llm, similarity_top_k=similarity_top_k)

    def get_index(self):
        """获取索引对象"""
        return self.index

