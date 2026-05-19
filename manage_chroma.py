import chromadb

db = chromadb.PersistentClient("./chroma_db")
collection = db.get_collection("documents")

# 打印统计信息
print(f"集合名称: {collection.name}")
print(f"文档数量: {collection.count()}")
print("-" * 50)

# 获取所有文档
all_docs = collection.get(include=["documents", "metadatas"])

for i, doc in enumerate(all_docs['documents'],10):
    print(f"[{i+1}] ID: {all_docs['ids'][i]}")
    print(f"    内容: {doc[:100]}..." if len(doc) > 100 else f"    内容: {doc}")
    print(f"    元数据: {all_docs['metadatas'][i]}")
    print()
# from llama_index.core import SimpleDirectoryReader
#
# documents = SimpleDirectoryReader("./data").load_data()
# print("文档数量：", len(documents))
# print("第一个文档的前200字符：", documents[0].text[:200])
# print("第二个文档的前200字符：", documents[1].text[:200])