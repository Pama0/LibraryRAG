# 可插拔 Retriever：Hybrid（dense + BM25）检索策略（第一版）

**日期**：2026-06-16
**状态**：设计已确认，待写实现 plan

## 背景与动机

可插拔检索组件方向的第二个增量。第一个增量（[Reranker](2026-06-16-pluggable-reranker-design.md)）
治**精度**（候选排序，`context_precision ≈ 0.37`）。本增量治**召回**——有没有把对的候选捞上来。

接缝沿用 reranker 那条已验证的链路：`make_X(name)` 工厂 + `QaCapability` 注入 +
`DocQueryWorkflow` 按名解析 + eval `VARIANTS` 透传。本版只新增 **Retriever 一个可注入类别**。

### 为什么是 Hybrid，不是 HyDE

最初候选是 HyDE（LLM 生成假想答案再 embed 检索）。但本知识库的**设计前提是「LLM 不认识
用户问的书本专有名词、只能靠 RAG」**——这恰好抽掉了 HyDE 的工作前提（它依赖 LLM 的参数知识
编出语义邻近的假想文档）。在该前提下 HyDE 会生成幻觉或泛泛文本，把检索 embedding 拽偏，
**召回反而变差**。

对「私有知识库 + 书本专有名词」，稳健的召回杠杆是**词面匹配**：用户打出的术语在书里逐字出现，
**BM25 直接 token 对 token 命中，不经过 LLM、不需要任何人「认识」这个词**。dense embedder
（中文 bge 类）对罕见/专有术语本身也可能表示不好（近似 OOV），正是 BM25 补上。故第一个
「非 vector」策略定为 **Hybrid = dense + BM25（RRF 融合）**。

## 范围

- ✅ `Retriever` 协议 + `VectorRetriever`（=基线）+ `HybridRetriever` + `make_retriever` 工厂。
- ✅ `build_book_filters` 共享工具（从 `QaCapability._make_filters` 上移）。
- ✅ 接入 `QaCapability._retrieve_nodes`；`DocQueryWorkflow` / eval `VARIANTS` 透传。
- ❌ 不做 HyDE / 纯 BM25 / 其它检索策略——注册表加一行的后续事。
- ❌ 不做 dedup / filter 等后续 stage；不造通用 `RetrievalPipeline` 容器（编排仍留
   `_retrieve_nodes`；待变换增多、编排臃肿时再提容器）。

## 设计

### 1. 依赖选型：rank_bm25 + jieba（纯 Python）

**不用** LlamaIndex 的 `llama-index-retrievers-bm25`（拉 `bm25s` + `PyStemmer`，后者 C 扩展、
Windows 易踩编译坑，tokenizer API 跨版本不稳）。改用：

- **`rank_bm25`**（纯 Python `BM25Okapi`）：`BM25Okapi(tokenized_corpus)` 建索引，
  `get_scores(tokenized_query)` 打分。API 稳定、零原生编译。
- **`jieba`**（纯 Python 中文分词）：`jieba.lcut(text)` → token 列表。

融合（RRF）自写。换取：零编译风险、API 稳定、可测性最好；代价是 BM25 索引/打分自管（~30 行
纯逻辑，可单测）。

### 2. 接口：源策略，依赖 call 时传

`Retriever` 是**数据源**（不像 reranker 是变换 `(query,nodes)→nodes`）。为让策略对象自身无依赖、
由 `make_retriever(name)` 零参构造（与 `make_reranker` 对称），依赖在**调用时**显式传入：

```python
@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, *, index_manager, book_titles, top_k: int) -> list: ...
```

- 不传 `llm`（HyDE 已弃，本版无 LLM 依赖；未来需要时再加，YAGNI）。
- 策略需要的 dense 索引与 BM25 语料都从 `index_manager` 取（`.get_index()` / `.chroma_collection`）。
- `book_titles`（scope）传原始域概念，策略各自构造所需过滤。

### 3. 共享工具：build_book_filters（core/retrieval）

把现有 `QaCapability._make_filters` 上移成共享函数（`_retrieve_nodes` 是其唯一调用点）：

```python
def build_book_filters(book_titles):
    """scope 硬约束 → chroma 元数据过滤器；空范围 → None（全库）。"""
    if not book_titles:
        return None
    return MetadataFilters(filters=[
        MetadataFilter(key="book_title", operator=FilterOperator.IN, value=list(book_titles)),
    ])
```

### 4. 两个实现（core/retrieval/retrieve.py）

**`VectorRetriever`**（= 基线，缺省）——等价当前行为：

```python
class VectorRetriever:
    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        retriever = index_manager.get_index().as_retriever(
            similarity_top_k=top_k, filters=build_book_filters(book_titles))
        return await retriever.aretrieve(query)
```

**`HybridRetriever`**——dense + BM25，RRF 融合：

- **BM25 build（懒构造 + 缓存 + 并发守卫）**：首次 retrieve 时从
  `index_manager.chroma_collection.get(include=["documents","metadatas"])` 重建全库
  `TextNode`（text=documents[i]、id_=ids[i]、metadata=metadatas[i]），`jieba.lcut` 分词建
  `BM25Okapi`，缓存到实例（`self._bm25` / `self._nodes`）。用 `asyncio.Lock` 防并发重复构造。
- **dense 分支**：`index.as_retriever(top_k, filters=build_book_filters(book_titles)).aretrieve(query)`。
- **BM25 分支**：`bm25.get_scores(jieba.lcut(query))` 对**全库**打分 → 按
  `metadata.book_title ∈ book_titles`（book_titles 非空时）**后过滤** → 取 top_k。
  （对全量打分后再过滤，不欠收。）
- **融合（自写 RRF）**：dense 列表 + BM25 列表，按 node id 取 `Σ 1/(rrf_k + rank)`（`rrf_k=60`），
  去重排序，截 `top_k`。返回 `NodeWithScore` 列表（与 dense 输出类型一致，下游 rerank/合成不变）。

**`make_retriever`**（按名 memoize，同 `make_reranker`）：

```python
def make_retriever(name):
    """None/"vector" → VectorRetriever；"hybrid" → HybridRetriever；未知 → ValueError。
    按名缓存：eval 每条 query 新建 workflow，memoize 让 HybridRetriever 的 BM25 索引一进程只建一次。"""
```

### 5. 接入 QaCapability

```python
# __init__ 新增：
retriever: "Retriever | None" = None
...
self.retriever = retriever or VectorRetriever()   # 检索不可跳过，基线=具体 VectorRetriever

# _retrieve_nodes：
async def _retrieve_nodes(self, query, book_titles):
    fetch_k = self.rerank_candidate_k if self.reranker else self.similarity_top_k
    nodes = await self.retriever.retrieve(
        query, index_manager=self.index_manager, book_titles=book_titles, top_k=fetch_k)
    if self.reranker:
        nodes = await self.reranker.rerank(query, nodes, self.similarity_top_k)
    return nodes
```

- 删除 `QaCapability._make_filters`（逻辑上移 `build_book_filters`）。
- **现有行为零变化**：不传 retriever → `VectorRetriever` → 原 `as_retriever` 路径，所有现存
  测试（含 reranker 那批）照过。`HybridRetriever + reranker` 天然组合（hybrid 过召回 20 →
  rerank 截 5）。

### 6. 装配链路（DocQueryWorkflow）

```python
# __init__ 新增 retriever: str | None = None（与 reranker 并列，注明为具名可插拔组件）：
self.qa = QaCapability(
    index_manager, llm, similarity_top_k, max_sub_queries,
    reranker=make_reranker(reranker),
    retriever=make_retriever(retriever),
)
```

eval `sut.py` 已 `**self._flags` 透传，无需改动。

### 7. eval 对接

`eval/harness/compare.py` 的 `VARIANTS` 加：

```python
"全开+hybrid": dict(..., retriever="hybrid"),
"全开+hybrid+rerank": dict(..., retriever="hybrid", reranker="bge-reranker-v2-m3"),
```

复用现成 ablation 框架，量化召回侧（`context_recall`）及综合指标增益。

### 8. 测试

- `build_book_filters`：空 → None；非空 → 含 IN 过滤器。
- BM25 helpers（用假语料，不依赖真 chroma/jieba 真实分词可注入假 tokenizer）：建索引、打分、
  scope 后过滤、取 top_k。
- RRF：纯函数，两列表融合排序去重、截断正确。
- `VectorRetriever.retrieve`：假 index_manager，断言 `as_retriever(top_k, filters)` + 返回。
- `HybridRetriever.retrieve`：假 index_manager（dense 假命中）+ 假 BM25/语料，断言两路都跑、
  scope 过滤生效、RRF 融合、BM25 只构造一次（缓存）。
- `make_retriever`：vector/None→VectorRetriever，hybrid→HybridRetriever，未知→ValueError，
  同名 memoize 复用（monkeypatch 假注册表，不触发真 BM25 构造）。
- `QaCapability`：缺省=VectorRetriever 基线不变；注入策略被调用；与 reranker 组合顺序正确。
- 现有 `tests/test_qa_capability.py` 不受影响（默认 VectorRetriever 走 `as_retriever` 同路径）。

### 9. 依赖

- `requirements` 加 `rank_bm25`、`jieba`（均纯 Python，轻）。

## 非目标 / 后续

- 其它检索策略（HyDE、纯 BM25、多查询）：注册表加一行的后续事。
- dedup / filter 等后续 stage、通用 `RetrievalPipeline` 容器：待变换增多再提。
- BM25 索引随入库动态更新：当前一进程内语料视为不变（eval/服务运行期稳定），先不处理。

## 相关

- 前一增量：`docs/superpowers/specs/2026-06-16-pluggable-reranker-design.md`。
- 接缝代码：`core/workflow/qa_capability.py`、装配 `core/workflow/doc_workflow.py`、
  语料源 `core/rag/data_loader.py`（`RAGIndexManager.chroma_collection`）、
  eval `eval/harness/compare.py` + `sut.py`。
- 记忆：`project_pluggable_retrieval_pipeline`。
