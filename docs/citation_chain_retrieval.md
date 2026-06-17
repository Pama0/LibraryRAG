# 法条引用链检索

## 问题背景

法律文本中大量存在跨条文引用，例如「依照本法第 X 条」「按照本条例第 Y 条」。当前 RAG 系统使用 `ArticleSplitter` 将法规按条文切分为独立 chunk 后存入向量库，**引用关系完全丢失**。

**典型问题**：用户问到「行政处罚法第 62 条关于听证的规定」时，第 62 条原文写的是「依照法律规定向当事人告知……」，其中「法律规定」指向本法第 44、45 条的具体内容。Simple RAG 只能返回「依照法律规定」这个空壳，无法给出第 44、45 条的实际内容。

### 数据规模

| 指标 | 数值 |
|------|------|
| 法规文档总数 | 539 |
| 包含引用的法规 | 223（41%） |
| 本法内引用（A 类） | 635 处 |
| 外部法规引用（B 类） | 197 处 |

引用关系在法律文本中极其普遍，这是一个框架调参无法解决的问题，需要自己设计引用识别 + 图结构 + 检索策略。

---

## 引用模式分类

| 类型 | 模式 | 数量 | 处理方式 |
|------|------|------|----------|
| A-显式 | `依照本法第X条` / `依据本办法第Y条` | 主要部分 | 正则精确提取 |
| A-隐式 | `依照第X条`（省略自指词「本法」） | 补充 | 正则提取 + 排除已匹配的显式和外部引用 |
| B-外部 | `依照《XXX》第X条` | 197 处 | 正则提取法规名和条文号 |
| C-隐式 | `按照国家有关规定` / `依照有关法律` | 大量 | 无法精确解析，暂不处理 |

---

## 系统架构

```
                     用户查询
                        │
                        ▼
              ┌─────────────────────┐
              │  CitationRAGWorkflow │
              └─────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ① AutoRetriever  ② CitationGraph  ③ 精确获取
    向量+元数据检索   BFS引用链扩展   MetadataFilter
         │              │              │
         │         被引用条文列表       │
         │              │              │
         └──────────────┼──────────────┘
                        ▼
              ④ 合并去重 + 标记来源
                        │
                        ▼
              ⑤ Response Synthesis
              [直接检索] / [引用扩展]
                        │
                        ▼
                     最终回答
```

### 数据流详解

**① AutoRetriever 检索**：LLM 将用户查询转为结构化检索请求（query + metadata filters），从 ChromaDB 检索 top-k 节点。

**② 引用链扩展**：对每个检索到的节点，查 CitationGraph 获取其引用的条文列表，BFS 展开到指定深度。

**③ 精确获取被引用条文**：通过 `file_name + article_no_int` 元数据精确匹配，从向量库获取被引用条文的内容。**不走向量相似度检索**，而是元数据精确查找，确保获取的是确切的条文。

**④ 合并去重**：以 `(file_name, article_no_int)` 为唯一键去重。原始节点标记 `is_citation_expansion=False`，扩展节点标记 `is_citation_expansion=True` + `citation_source`。

**⑤ Response Synthesis**：节点文本前加 `[直接检索]` / `[引用扩展·第X条引用]` 标记，使用自定义 prompt 让 LLM 区分来源，在回答中说明引用关系。

---

## 模块说明

### 1. 引用抽取器 `core/rag/citation_extractor.py`

从条文文本中抽取引用关系，返回 `Citation` 对象列表。

**核心数据结构**：

```python
@dataclass
class Citation:
    source_article: str       # 引用方条文号（中文），如 "五十八"
    source_article_int: int   # 引用方条文号（数字），如 58
    target_article: str       # 被引用条文号（中文），如 "五十三"
    target_article_int: int   # 被引用条文号（数字），如 53
    citation_type: str        # "internal" / "external"
    citation_verb: str        # "依照" / "参照" / "按照" 等
    target_law: str | None    # 外部法规名（仅 B 类）
    context: str              # 引用所在原文片段
```

**三条正则模式**：

| 模式 | 匹配内容 | 示例 |
|------|----------|------|
| `PATTERN_INTERNAL_EXPLICIT` | 动词 + 自指词 + 条文号 | `依照本法第五十三条的规定` |
| `PATTERN_INTERNAL_IMPLICIT` | 动词 + 条文号（排除已匹配和外部引用） | `依照第五十三条` |
| `PATTERN_EXTERNAL` | 动词 + 《法规名》+ 条文号 | `依照《商标法》第三十条` |

**关键处理逻辑**：

- **预处理**：去除 `【法规名】` 章节上下文行和条文号标头，防止跨行误匹配（如章节标题中的「适用」与条文号「第二十二条」拼成伪引用）
- **去重**：隐式模式跳过已被显式模式匹配的 span
- **外部引用排除**：隐式模式跳过前面 15 字符内有 `《` 的匹配
- **自引用过滤**：`source_article_int == target_article_int` 的结果必定是误提取，直接过滤
- **复合条文拆分**：`第五十三条、第五十六条` 拆为两条独立引用
- 复用 `parser.py` 中的 `cn_num_to_int()` 做中文数字转换

### 2. 引用图 `core/rag/citation_graph.py`

构建并持久化条文间引用关系的有向图，使用邻接表存储。

**数据结构**：`{file_name: {article_int: [Citation, ...]}}`

**核心方法**：

| 方法 | 说明 |
|------|------|
| `build_from_nodes(nodes)` | 从 TextNode 列表全量构建 |
| `update_file(file_name, nodes)` | 增量更新单个文件的引用关系 |
| `remove_file(file_name)` | 删除文件的引用关系 |
| `get_citations(file_name, article_int)` | 获取某条文的直接引用列表 |
| `expand(file_name, article_ints, depth=1)` | BFS 展开引用链，depth=1 取直接引用，depth=2 取引用的引用 |
| `get_reverse_citations(file_name, article_int)` | 反向查询：谁引用了指定条文 |
| `save(path)` / `load(path)` | JSON 持久化 |

**BFS 展开逻辑**（`expand` 方法）：

```
队列初始化: [(file_name, start_article_int, depth=0)]
已访问集合: {(file_name, start_article_int)}

while 队列非空:
    弹出 (cur_file, cur_art, cur_depth)
    if cur_depth >= depth: 跳过
    for cite in get_citations(cur_file, cur_art):
        key = (file_name, cite.target_article_int)
        if key not in 已访问:
            加入结果列表
            加入队列: (file_name, cite.target_article_int, cur_depth + 1)
```

**注意**：BFS 展开时 `file_name` 保持不变（本法引用在同一个文件内展开），外部引用暂不展开。

**构建时机**：`RAGIndexManager.add_documents()` 完成索引后自动构建，持久化到 `./citation_graph.json`。增量更新时根据变更情况决定全量或增量重建。

### 3. 引用链检索工作流 `core/workflow/citation_rag.py`

`CitationRAGWorkflow` 继承 LlamaIndex 的 `Workflow`，单步执行。

**初始化参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `index` | - | VectorStoreIndex 实例 |
| `llm` | - | LLM 实例 |
| `citation_graph` | - | CitationGraph 实例 |
| `expand_depth` | 1 | 引用链扩展深度 |
| `max_expansions` | 10 | 最大扩展数量（防止引用链爆炸） |

**回退策略**（AutoRetriever 空结果时）：

1. **file_name 修正**：LLM 生成的 file_name 常与实际不匹配（如 `专利法.docx` vs `中华人民共和国专利法.docx`），自动尝试补 `.docx` 后缀和 `中华人民共和国` 前缀
2. **去掉 file_name 重试**：修正失败后只保留 article 过滤条件重试
3. **纯向量检索**：以上都失败，退回 `similarity_top_k=5` 的纯向量检索

**引用扩展的节点获取**：通过 `MetadataFilter(file_name) AND MetadataFilter(article_no_int)` 精确匹配，不走向量相似度，确保获取的是确切的被引用条文。

**响应标记**：扩展节点的文本前加 `[引用扩展·第X条引用]`，原始节点加 `[直接检索]`，配合自定义 prompt 让 LLM 在回答中区分来源。

---

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `core/rag/citation_extractor.py` | 新建 | 引用模式识别与抽取 |
| `core/rag/citation_graph.py` | 新建 | 引用有向图构建、查询、持久化 |
| `core/workflow/citation_rag.py` | 新建 | 引用链检索工作流 |
| `core/rag/data_loader.py` | 修改 | 集成引用图构建（`citation_graph` 属性、`_update_citation_graph`、`_fetch_all_nodes`） |
| `core/tools/tools.py` | 修改 | 新增 `create_citation_rag_tool()` |
| `app.py` | 修改 | 组装引用图和新工具 |
| `docs/citation_map.md` | 新建 | 引用关系对照文档（225 个法规的引用详情表） |
| `citation_graph.json` | 生成 | 引用图持久化文件 |

**未修改**：`parser.py`、`auto_retriever_prompt.py`、`vector_store_info.py`、`simple_rag.py`、`multi_strategy_rag.py`

---

## 验证结果

### 专利法第 58 条

| | Simple RAG | Citation RAG |
|---|---|---|
| 检索节点 | 5 条 | 5 + 2 扩展 = 7 条 |
| 回答 | "无法找到第五十八条" | 完整解释第 58 条，并展开第 53 条（反垄断例外）和第 55 条（公共健康例外） |

### 行政处罚法第 62 条

| | Simple RAG | Citation RAG |
|---|---|---|
| 检索节点 | 5 条 | 5 + 2 扩展 = 7 条 |
| 回答 | "依照法律规定告知"（模糊） | "依据第四十四条告知义务、依据第四十五条申辩权"（精确到被引用条文） |

### 刑法第 37 条之一

| | Simple RAG | Citation RAG |
|---|---|---|
| 检索节点 | 5 条 | 5 + 1 扩展 = 6 条 |
| 回答 | "依照刑法第三百一十三条定罪处罚"（只提条文号） | 解释第 313 条的具体罪名和量刑 |

---

## 引用图统计

当前 `citation_graph.json` 中的数据：

```
法规数: 223
总引用数: 832
  本法内引用 (internal): 635
  外部法规引用 (external): 197
```

引用最丰富的法规：中华人民共和国刑法（66 处引用）。

完整的法规引用对照表见 `docs/citation_map.md`。

---

## 扩展方向

1. **B 类外部引用展开**：当前 B 类引用只记录了目标法规名和条文号，未跨文档获取具体内容。可通过模糊匹配法规名 + article_no_int 精确查找实现
2. **C 类隐式引用**：使用 LLM 辅助识别「按照国家有关规定」等模糊引用的目标
3. **引用链深度调参**：当前默认 depth=1，对某些场景（如刑法大量交叉引用）depth=2 可能更有价值
4. **反向引用推荐**：`get_reverse_citations()` 已实现，可用于「哪些条文引用了这一条」的场景
