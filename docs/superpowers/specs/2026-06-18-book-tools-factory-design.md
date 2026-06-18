# 设计：检索工具解耦为工厂 + 注册表（`core/agent/tools`）

**日期**：2026-06-18
**状态**：已批准，待实现

## 背景与目标

`core/agent/qa_agent.py` 把 `book_search` / `list_books` 两个工具以闭包形式写死在
`_make_tools()` 里，且 `book_search`（`_search`）耦合了 `QaAgent` 的实例状态：

- 共享依赖：`self.index_manager`、`self.similarity_top_k`
- per-run 状态：`self._run_scope`（本轮检索范围）、`self._run_sources`（本轮命中 nodes 收集）

后果：

1. 工具无法被其他 agent 复用。`core/agent/agent.py` 的 `BookAgent` 现在靠外部注入
   `tools`，但代码库里没有任何地方真正构造这些工具（CLAUDE.md 提到的
   `core.tools.book_tools.create_book_search_tool/create_list_books_tool` 文件不存在，文档过时）。
2. 工具与 agent 强耦合，新增工具要改 `QaAgent` 内部。

**目标**：把两个工具抽成独立模块，用工厂 + 注册表模式组装，让任意 agent 复用；
保持运行行为完全不变。

## 非目标（本次明确不做）

- **不修旧瑕疵**：检索片段 `[:500]` 截断、空查询/空库占位文案、`sources` 不去重、
  `list_books` 不带 scope filter —— 全部原样保留。检索管线接入（hybrid/rerank）、
  source 去重、片段带 metadata 等改进**另开 spec**。
- **不接线 `BookAgent`**：本次只让工具**可被复用**，不改装配层（`main.py` / `api/main.py`）
  把工具注入 `BookAgent`。

## 现有约定（对齐风格）

`core/retrieval/retrieve.py` 已有"注册表 + 工厂"先例：`_REGISTRY`（name→类）+
`make_retriever(name)`，策略对象零参构造、依赖在调用时传入。本设计沿用同风格，
依赖统一收口到一个共享 `ToolContext`，使注册表能用统一签名实例化所有工具。

## 架构

新增包 `core/agent/tools/`：

```
core/agent/tools/
  __init__.py        # 导出 ToolContext, build_book_tools, register_tool
  book_tools.py      # ToolContext + 两个工具类 + 注册表 + 工厂
```

依赖方向：`core/agent/tools` → `core/retrieval`（用 `build_book_filters`），
仍是 core 内单向依赖，不触碰 `api/`。`scripts/check_layering.py` 应继续通过。

## 组件

### 1. `ToolContext`（共享依赖 + per-run 状态）

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ToolContext:
    """工具共享依赖 + 可重置的 per-run 状态。

    所有工具只接此一个 ctx 构造，故注册表能统一实例化。scope/sources 由 agent
    在每次 run 前设置/重置。
    """
    index_manager: object
    similarity_top_k: int = 5
    scope: Optional[list[str]] = None              # 本轮检索范围（None=全库）
    sources: list = field(default_factory=list)    # 本轮收集的命中 NodeWithScore
```

### 2. 工具类（各自"定义好方法"）

每个工具是独立类，自带 `name` / `description` 类属性、一个执行方法（`__call__`），
和一个 `to_function_tool()` 包装方法。

```python
@register_tool
class BookSearchTool:
    name = "book_search"
    description = "书籍知识库检索：按 query 返回相关原文片段，范围由用户选定。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    async def __call__(self, query: str) -> str:
        # = 现 QaAgent._search 逻辑原样搬：
        #   - 非 str → str()；strip 后空 → "请提供要检索的问题。"
        #   - index 为 None → "知识库为空，请先上传 PDF。"
        #   - as_retriever(similarity_top_k=ctx.similarity_top_k,
        #                   filters=build_book_filters(ctx.scope))
        #   - 无命中 → "（未检索到相关内容）"
        #   - 命中：ctx.sources.extend(nodes)；返回 "\n---\n".join(片段[:500])
        ...

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )


@register_tool
class ListBooksTool:
    name = "list_books"
    description = "列出当前已入库书籍清单。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def __call__(self) -> str:
        # = 现 QaAgent list_books 逻辑原样搬：
        #   ctx.index_manager.chroma_collection.get(include=["metadatas"])
        #   按 book_title 计数；空 → "知识库当前为空。"
        ...

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )
```

注意：`FunctionTool.from_defaults` 的 `fn` 传 `self.__call__`（已绑定），其 docstring
不再被 LlamaIndex 用作描述（显式传 `name`/`description`），故工具行为与现状一致。

### 3. 注册表 + 工厂

```python
_TOOL_REGISTRY: dict[str, type] = {}   # name → 工具类

def register_tool(cls):
    """装饰器：把工具类按其 name 登记到注册表。新增工具加一行 @register_tool 即可。"""
    _TOOL_REGISTRY[cls.name] = cls
    return cls

def build_book_tools(ctx: ToolContext, names: Optional[list[str]] = None) -> list:
    """工厂：按名从注册表实例化工具并包成 FunctionTool 列表。

    names=None → 注册表中全部工具（顺序为登记顺序：book_search, list_books）。
    未知名字 → ValueError（对齐 make_retriever 行为）。
    """
    names = names or list(_TOOL_REGISTRY)
    tools = []
    for n in names:
        if n not in _TOOL_REGISTRY:
            raise ValueError(f"未知工具名字：{n!r}，可选：{list(_TOOL_REGISTRY)}")
        tools.append(_TOOL_REGISTRY[n](ctx).to_function_tool())
    return tools
```

## `QaAgent` 改造

`core/agent/qa_agent.py`：

- `__init__`：新增 `self.ctx = ToolContext(index_manager, similarity_top_k)`；
  删除 `self._run_scope` / `self._run_sources`（状态收口到 ctx）。
- 删除 `_search` 与 `_make_tools` 方法。
- `_ensure_agent()`：`tools=build_book_tools(self.ctx)`。
- `run()`：开头 `self.ctx.scope = book_titles`、`self.ctx.sources = []`；
  原读 `len(self._run_sources)` / `list(self._run_sources)` 改为 `self.ctx.sources`。
- `index_manager` / `similarity_top_k` 仍存为实例属性或经 ctx 访问（按实现简洁取舍，
  对外契约 `run()` 签名与返回 `(answer, sources)` 不变）。

## 数据流（run 一次）

1. `QaAgent.run(ctx_wf, query, book_titles)` 设 `self.ctx.scope=book_titles`、`self.ctx.sources=[]`。
2. `_ensure_agent()` 用 `build_book_tools(self.ctx)` 拿到 FunctionTool 列表（首次构造 FunctionAgent）。
3. FunctionAgent 多轮调用 `book_search` → `BookSearchTool.__call__` 读 `ctx.scope` 检索、
   `ctx.sources.extend(nodes)`。
4. 桥接：`ToolCall` → `RetrievalStartEvent`；`ToolCallResult` → `RetrievalDoneEvent(count=len(ctx.sources))`。
5. 收尾：`AnswerDeltaEvent`，返回 `(answer, list(ctx.sources))`。

## 错误处理

- 与现状一致：工具内部不抛业务异常，用占位文案兜底；`run()` 的 agent 异常仍由调用方
  `other_branch` 降级处理（不在本次范围）。
- `build_book_tools` 遇未知名字抛 `ValueError`（新行为，仅在显式传 `names` 时触发；
  默认全量不触发）。

## 测试

新增 `tests/test_book_tools.py`：

- `BookSearchTool.__call__`：复用现有 Fake（`FakeIndexManager`/`_Node`），断言拼接片段、
  `ctx.sources` 收集数量；空命中返回占位且不收集；空 query 返回提示；index 为 None 返回空库提示。
- `ListBooksTool.__call__`：有书计数、空库文案。
- `build_book_tools`：默认返回两个工具且 `metadata.name` 为 `book_search`/`list_books`；
  传未知 name 抛 `ValueError`；ctx.scope 透传到 `as_retriever` 的 filters（沿用 `FakeIndex.last_kw`）。

改 `tests/test_qa_agent.py`：

- 移除直接测 `qa._search` 的两个用例（迁移到 `test_book_tools.py`）。
- `run` 桥接用例：`qa._run_sources = ["stale"]` 改为 `qa.ctx.sources = ["stale"]`；
  断言 sources 重置仍成立。

## 文档

更新 `CLAUDE.md`「工具在组装层创建，注入 Agent」段：把过时的
`core.tools.book_tools.create_book_search_tool / create_list_books_tool` 改为
`core.agent.tools.build_book_tools(ToolContext(index_manager))`，并说明工具现位于
`core/agent/tools/`、用注册表 + 工厂组装。

## 验收

- `pytest tests/test_book_tools.py tests/test_qa_agent.py` 全绿。
- `python scripts/check_layering.py` 通过。
- 行为对照：other 分支问答的检索/流式事件/返回 sources 与改造前一致。
