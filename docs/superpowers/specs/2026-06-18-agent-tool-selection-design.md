# 设计：工具说明动态拼接 + per-agent 工具选择/usage 覆盖

**日期**：2026-06-18
**状态**：已批准，待实现

## 背景与目标

`core/agent/qa_agent.py` 的 `QA_AGENT_SYSTEM_PROMPT` 把工具清单（`book_search`/`list_books`
及其用法说明）**写死**在 prompt 文本里（当前 37-39 行）。问题：

- 工具集变了（换/加/减）prompt 不会跟着变，易脱节、误导 agent 调用不存在的工具。
- 不同 agent 想引入不同工具子集时，没有统一机制。

**目标**：

1. 工具用法说明不再写死，从注册的工具**动态拼接**进 system prompt。
2. 每个 agent 可声明自己要哪些工具（注册表子集），并可**逐个覆盖**某工具的用法说明。
3. 工具列表与 prompt 工具清单由**同一份选择**派生，杜绝脱节。

## 非目标

- 不改检索/工具的运行行为（`assemble_tools` 默认 `selection=None` 时拼出的清单须与现 prompt
  逐字一致）。
- 不接线 `BookAgent`（仅让能力可复用）。
- 不新增第三个真实工具（测试里可临时注册一个假工具验证回退逻辑）。

## 现状（前序重构产物）

`core/agent/tools/book_tools.py` 已有：`ToolContext`、`BookSearchTool`/`ListBooksTool`
（各有 `name`/`description` + `to_function_tool()`）、`@register_tool` 装饰器 +
`_TOOL_REGISTRY`、`build_book_tools(ctx, names=None)` 工厂。本设计在其上扩展。

## 组件

### 1. 工具类新增 `prompt_usage`（默认用法说明）

```python
class BookSearchTool:
    name = "book_search"
    description = "书籍知识库检索：按 query 返回相关原文片段，范围由用户选定。"
    prompt_usage = "book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。"
    ...

class ListBooksTool:
    name = "list_books"
    description = "列出当前已入库书籍清单。"
    prompt_usage = "list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。"
    ...
```

`prompt_usage` 文本逐字搬自现 `QA_AGENT_SYSTEM_PROMPT`（去掉行首编号，编号由渲染时生成）。

### 2. `ToolSpec`：agent 的工具选择项

```python
from dataclasses import dataclass

@dataclass
class ToolSpec:
    """一个工具选择项：name 指向注册表里的工具；usage 覆盖其默认 prompt_usage（None=用默认）。"""
    name: str
    usage: Optional[str] = None
```

agent 用 `list[str | ToolSpec]` 声明工具：裸字符串=默认 usage；`ToolSpec(name, usage=...)`=覆盖。

### 3. `assemble_tools`：一次产出「FunctionTool 列表 + prompt 工具清单」

```python
def _normalize(selection: Optional[list]) -> list:
    """selection=None → 注册表全部（登记顺序）；str → ToolSpec；未知名 → ValueError。"""
    if selection is None:
        selection = list(_TOOL_REGISTRY)
    specs = []
    for item in selection:
        spec = ToolSpec(item) if isinstance(item, str) else item
        if spec.name not in _TOOL_REGISTRY:
            raise ValueError(f"未知工具名字：{spec.name!r}，可选：{list(_TOOL_REGISTRY)}")
        specs.append(spec)
    return specs


def _usage_of(spec: ToolSpec) -> str:
    """覆盖优先；否则工具类 prompt_usage；再否则回退 description。"""
    cls = _TOOL_REGISTRY[spec.name]
    return spec.usage or getattr(cls, "prompt_usage", None) or cls.description


def assemble_tools(ctx: ToolContext, selection: Optional[list] = None) -> tuple[list, str]:
    """按 selection 装配工具。返回 (FunctionTool 列表, 编号好的工具清单文本)。

    selection=None → 注册表全部、默认 usage。工具与清单由同一份 specs 派生，必然一致。
    """
    specs = _normalize(selection)
    tools = [_TOOL_REGISTRY[s.name](ctx).to_function_tool() for s in specs]
    prompt = "\n".join(f"{i}. {_usage_of(s)}" for i, s in enumerate(specs, 1))
    return tools, prompt
```

### 4. `build_book_tools` 降为薄包装（保留既有 API）

```python
def build_book_tools(ctx: ToolContext, selection: Optional[list] = None) -> list:
    """仅取工具列表（不需要 prompt 清单的调用方用）。参数由 names 推广为 selection。"""
    return assemble_tools(ctx, selection)[0]
```

既有调用（仅传 `ctx`）与 CLAUDE.md 不破；参数名由 `names` 改为 `selection`，语义向后兼容
（一串字符串仍合法）。

### 5. `QaAgent` 暴露 `tool_selection`，prompt 改模板

`core/agent/qa_agent.py`：

- `QA_AGENT_SYSTEM_PROMPT` 的「工具：1. … 2. …」三行替换为：
  ```
  工具：
  {tools}

  回答：中文，结构清晰，必要时引用书名/章节；先给结论再展开。
  ```
  （模板内无其它花括号，`str.format(tools=...)` 安全。）
- `__init__` 新增 `tool_selection: Optional[list] = None`，存为 `self.tool_selection`。
  这是"不同 agent 引入不同工具/自定义 usage"的入口；默认 `None`=全集默认 usage。
- `_ensure_agent`：
  ```python
  tools, tools_prompt = assemble_tools(self.ctx, self.tool_selection)
  self.agent = FunctionAgent(
      tools=tools,
      llm=self.llm,
      system_prompt=QA_AGENT_SYSTEM_PROMPT.format(tools=tools_prompt),
      early_stopping_method="generate",
  )
  ```
- import 改为 `from core.agent.tools.book_tools import ToolContext, ToolSpec, assemble_tools`
  （`build_book_tools` 不再被 QaAgent 直接用，但模块仍导出）。

## 数据流（_ensure_agent 首次构造）

1. `self.tool_selection`（默认 None）→ `assemble_tools(ctx, selection)`。
2. `_normalize` 把 None/str/ToolSpec 统一成 `list[ToolSpec]`，校验名字。
3. 逐 spec 实例化工具类 → `to_function_tool()`；同序渲染 `{编号}. {usage}`。
4. 返回 `(tools, prompt)`；QaAgent 把 prompt 填进模板，连同 tools 交给 FunctionAgent。

## 错误处理

- 未知工具名 → `ValueError`（与 `make_retriever`/原 `build_book_tools` 一致），构造期暴露。
- 工具类无 `prompt_usage` → 回退 `description`（不崩）。

## 测试

`tests/test_book_tools.py` 新增：

- `assemble_tools(ctx)` 默认：返回 2 个 FunctionTool（`metadata.name` 为
  `book_search`/`list_books`）；prompt 含 "1. " 与 "2. "、含 `book_search(query)` 与
  `list_books()`。
- 子集：`assemble_tools(ctx, ["book_search"])` → 1 个工具，prompt 仅一条且不含 `list_books`。
- 覆盖：`assemble_tools(ctx, [ToolSpec("book_search", usage="自定义X")])` → prompt 含
  "自定义X" 且不含默认 `book_search(query)` 文案。
- 未知名：`assemble_tools(ctx, ["nope"])` 抛 `ValueError`。
- 回退：临时 `@register_tool` 一个只有 `name`/`description`、无 `prompt_usage` 的假工具，
  `assemble_tools(ctx, ["<假名>"])` 的 prompt 用其 `description`。测试末尾从 `_TOOL_REGISTRY`
  清掉该假工具，避免污染其它用例。
- `build_book_tools(ctx)` 仍返回 2 个工具（薄包装回归）。

`tests/test_qa_agent.py` 新增：

- `QaAgent(FakeIndexManager(...), MockLLM(), tool_selection=[ToolSpec("book_search", usage="覆盖语")])`，
  调 `_ensure_agent()`，断言 `agent.system_prompt` 含 "覆盖语" 且不含默认
  `book_search(query)` 文案（已确认 LlamaIndex `FunctionAgent.system_prompt` 可读，回传构造时传入值）。
  另加一条默认用例：`tool_selection=None` 时 `agent.system_prompt` 同时含 `book_search(query)`
  与 `list_books()` 两条默认 usage（验证动态拼接、默认行为不变）。

## 文档

更新 `CLAUDE.md`「工具在组装层创建」段：补一句工具清单经 `assemble_tools` 动态拼进 agent
system prompt，agent 可用 `ToolSpec` 选择工具子集并覆盖 usage。

## 验收

- `pytest tests/test_book_tools.py tests/test_qa_agent.py` 全绿。
- `python scripts/check_layering.py` 通过。
- 默认行为对照：`tool_selection=None` 时 QaAgent 最终 system prompt 的工具清单与改造前逐字一致。
