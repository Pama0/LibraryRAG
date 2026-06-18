# 工具说明动态拼接 + per-agent 工具选择/usage 覆盖 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 agent 的工具用法说明不再写死在 prompt，而是按 agent 实际选择的工具（可逐个覆盖 usage）动态拼接进 system prompt。

**Architecture:** 在 `core/agent/tools/book_tools.py` 给工具类加默认 `prompt_usage`，新增 `ToolSpec`（选择项，可覆盖 usage）与 `assemble_tools(ctx, selection)`（一次产出 FunctionTool 列表 + 编号好的工具清单文本，二者同源不脱节）；`build_book_tools` 降为薄包装。`QaAgent` 暴露 `tool_selection` 参数，`QA_AGENT_SYSTEM_PROMPT` 改成带 `{tools}` 占位的模板，`_ensure_agent` 用 `assemble_tools` 填充。

**Tech Stack:** Python 3.12, LlamaIndex（FunctionTool/FunctionAgent，`FunctionAgent.system_prompt` 可读）, pytest（asyncio_mode=auto）。

## Global Constraints

- 默认行为不变：`assemble_tools(ctx, selection=None)` 拼出的工具清单须与改造前 `QA_AGENT_SYSTEM_PROMPT` 的「工具：」三行逐字一致（去行首编号，编号由渲染生成）。
- `prompt_usage` 文本逐字搬自现 prompt：
  - book_search：`book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。`
  - list_books：`list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。`
- usage 取值优先级：`ToolSpec.usage` 覆盖 > 工具类 `prompt_usage` > 回退 `description`。
- 未知工具名 → `ValueError`（消息形如 `未知工具名字：{name!r}，可选：{list(_TOOL_REGISTRY)}`）。
- `build_book_tools(ctx)` 既有调用（仅传 ctx）与 CLAUDE.md 引用不破。
- 分层：`core/agent/tools` 只依赖 core 内模块，禁止依赖 `api/`；守卫 `python scripts/check_layering.py` 通过。
- 不接线 BookAgent；不新增第三个真实工具（回退测试用临时假工具，测试末尾清出注册表）。
- 所有 I/O async/await；类型注解；中文注释可接受。从项目根目录运行。
- `QaAgent.run()` 对外契约不变：`run(ctx, query, book_titles) -> (answer, sources)`。

---

### Task 1: 动态工具装配（book_tools `prompt_usage`/`ToolSpec`/`assemble_tools` + QaAgent 接入）

**Files:**
- Modify: `core/agent/tools/book_tools.py`
- Modify: `core/agent/qa_agent.py`
- Modify: `tests/test_book_tools.py`
- Modify: `tests/test_qa_agent.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Produces:
  - `ToolSpec(name: str, usage: Optional[str] = None)` — dataclass。
  - `assemble_tools(ctx: ToolContext, selection: Optional[list] = None) -> tuple[list, str]` —
    返回 `(FunctionTool 列表, 编号工具清单文本)`；`selection` 元素可为 `str` 或 `ToolSpec`，
    `None` → 注册表全部默认 usage。
  - `build_book_tools(ctx, selection=None) -> list` — 薄包装，= `assemble_tools(...)[0]`。
  - `QaAgent.__init__` 新增 `tool_selection: Optional[list] = None`，存 `self.tool_selection`。

- [ ] **Step 1: 在 `tests/test_book_tools.py` 末尾追加 assemble_tools 的失败测试**

在文件末尾追加（沿用文件已有的 `_ctx` / `FakeIndexManager` / `_Node` 等辅助）：

```python
from core.agent.tools.book_tools import (  # noqa: E402  追加导入
    ToolSpec,
    assemble_tools,
    register_tool,
    _TOOL_REGISTRY,
)


def test_assemble_tools_default_returns_both_and_numbered_prompt():
    tools, prompt = assemble_tools(_ctx())
    assert sorted(t.metadata.name for t in tools) == ["book_search", "list_books"]
    assert "1. " in prompt and "2. " in prompt
    assert "book_search(query)" in prompt
    assert "list_books()" in prompt


def test_assemble_tools_subset_only_selected():
    tools, prompt = assemble_tools(_ctx(), ["book_search"])
    assert [t.metadata.name for t in tools] == ["book_search"]
    assert "book_search(query)" in prompt
    assert "list_books" not in prompt


def test_assemble_tools_usage_override_replaces_default():
    _, prompt = assemble_tools(_ctx(), [ToolSpec("book_search", usage="自定义X")])
    assert "自定义X" in prompt
    assert "book_search(query)" not in prompt


def test_assemble_tools_unknown_name_raises():
    import pytest
    with pytest.raises(ValueError):
        assemble_tools(_ctx(), ["nope"])


def test_assemble_tools_falls_back_to_description_when_no_prompt_usage():
    @register_tool
    class _TmpTool:
        name = "_tmp_tool"
        description = "临时工具描述"

        def __init__(self, ctx):
            self.ctx = ctx

        def __call__(self) -> str:
            return ""

        def to_function_tool(self):
            from llama_index.core.tools import FunctionTool
            return FunctionTool.from_defaults(
                fn=self.__call__, name=self.name, description=self.description
            )

    try:
        _, prompt = assemble_tools(_ctx(), ["_tmp_tool"])
        assert "临时工具描述" in prompt
    finally:
        _TOOL_REGISTRY.pop("_tmp_tool", None)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_book_tools.py -q`
Expected: FAIL（`ImportError: cannot import name 'ToolSpec'` / `assemble_tools`）。

- [ ] **Step 3: 改 `core/agent/tools/book_tools.py`**

顶部 import 增补 `dataclass` 已在用；确认有 `from dataclasses import dataclass, field`（现有）。
在 `ToolContext` 定义之后、`BookSearchTool` 之前，加 `ToolSpec`：

```python
@dataclass
class ToolSpec:
    """一个工具选择项：name 指向注册表里的工具；usage 覆盖其默认 prompt_usage（None=用默认）。"""
    name: str
    usage: Optional[str] = None
```

给 `BookSearchTool` 加类属性（紧随其 `description` 之后）：

```python
    prompt_usage = "book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。"
```

给 `ListBooksTool` 加类属性（紧随其 `description` 之后）：

```python
    prompt_usage = "list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。"
```

把现有 `build_book_tools` 函数整体替换为下面三个函数（新增 `_normalize`/`_usage_of`/`assemble_tools`，
`build_book_tools` 降为薄包装）：

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

    selection 元素可为 str（用默认 usage）或 ToolSpec（可覆盖 usage）；None → 注册表全部、
    默认 usage。工具与清单由同一份 specs 派生，必然一致。
    """
    specs = _normalize(selection)
    tools = [_TOOL_REGISTRY[s.name](ctx).to_function_tool() for s in specs]
    prompt = "\n".join(f"{i}. {_usage_of(s)}" for i, s in enumerate(specs, 1))
    return tools, prompt


def build_book_tools(ctx: ToolContext, selection: Optional[list] = None) -> list:
    """仅取工具列表（不需要 prompt 清单的调用方用）。"""
    return assemble_tools(ctx, selection)[0]
```

更新 `core/agent/tools/__init__.py` 的导出，新增 `ToolSpec` 与 `assemble_tools`：

```python
from core.agent.tools.book_tools import (
    BookSearchTool,
    ListBooksTool,
    ToolContext,
    ToolSpec,
    assemble_tools,
    build_book_tools,
    register_tool,
)

__all__ = [
    "ToolContext",
    "ToolSpec",
    "BookSearchTool",
    "ListBooksTool",
    "assemble_tools",
    "build_book_tools",
    "register_tool",
]
```

- [ ] **Step 4: 跑 book_tools 测试确认通过**

Run: `python -m pytest tests/test_book_tools.py -q`
Expected: 全 passed（原有 + 新增 5 条）。

- [ ] **Step 5: 在 `tests/test_qa_agent.py` 追加 QaAgent 动态 prompt 失败测试**

文件顶部已 `from core.agent.qa_agent import QaAgent`。追加导入与两条测试：

```python
from core.agent.tools.book_tools import ToolSpec  # noqa: E402  追加


def test_qa_agent_default_prompt_lists_both_tools():
    qa = QaAgent(FakeIndexManager(nodes=[]), MockLLM())
    agent = qa._ensure_agent()
    assert "book_search(query)" in agent.system_prompt
    assert "list_books()" in agent.system_prompt


def test_qa_agent_tool_selection_overrides_usage_in_prompt():
    qa = QaAgent(
        FakeIndexManager(nodes=[]),
        MockLLM(),
        tool_selection=[ToolSpec("book_search", usage="覆盖语")],
    )
    agent = qa._ensure_agent()
    assert "覆盖语" in agent.system_prompt
    assert "book_search(query)" not in agent.system_prompt
```

- [ ] **Step 6: 跑确认失败**

Run: `python -m pytest tests/test_qa_agent.py -q`
Expected: FAIL（`TypeError: __init__() got an unexpected keyword argument 'tool_selection'` 或 system_prompt 不含 `book_search(query)`，因当前 prompt 写死且无 `{tools}` 渲染）。

- [ ] **Step 7: 改 `core/agent/qa_agent.py`**

import 改为：
```python
from core.agent.tools.book_tools import ToolContext, assemble_tools
```

`QA_AGENT_SYSTEM_PROMPT` 中现有的工具段（"工具：" 起的三行）替换为模板占位：
```
工具：
{tools}

回答：中文，结构清晰，必要时引用书名/章节；先给结论再展开。"""
```
即把
```
工具：
1. book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。
2. list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。

回答：中文，结构清晰，必要时引用书名/章节；先给结论再展开。"""
```
替换为上面的 `{tools}` 版本（保留前面铁律段不变）。

`__init__` 增参并保存（其余不变）：
```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_iterations: int = 6,
        tool_selection: Optional[list] = None,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.tool_selection = tool_selection
        self.ctx = ToolContext(
            index_manager=index_manager, similarity_top_k=similarity_top_k
        )
        self.agent = None
```
（保留原 `__init__` 里关于懒构造的注释。）

`_ensure_agent` 改为用 `assemble_tools`：
```python
    def _ensure_agent(self) -> FunctionAgent:
        if self.agent is None:
            tools, tools_prompt = assemble_tools(self.ctx, self.tool_selection)
            self.agent = FunctionAgent(
                tools=tools,
                llm=self.llm,
                system_prompt=QA_AGENT_SYSTEM_PROMPT.format(tools=tools_prompt),
                early_stopping_method="generate",
            )
        return self.agent
```

- [ ] **Step 8: 跑两测试文件确认通过**

Run: `python -m pytest tests/test_qa_agent.py tests/test_book_tools.py -q`
Expected: 全 passed。

- [ ] **Step 9: 更新 `CLAUDE.md`「工具在组装层创建，注入 Agent」段**

在该段已有内容后补一句：工具用法清单经 `core.agent.tools.assemble_tools(ctx, selection)`
动态拼进 agent 的 system prompt；agent 可用 `ToolSpec(name, usage=...)` 选择工具子集并覆盖单个工具的
usage 说明（`QaAgent` 的 `tool_selection` 参数即此入口）。

- [ ] **Step 10: 全量回归 + 分层守卫**

Run: `python -m pytest tests/ -q && python scripts/check_layering.py`
Expected: 全 passed；分层检查通过。

- [ ] **Step 11: 提交**

```bash
git add core/agent/tools/book_tools.py core/agent/tools/__init__.py core/agent/qa_agent.py tests/test_book_tools.py tests/test_qa_agent.py CLAUDE.md
git commit -m "feat(tools): 工具说明动态拼接 + ToolSpec 选择/usage 覆盖

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage**：prompt_usage（Step 3）；ToolSpec（Step 3）；assemble_tools 同源产出（Step 3）；
  build_book_tools 薄包装（Step 3）；QaAgent tool_selection + 模板 prompt + _ensure_agent（Step 7）；
  __init__.py 导出（Step 3）；测试 5+2 条（Step 1、5）；CLAUDE.md（Step 9）；分层（Step 10）；
  默认行为不变由 Global Constraints 逐字约束 + 默认用例（test_qa_agent_default_prompt_lists_both_tools）保证。全覆盖。
- **Placeholder scan**：无 TODO/TBD，所有代码步给出完整代码。
- **Type consistency**：`ToolSpec(name, usage=None)`、`assemble_tools(ctx, selection=None) -> tuple[list, str]`、
  `build_book_tools(ctx, selection=None)`、`QaAgent(..., tool_selection=None)`、`agent.system_prompt`
  跨步骤一致；`_usage_of` 优先级与 spec 一致；`_TOOL_REGISTRY` 沿用前序模块既有名。
