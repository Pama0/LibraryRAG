"""书籍知识库检索工具集：工厂 + 注册表组装，供任意 agent 复用。

设计同 core/retrieval/retrieve.py 的注册表风格：每个工具一个类（自带 name/
description + 面向 agent 的 prompt_usage + 执行方法），@register_tool 入表。
agent 用 selection（str | ToolSpec 列表）声明要哪些工具、可逐个覆盖 usage；
assemble_tools 由同一份 selection 同时产出 FunctionTool 列表与 system prompt 的
工具清单文本，二者不脱节。共享依赖与 per-run 状态收口到 ToolContext。
"""
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.tools import FunctionTool

from core.retrieval.retrieve import build_book_filters


@dataclass
class ToolContext:
    """工具共享依赖 + 可重置的 per-run 状态。

    所有工具只接此一个 ctx 构造，故注册表能统一实例化。scope/sources 由 agent 在
    每次 run 前设置/重置：scope 是本轮检索范围（None=全库），sources 收集本轮命中。
    """
    index_manager: object
    similarity_top_k: int = 5
    scope: Optional[list[str]] = None
    sources: list = field(default_factory=list)


@dataclass
class ToolSpec:
    """一个工具选择项：name 指向注册表里的工具；usage 覆盖其默认 prompt_usage（None=用默认）。"""
    name: str
    usage: Optional[str] = None


_TOOL_REGISTRY: dict[str, type] = {}  # name → 工具类


def register_tool(cls):
    """装饰器：按 cls.name 登记工具类。新增工具加一行 @register_tool 即可。"""
    _TOOL_REGISTRY[cls.name] = cls
    return cls


@register_tool
class BookSearchTool:
    """书籍知识库检索：按 query 取 top-k 原文片段并把命中 nodes 收进 ctx.sources。"""

    name = "book_search"
    description = "书籍知识库检索：按 query 返回相关原文片段，范围由用户选定。"
    prompt_usage = "book_search(query) — 在书籍知识库检索，返回相关原文片段。检索范围已由用户选定，你无需也无法指定书名，只管传好 query。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    async def __call__(self, query: str) -> str:
        if not isinstance(query, str):
            query = str(query)
        query = query.strip()
        if not query:
            return "请提供要检索的问题。"
        index = self.ctx.index_manager.get_index()
        if index is None:
            return "知识库为空，请先上传 PDF。"
        retriever = index.as_retriever(
            similarity_top_k=self.ctx.similarity_top_k,
            filters=build_book_filters(self.ctx.scope),
        )
        nodes = await retriever.aretrieve(query)
        if not nodes:
            return "（未检索到相关内容）"
        self.ctx.sources.extend(nodes)
        return "\n---\n".join(
            (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:500]
            for n in nodes
        )

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )


@register_tool
class ListBooksTool:
    """列出当前已入库书籍清单（按 book_title 计数）。"""

    name = "list_books"
    description = "列出当前已入库书籍清单。"
    prompt_usage = "list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def __call__(self) -> str:
        data = self.ctx.index_manager.chroma_collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas", []) or []:
            title = (meta or {}).get("book_title")
            if not title:
                continue
            counts[title] = counts.get(title, 0) + 1
        if not counts:
            return "知识库当前为空。"
        return "已入库书籍：\n" + "\n".join(
            f"- 《{t}》（{c} 块）" for t, c in sorted(counts.items())
        )

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )


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
    """仅取工具列表（不需要 prompt 清单的调用方用）。selection 同 assemble_tools。"""
    return assemble_tools(ctx, selection)[0]
