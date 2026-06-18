"""检索工具的注册表 + 工厂（组装核心），供任意 agent 复用。

设计同 core/retrieval/retrieve.py 的注册表风格：工具类自带 name/description +
面向 agent 的 prompt_usage + 执行方法，挂 @register_tool 入表。工具类实现在
core/agent/tools/func/ 下（导入即注册，见 func/__init__）。agent 用 selection
（str | ToolSpec 列表）声明要哪些工具、可逐个覆盖 usage；assemble_tools 由同一份
selection 同时产出 FunctionTool 列表与 system prompt 的工具清单文本，二者不脱节。
共享依赖与 per-run 状态收口到 ToolContext。
"""
from dataclasses import dataclass, field
from typing import Optional

from core.retrieval.rerank import Reranker
from core.retrieval.retrieve import Retriever, make_retriever


@dataclass
class ToolContext:
    """工具共享依赖 + 可重置的 per-run 状态。

    所有工具只接此一个 ctx 构造，故注册表能统一实例化。scope/sources 由 agent 在
    每次 run 前设置/重置：scope 是本轮检索范围（None=全库），sources 收集本轮命中。

    检索策略可插拔（与 core/workflow/qa_capability 同一套）：retriever 默认向量基线，
    reranker 默认无（None）；agent 装配时可注入更强组合（如 hybrid + bge 重排）。有
    reranker 时按 rerank_candidate_k 过召回候选池，再重排截断到 similarity_top_k。
    """
    index_manager: object
    similarity_top_k: int = 5
    scope: Optional[list[str]] = None
    sources: list = field(default_factory=list)
    retriever: "Retriever" = field(default_factory=lambda: make_retriever("vector"))
    reranker: "Reranker | None" = None
    rerank_candidate_k: int = 20


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
