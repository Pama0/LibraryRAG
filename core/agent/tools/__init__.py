"""agent 可复用工具包：检索工具的工厂 + 注册表。"""
from core.agent.tools.book_tools import (
    ToolContext,
    ToolSpec,
    assemble_tools,
    build_book_tools,
    register_tool,
)
# 导入 func 子包触发各工具的 @register_tool 落表（须在 register_tool 定义之后）。
from core.agent.tools.func import BookSearchTool, ListBooksTool

__all__ = [
    "ToolContext",
    "ToolSpec",
    "BookSearchTool",
    "ListBooksTool",
    "assemble_tools",
    "build_book_tools",
    "register_tool",
]
