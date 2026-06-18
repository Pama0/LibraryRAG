"""检索工具实现：导入即注册（各工具类上的 @register_tool 写入 _TOOL_REGISTRY）。

工具类挂了 @register_tool，但装饰器只在模块被导入时才执行。这里集中导入，使
`import core.agent.tools` 一并触发注册，注册表才非空。新增工具：在此加一行导入。
"""
from core.agent.tools.func.book_search_tool import BookSearchTool
from core.agent.tools.func.list_book_tool import ListBooksTool

__all__ = ["BookSearchTool", "ListBooksTool"]
