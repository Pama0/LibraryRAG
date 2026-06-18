# CLAUDE.md

LLM 学习/实验项目，技术书籍知识库助手（上传 PDF + RAG 问答）。底层 LLM 为 DeepSeek（`OpenAILike` 接入，已关 thinking，见 `configs/llm.py`）；评测侧（`eval/`）刻意沿用智谱 GLM，与被测系统解耦。

## Environment

- Python 3.12+，虚拟环境 `.venv`
- 激活：`.venv\Scripts\activate` (PowerShell) / `source .venv/Scripts/activate` (Git Bash)
- API Key 存放在 `.env`：`DEEPSEEK_API_KEY`（主系统）+ `ZHIPU_API_KEY`（评测侧）

## Running

```bash
python main.py                                   # book CLI 对话
python -m uvicorn api.main:app --port 8000       # Web 服务（前端对接）
```

## ⚠️ Gotchas

### 模块导入：必须从项目根目录运行

```python
# ✅ 根目录脚本用绝对导入
from configs.llm import configure_llm
from core.rag.data_loader import load_and_process_data

# ✅ 子模块内用相对导入
from .rag.pdf_parser import BookPDFParser

# ❌ 不要直接运行子目录脚本
# python core/rag/data_loader.py  → ModuleNotFoundError
```

### 分层：core 不依赖 api

依赖方向单向 `api/`(Web) → `core/`(领域) → `configs/`。守卫：`python scripts/check_layering.py`。

### 工具在组装层创建，注入 Agent

检索工具现位于 `core/agent/tools/`：用
`core.agent.tools.build_book_tools(ToolContext(index_manager))` 经注册表 + 工厂组装
成 `FunctionTool` 列表。`QaAgent` 已接入（`self.ctx` 即 `ToolContext`，持
`index_manager`/`similarity_top_k`/`scope`/`sources`）；其它 agent（如
`BookAgent`）可同样注入复用。新增 workflow 工具见 `core/workflow/README.md`。

### 法条遗留

早期法律条文 RAG 代码已冻结于 `legacy/`（不保证可运行，见 `legacy/README.md`）。

## Code Style

- 所有 I/O 操作用 `async/await`
- 函数签名加类型注解
- 中文注释可接受
