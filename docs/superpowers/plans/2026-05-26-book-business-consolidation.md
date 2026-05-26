# book 业务整合进 core 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把散落在 `api/` 的 book 领域逻辑整合进 `core/`，消除 `core→api` 反向依赖，统一主 agent，法条遗留代码冻结进 `legacy/`。

**Architecture:** 单向分层 `api/`(Web 适配) → `core/`(领域) → `configs/`(基础设施)。分 6 阶段顺序迁移，每阶段末用模块导入冒烟 + 分层守卫脚本验证可加载，逐阶段提交。

**Tech Stack:** Python 3.12、LlamaIndex（FunctionAgent / Workflow）、ChromaDB、SQLAlchemy 2.0 async、FastAPI。无 pytest —— 验证用 `python -c` 导入冒烟与独立守卫脚本。

参考设计稿：`docs/superpowers/specs/2026-05-26-book-business-consolidation-design.md`

---

## 通用约定

- **分支**：已在 `refactor/book-consolidation`。所有提交落此分支。
- **冒烟命令**：`python -c "import <module>"` —— 只触发模块导入（不构造 app、不加载模型、不需 API Key），用于验证 import 接线。**不要**用 `import api.main`（它在导入时 `create_app()` 会初始化 LLM/embedding，重且需 .env）。
- **app 启动验证**（人工/阶段性）：`python -m uvicorn api.main:app --port 8000` 能起即通过；CLI 验证 `python main.py`。
- **legacy/ 定位**：冻结归档，**不保证可运行**。用 `git mv` 保留历史；对 legacy 内部互相引用做尽力 rewire，但 `add_documents`（法条增量索引）等被本次从 `data_loader` 删除的逻辑仅存于 git 历史，将来重启需从历史恢复。

---

## 文件结构总览（目标态）

```
core/
  agent/
    agent.py            重建：BookAgent(FunctionAgent+会话锁+memory+book SYSTEM_PROMPT+CLI chat)
    source_context.py   迁入：begin_collection/add_sources/get_sources/node_to_source_ref + SourceRef
  tools/book_tools.py   改 import → core.agent.source_context
  rag/
    data_loader.py      瘦身：RAGIndexManager 仅 book
    pdf_parser.py       不变
  persistence/          新建
    __init__.py
    db.py               迁入（修 PROJECT_ROOT 深度）
    repositories.py     迁入（改 import 来源）
  workflow/
    __init__.py         保留（待命）
    README.md           新增：workflow→tool 接线约定
api/
  main.py               改：组装 core，构造 BookAgent
  schemas.py            改：SourceRef 从 core 导入并 re-export
  routers/{chat,sessions,documents}.py  改 import 指向 core
  （agent_service.py / db.py / repositories.py / source_context.py 迁出或删除）
main.py（根）            改：book CLI 启动器
scripts/check_layering.py  新增：分层守卫
legacy/                 新建：app.py、init_index.py、run_eval.py、test_single_file_rag.py、
                        tools.py、workflow/*(法条)、rag/*(法条)、evals/*、citation_graph.json
```

---

## Task 0: 分层守卫脚本

**Files:**
- Create: `scripts/check_layering.py`

- [ ] **Step 1: 写守卫脚本**

```python
# scripts/check_layering.py
"""分层守卫：core/ 不得 import api/。CI/本地手动运行。"""
import pathlib
import re
import sys

CORE = pathlib.Path(__file__).resolve().parent.parent / "core"
PATTERN = re.compile(r"^\s*(?:from|import)\s+api(?:\.|\s|$)", re.MULTILINE)


def main() -> int:
    offenders = []
    for py in CORE.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if PATTERN.search(text):
            offenders.append(str(py))
    if offenders:
        print("分层违规：以下 core 文件 import 了 api：")
        for f in offenders:
            print("  -", f)
        return 1
    print("分层守卫通过：core/ 未 import api/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行，确认当前是违规状态（红）**

Run: `python scripts/check_layering.py`
Expected: 退出码 1，列出 `core/tools/book_tools.py`（这是病灶，Task 2 修复后转绿）

- [ ] **Step 3: 提交守卫脚本**

```bash
git add scripts/check_layering.py
git commit -m "chore: add core->api layering guard script"
```

---

## Task 1: data_loader 瘦身为 book-only

**Files:**
- Modify: `core/rag/data_loader.py`

- [ ] **Step 1: 替换 import 段（删除法条相关）**

把文件顶部 import（第 1-11 行）替换为：

```python
import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.rag.pdf_parser import BookPDFParser
```

（删除 `SimpleDirectoryReader`、`ArticleSplitter`、`CitationGraph` 三个 import）

- [ ] **Step 2: 替换 `__init__`（去 splitter/citation_graph，默认集合改 book_knowledge）**

把 `__init__` 方法整体替换为：

```python
    def __init__(
            self,
            persist_dir: str = "./chroma_db",
            collection_name: str = "book_knowledge",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.db = chromadb.PersistentClient(path=persist_dir)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        if self.chroma_collection.count() > 0:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
            )
        else:
            self.index = None
```

- [ ] **Step 3: 删除法条专用方法**

删除以下方法的完整定义：`add_documents`、`_update_citation_graph`、`_fetch_all_nodes`。
保留：`_get_indexed_file_info`、`_delete_nodes_by_file`、`add_book`、`add_book_quick`、`get_query_engine`、`get_index`。

- [ ] **Step 4: 冒烟 —— 导入 + 构造（book 集合）**

Run: `python -c "from core.rag.data_loader import RAGIndexManager; m=RAGIndexManager(collection_name='book_knowledge'); print('index:', m.index is not None, 'count:', m.chroma_collection.count())"`
Expected: 正常输出（如 `index: True count: 674`），无 ImportError、无 `ArticleSplitter`/`CitationGraph` 相关报错

- [ ] **Step 5: 提交**

```bash
git add core/rag/data_loader.py
git commit -m "refactor(rag): slim RAGIndexManager to book-only, drop legal paths"
```

---

## Task 2: source_context + SourceRef 迁入 core，去耦 book_tools

**Files:**
- Create: `core/agent/source_context.py`
- Modify: `core/tools/book_tools.py:17`
- Modify: `api/schemas.py`（SourceRef 改为从 core 导入）
- Modify: `api/source_context.py` → 删除（迁移后）

- [ ] **Step 1: 创建 `core/agent/source_context.py`（含 SourceRef）**

```python
"""请求级 source 收集容器（领域层）

工具调用时把检索到的 source_nodes 通过 contextvar 写入；
Web handler 在调用 Agent 前 begin_collection()，调用后 get_sources()。
contextvar 默认 task-local，子任务自动继承。

SourceRef 作为领域值对象定义在此，api 层从这里导入用于响应 DTO。
"""
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel


class SourceRef(BaseModel):
    book_title: str
    chapter: str
    page: int
    excerpt: str  # 引用片段


_current_sources: ContextVar[Optional[list]] = ContextVar(
    "current_sources", default=None
)


def begin_collection() -> None:
    """请求开始时调用，重置收集器"""
    _current_sources.set([])


def add_sources(refs: list) -> None:
    """工具内调用，追加 SourceRef 列表"""
    bucket = _current_sources.get()
    if bucket is None:
        return
    bucket.extend(refs)


def get_sources() -> list:
    """请求结束时取出收集到的 sources（去重保序）"""
    bucket = _current_sources.get()
    if not bucket:
        return []
    seen = set()
    unique = []
    for s in bucket:
        key = (s.book_title, s.chapter, s.page, s.excerpt[:50])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


def node_to_source_ref(node) -> SourceRef:
    """LlamaIndex node -> SourceRef"""
    meta = node.metadata or {}
    return SourceRef(
        book_title=meta.get("book_title", "未知"),
        chapter=meta.get("chapter", ""),
        page=meta.get("page", meta.get("page_start", 0)) or 0,
        excerpt=(node.get_content() if hasattr(node, "get_content") else node.text)[:300],
    )
```

- [ ] **Step 2: 改 `core/tools/book_tools.py` 的 import（第 17 行）**

把：
```python
from api.source_context import add_sources, node_to_source_ref
```
改为：
```python
from core.agent.source_context import add_sources, node_to_source_ref
```

- [ ] **Step 3: 改 `api/schemas.py` —— SourceRef 从 core 导入并 re-export**

把 `api/schemas.py` 中 `SourceRef` 的类定义（第 12-16 行）删除，在文件顶部 import 段之后加入：
```python
from core.agent.source_context import SourceRef  # 领域值对象，re-export 供 DTO 使用
```
（其余 DTO 保持不变；`SourceRef` 名字在本模块仍可被 `from api.schemas import SourceRef` 引用）

- [ ] **Step 4: 删除旧 `api/source_context.py`**

```bash
git rm api/source_context.py
```

- [ ] **Step 5: 冒烟 —— core 工具与 api schemas 均可导入**

Run: `python -c "import core.tools.book_tools; import core.agent.source_context; import api.schemas; from api.schemas import SourceRef; print('ok', SourceRef.__module__)"`
Expected: `ok core.agent.source_context`，无 ImportError

- [ ] **Step 6: 分层守卫转绿**

Run: `python scripts/check_layering.py`
Expected: 退出码 0，`分层守卫通过：core/ 未 import api/`

- [ ] **Step 7: 提交**

```bash
# api/source_context.py 的删除已在 Step 4 由 git rm 暂存，直接一并提交
git add core/agent/source_context.py core/tools/book_tools.py api/schemas.py
git commit -m "refactor: move source_context + SourceRef into core, break core->api dependency"
```

---

## Task 3: 统一主 agent（AgentService 并入 core/agent/agent.py）

**Files:**
- Modify: `core/agent/agent.py`（整体重建为 BookAgent）
- Modify: `api/main.py`（构造 BookAgent）
- Modify: `api/routers/chat.py`（import + 类型）
- Modify: `api/routers/sessions.py`（import + 类型）
- Delete: `api/agent_service.py`

- [ ] **Step 1: 重建 `core/agent/agent.py`**

整体替换为（吸收 AgentService 能力 + 保留 CLI chat）：

```python
"""主 Agent（book 知识库助手）

- 基于 LlamaIndex FunctionAgent（支持流式事件，前端依赖）
- per-session 并发锁，防同会话并发互踩
- 从 DB 历史构造 ChatMemoryBuffer（鸭子类型读 .role/.content，不反向依赖 persistence）
- 保留交互式 CLI chat() 便捷入口
"""
import asyncio
from typing import List, Optional

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool


BOOK_SYSTEM_PROMPT = """你是一个书籍知识库助手，帮助用户从已入库的技术书籍中查找答案。

可用工具：
1. book_search(query, book_title?) — 在书籍知识库中检索内容。
   - 用户问具体技术问题时必须先调用，不要凭记忆作答。
   - 如果用户指明了书名（如"在《MySQL是怎样运行的》里..."），把书名传给 book_title 参数。
   - 留空 book_title 则跨所有书检索。
2. list_books() — 列出当前已入库的书籍清单。
   - 用户问"有哪些书"、"知识库里有什么"时调用。
   - 当 book_search 返回"没有检索到相关内容"时，可调用此工具帮助用户了解可选范围。

回答规则：
- 答案必须基于检索结果。检索为空就如实告知，不要编造。
- 中文回答，简洁清楚，必要时引用书名/章节。
- 不要重复调用同一个工具，除非确实需要换关键词或换书重试。"""


class BookAgent:
    """统一主 agent：一个全局 FunctionAgent + per-session 并发锁。"""

    def __init__(
        self,
        tools: List[FunctionTool],
        llm: LLM,
        system_prompt: str = BOOK_SYSTEM_PROMPT,
        memory_token_limit: int = 4000,
        timeout: float = 300.0,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.memory_token_limit = memory_token_limit

        self.agent = FunctionAgent(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            timeout=timeout,
            verbose=True,
        )
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, session_id: Optional[str]) -> asyncio.Lock:
        """获取 session 锁；session_id 为空返回一次性锁"""
        if not session_id:
            return asyncio.Lock()
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def build_memory(self, db_messages) -> ChatMemoryBuffer:
        """根据数据库历史消息构造 ChatMemoryBuffer。

        db_messages: 按 id 升序的消息行，鸭子类型读取 .role / .content。
        """
        memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
        for m in db_messages:
            role = role_map.get(m.role)
            if role is None or not m.content:
                continue
            memory.put(ChatMessage(role=role, content=m.content))
        return memory

    def reset(self, session_id: str) -> bool:
        """清理指定 session 的并发锁（DB 数据由 sessions 路由删除）"""
        return self._locks.pop(session_id, None) is not None

    async def ask_question(self, question: str, memory: ChatMemoryBuffer) -> str:
        """单轮提问（CLI / 简单调用用）"""
        response = await self.agent.run(user_msg=question, memory=memory)
        return str(response)

    async def chat(self) -> None:
        """交互式 CLI 对话（单会话内存记忆）"""
        memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        print("=" * 50)
        print("book 知识库助手已启动")
        print(f"可用工具：{[t.metadata.name for t in self.tools]}")
        print("输入 'exit' 退出")
        print("=" * 50)
        while True:
            try:
                user_input = input("\n用户：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("再见！")
                break
            answer = await self.ask_question(user_input, memory)
            print(f"\n助手：{answer}")
```

- [ ] **Step 2: 改 `api/main.py` —— 构造 BookAgent**

把第 12 行 `from api.agent_service import AgentService` 改为：
```python
from core.agent.agent import BookAgent
```
把第 60 行 `agent_service = AgentService(tools=tools, llm=llm)` 改为：
```python
    book_agent = BookAgent(tools=tools, llm=llm)
```
把第 63、65 行的路由工厂调用里传入的 `agent_service` 改为 `book_agent`：
```python
    chat_router = create_chat_router(book_agent)
    doc_router = create_documents_router(index_manager)
    sessions_router = create_sessions_router(book_agent)
```

- [ ] **Step 3: 改 `api/routers/chat.py` 的 import 与类型**

把第 10 行 `from api.agent_service import AgentService` 改为：
```python
from core.agent.agent import BookAgent
```
把第 13 行 `from api.source_context import begin_collection, get_sources` 改为：
```python
from core.agent.source_context import begin_collection, get_sources
```
把工厂签名 `def create_chat_router(agent_service: AgentService) -> APIRouter:` 改为：
```python
def create_chat_router(agent_service: BookAgent) -> APIRouter:
```
（变量名 `agent_service` 保持不变，仅换类型，减少 body 改动）

- [ ] **Step 4: 改 `api/routers/sessions.py` 的 import 与类型**

把第 5 行 `from api.agent_service import AgentService` 改为：
```python
from core.agent.agent import BookAgent
```
把工厂签名 `def create_sessions_router(agent_service: AgentService) -> APIRouter:` 改为：
```python
def create_sessions_router(agent_service: BookAgent) -> APIRouter:
```

- [ ] **Step 5: 删除 `api/agent_service.py`**

```bash
git rm api/agent_service.py
```

- [ ] **Step 6: 冒烟 —— 路由与 agent 模块导入**

Run: `python -c "import core.agent.agent; import api.routers.chat; import api.routers.sessions; print('ok')"`
Expected: `ok`，无 ImportError（确认 `AgentService` 已无残留引用）

- [ ] **Step 7: 提交**

```bash
git add core/agent/agent.py api/main.py api/routers/chat.py api/routers/sessions.py
git commit -m "refactor(agent): unify main agent into core BookAgent, drop api AgentService"
```

---

## Task 4: 持久化迁入 core/persistence

**Files:**
- Create: `core/persistence/__init__.py`
- Create: `core/persistence/db.py`（自 api/db.py，修路径深度）
- Create: `core/persistence/repositories.py`（自 api/repositories.py，改 import）
- Delete: `api/db.py`、`api/repositories.py`
- Modify: `api/routers/chat.py`、`api/routers/sessions.py`、`api/main.py`（import 来源）

- [ ] **Step 1: `git mv` 两个文件并建包**

```bash
mkdir -p core/persistence
touch core/persistence/__init__.py
git mv api/db.py core/persistence/db.py
git mv api/repositories.py core/persistence/repositories.py
git add core/persistence/__init__.py
```

- [ ] **Step 2: 修 `core/persistence/db.py` 的 PROJECT_ROOT 深度**

把（原 api/db.py 第 21 行）：
```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```
改为（core/persistence/db.py 现在比 api/ 深一层，需往上三级到项目根）：
```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 3: 改 `core/persistence/repositories.py` 的 import 来源**

把：
```python
from api.db import MessageRow, SessionRow
from api.schemas import SourceRef
```
改为：
```python
from core.persistence.db import MessageRow, SessionRow
from core.agent.source_context import SourceRef
```

- [ ] **Step 4: 改 `api/routers/chat.py` 的 DB/repo import**

把第 9、11 行：
```python
from api import repositories as repo
from api.db import get_session
```
改为：
```python
from core.persistence import repositories as repo
from core.persistence.db import get_session
```

- [ ] **Step 5: 改 `api/routers/sessions.py` 的 DB/repo import**

把第 3、4 行：
```python
from api import repositories as repo
from api.db import get_session
```
改为：
```python
from core.persistence import repositories as repo
from core.persistence.db import get_session
```

- [ ] **Step 6: 改 `api/main.py` 的 init_db import**

把第 13 行 `from api.db import init_db` 改为：
```python
from core.persistence.db import init_db
```

- [ ] **Step 7: 冒烟 —— 持久化与路由导入 + DB 路径正确**

Run: `python -c "from core.persistence.db import DB_PATH, init_db; import core.persistence.repositories; import api.routers.chat; import api.routers.sessions; print('DB_PATH=', DB_PATH)"`
Expected: `DB_PATH=` 指向**项目根**的 `bookkb.db`（路径末尾是 `llmaLearn\bookkb.db`，不是 `core\persistence\bookkb.db`），无 ImportError

- [ ] **Step 8: 提交**

```bash
git add -A core/persistence api/routers/chat.py api/routers/sessions.py api/main.py
git commit -m "refactor(persistence): move SQLite layer into core/persistence, fix db path depth"
```

---

## Task 5: 法条代码冻结进 legacy/

**Files:**
- Create: `legacy/`（含 `workflow/`、`rag/`）
- Move（git mv）: 见下
- Create: `core/workflow/README.md`
- Modify: legacy 内部互相引用 rewire

- [ ] **Step 1: 建 legacy 结构并迁移法条代码**

```bash
mkdir -p legacy/workflow legacy/rag
# 根目录法条入口/脚本
git mv app.py legacy/app.py
git mv init_index.py legacy/init_index.py
git mv run_eval.py legacy/run_eval.py
git mv test_single_file_rag.py legacy/test_single_file_rag.py
# 工具
git mv core/tools/tools.py legacy/tools.py
# 法条 workflow
git mv core/workflow/simple_rag.py legacy/workflow/simple_rag.py
git mv core/workflow/citation_rag.py legacy/workflow/citation_rag.py
git mv core/workflow/multi_strategy_rag.py legacy/workflow/multi_strategy_rag.py
git mv core/workflow/query_engine_workflow.py legacy/workflow/query_engine_workflow.py
git mv core/workflow/eval_workflow legacy/workflow/eval_workflow
# 法条 rag 模块
git mv core/rag/citation_extractor.py legacy/rag/citation_extractor.py
git mv core/rag/citation_graph.py legacy/rag/citation_graph.py
git mv core/rag/parser.py legacy/rag/parser.py
git mv core/rag/vector_store_info.py legacy/rag/vector_store_info.py
git mv core/rag/auto_retriever_prompt.py legacy/rag/auto_retriever_prompt.py
git mv core/rag/data_loo.py legacy/rag/data_loo.py
git mv core/rag/indexer.py legacy/rag/indexer.py
# 评测整套 + 引用图数据
git mv evals legacy/evals
git mv citation_graph.json legacy/citation_graph.json
```

- [ ] **Step 2: 建 legacy 包标记 + 说明**

```bash
touch legacy/__init__.py legacy/workflow/__init__.py legacy/rag/__init__.py
```
创建 `legacy/README.md`：
```markdown
# legacy/ —— 法条业务冻结归档

本目录是项目早期"法律条文 RAG"相关代码的冻结快照，2026-05-26 整合时从主干移出。
**不保证可直接运行**：部分共享逻辑（如 `RAGIndexManager.add_documents` 法条增量索引）
已从 `core/rag/data_loader.py` 删除，仅存于该提交之前的 git 历史。

将来重启法条业务时：
1. 从 git 历史恢复 `data_loader.add_documents` / `_update_citation_graph` / `_fetch_all_nodes`；
2. 校验本目录内 import（已尽力 rewire 为 `legacy.*`）；
3. 重建 chroma `documents` 集合。
```

- [ ] **Step 3: rewire legacy 内部互相引用（core.* → legacy.*）**

在 `legacy/` 下，把对已迁移模块的引用从 `core.` 前缀改为 `legacy.`。逐文件处理（仅改 import 行）：

- `legacy/app.py`：
  - `from core.tools.tools import ...` → `from legacy.tools import ...`
  - `from core.agent.agent import MyAgent` → **删除该行并停用**（MyAgent 已重建为 BookAgent；legacy app 是法条 CLI，标记不可运行即可，保留文件作参考）
- `legacy/tools.py`：
  - `from core.workflow.multi_strategy_rag import ...` → `from legacy.workflow.multi_strategy_rag import ...`
  - `from core.workflow.simple_rag import ...` → `from legacy.workflow.simple_rag import ...`
  - `from core.workflow.citation_rag import ...` → `from legacy.workflow.citation_rag import ...`
  - `from core.rag.citation_graph import CitationGraph` → `from legacy.rag.citation_graph import CitationGraph`
- `legacy/workflow/simple_rag.py`：
  - `from core.rag.vector_store_info import VECTOR_STORE_INFO` → `from legacy.rag.vector_store_info import VECTOR_STORE_INFO`
  - `from core.rag.auto_retriever_prompt import LEGAL_AUTO_RETRIEVER_PROMPT` → `from legacy.rag.auto_retriever_prompt import LEGAL_AUTO_RETRIEVER_PROMPT`
- `legacy/workflow/citation_rag.py`：把 `from core.rag.citation_graph import ...` → `from legacy.rag.citation_graph import ...`（若有）
- `legacy/run_eval.py`、`legacy/evals/*.py`：把 `from core.workflow.simple_rag import ...`、`from evals.evals import ...` 等 `core.`/`evals.` 引用改为 `legacy.workflow.*` / `legacy.evals.*`（按文件实际 import 行逐一改）

> 说明：本步是尽力 rewire，目的是让 legacy 内部引用自洽、可被静态阅读；不追求整体可运行（见 legacy/README.md）。

- [ ] **Step 4: 确认 core 已无法条残留引用**

Run: `python scripts/check_layering.py`
Expected: 退出码 0

Run: `grep -rn "citation\|ArticleSplitter\|vector_store_info\|auto_retriever_prompt\|core.tools.tools\|multi_strategy_rag\|simple_rag\|citation_rag" core/ api/ 2>/dev/null || echo "core/api 已无法条引用"`
Expected: `core/api 已无法条引用`（无任何输出行）

- [ ] **Step 5: 冒烟 —— core/api 主干仍可导入**

Run: `python -c "import core.agent.agent, core.tools.book_tools, core.rag.data_loader, core.persistence.db, core.persistence.repositories, api.routers.chat, api.routers.sessions, api.routers.documents, api.schemas; print('trunk ok')"`
Expected: `trunk ok`

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "refactor: freeze legacy legal code under legacy/, keep trunk book-only"
```

---

## Task 6: 根入口改 book CLI + workflow 口子 + 文档

**Files:**
- Modify: `main.py`（根）
- Create: `core/workflow/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: 重写根 `main.py` 为 book CLI 启动器**

整体替换为：

```python
"""book 知识库助手 CLI 入口。

组装 core 组件 + book 工具 + 主 agent，进入交互式对话。
（Web 服务入口见 api/main.py：python -m uvicorn api.main:app）
"""
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from configs.embedding import configure_embedding
from configs.llm import configure_llm
from core.agent.agent import BookAgent
from core.rag.data_loader import RAGIndexManager
from core.tools.book_tools import create_book_search_tool, create_list_books_tool

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")


async def run() -> None:
    configure_embedding()
    llm = configure_llm()

    index_manager = RAGIndexManager(
        persist_dir=CHROMA_DIR,
        collection_name="book_knowledge",
    )

    tools = [
        create_book_search_tool(index_manager, llm),
        create_list_books_tool(index_manager),
    ]
    agent = BookAgent(tools=tools, llm=llm)
    await agent.chat()


if __name__ == "__main__":
    asyncio.run(run())
```

- [ ] **Step 2: 创建 `core/workflow/README.md`（workflow→tool 口子约定）**

```markdown
# core/workflow/ —— Agent workflow 包

本包用于放置自定义 RAG workflow（LlamaIndex `Workflow`），再封装为 Agent 工具。
当前 book 业务的 `book_search` 走 `core/tools/book_tools.py` 内联检索；
如需更复杂的多步检索流程，按以下约定新增：

## 新增一个 workflow 并封装为 tool

1. 在本目录写 `book_rag.py`：

   ```python
   from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

   class BookRagWorkflow(Workflow):
       def __init__(self, index_manager, llm, **kw):
           super().__init__(**kw)
           self.index_manager = index_manager
           self.llm = llm
       @step
       async def run_step(self, ev: StartEvent) -> StopEvent:
           ...  # 自定义检索 + 合成，返回带 source_nodes 的 Response
   ```

2. 在 `core/tools/book_tools.py` 加工厂：

   ```python
   from core.agent.source_context import add_sources, node_to_source_ref
   from core.workflow.book_rag import BookRagWorkflow

   def create_book_rag_tool(index_manager, llm) -> FunctionTool:
       workflow = BookRagWorkflow(index_manager=index_manager, llm=llm)
       async def book_rag_search(query: str, book_title: str | None = None) -> str:
           result = await workflow.run(query=query, book_title=book_title)
           add_sources([node_to_source_ref(n) for n in result.source_nodes])
           return str(result)
       return FunctionTool.from_defaults(
           fn=book_rag_search, name="book_rag", description="...")
   ```

3. 在 `api/main.py` 与根 `main.py` 的工具装配处，把 `create_book_rag_tool(index_manager, llm)`
   加入 `tools` 列表。

口子三要素：本包常驻 + `book_tools` 工厂约定 + `core.agent.source_context.add_sources` 公开钩子。
```

- [ ] **Step 3: 更新 `CLAUDE.md`（去法条/评测，改运行说明与架构）**

把 `## Running` 段替换为：
```markdown
## Running

```bash
python main.py                                   # book CLI 对话
python -m uvicorn api.main:app --port 8000       # Web 服务（前端对接）
```
```
把顶部项目描述行改为：
```markdown
LLM 学习/实验项目，技术书籍知识库助手（上传 PDF + RAG 问答），使用智谱 AI GLM 作为底层 LLM。
```
把 `## ⚠️ Gotchas` 段中"Ragas 评测需要独立 LLM 实例"与"工具在 app.py 组装层创建"两小节替换为：
```markdown
### 分层：core 不依赖 api

依赖方向单向 `api/`(Web) → `core/`(领域) → `configs/`。守卫：`python scripts/check_layering.py`。

### 工具在组装层创建，注入 Agent

`api/main.py`（Web）与根 `main.py`（CLI）在各自组装层用
`core.tools.book_tools.create_book_search_tool / create_list_books_tool`
创建工具，注入 `core.agent.agent.BookAgent`。新增 workflow 工具见 `core/workflow/README.md`。

### 法条遗留

早期法律条文 RAG 代码已冻结于 `legacy/`（不保证可运行，见 `legacy/README.md`）。
```

- [ ] **Step 4: 冒烟 —— 根 CLI 与 workflow 包导入**

Run: `python -c "import main; import core.workflow; print('cli importable')"`
Expected: `cli importable`（`import main` 只触发 import，不执行 `asyncio.run`）

- [ ] **Step 5: 提交**

```bash
git add main.py core/workflow/README.md CLAUDE.md
git commit -m "refactor: book CLI entrypoint, workflow->tool seam docs, update CLAUDE.md"
```

---

## 最终验证（人工）

- [ ] **Web 服务启动**：`python -m uvicorn api.main:app --port 8000`，访问 `GET /api/health` 返回 `{"status":"ok","vectors":<n>}`
- [ ] **会话历史落盘正确**：发一次 `POST /api/chat`，确认根目录 `bookkb.db` 有新记录（而非 `core/persistence/` 下新建了 db）
- [ ] **CLI 启动**：`python main.py`，问"知识库里有哪些书"，确认 `list_books` 被调用
- [ ] **分层守卫**：`python scripts/check_layering.py` 绿
- [ ] **前端流式**：前端连 `/api/chat/stream`，确认 tool_call / delta / sources 事件正常（FunctionAgent 流式未回归）

---

## 自检记录（spec 覆盖）

- §3 目标布局 → Task 1-6 全覆盖
- §4 模块去向 → Task 1(data_loader 瘦身)、2(source_context)、3(agent+删 agent_service)、4(persistence)、5(legacy)
- §5 workflow→tool 口子 → Task 6 Step 2（README）+ `core/workflow/` 常驻（Task 5 未移走该包）
- §6 六阶段 → Task 1-6 一一对应
- §7 风险点：SourceRef 上移(Task 2)、bookkb.db 路径(Task 4 Step 2/7)、build_memory 鸭子类型(Task 3 Step 1)、每阶段冒烟(各 Task 末)
```
