import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs.llm import configure_llm
from configs.embedding import configure_embedding
from core.rag.data_loader import RAGIndexManager
from core.tools.book_tools import create_book_search_tool, create_list_books_tool, create_simple_book_search
from core.agent.agent import BookAgent
from core.persistence.db import init_db
from api.routers.chat import create_chat_router
from api.routers.documents import create_documents_router
from api.routers.sessions import create_sessions_router

# 项目根目录（api/main.py 的上一级），用于拼接绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def _setup_logging() -> None:
    """把应用日志（含 AGENT EVENT）完整写入 logs/agent.log。

    - UTF-8 编码，避免中文乱码
    - RotatingFileHandler：单文件 10MB，保留 5 份，防止无限增长
    - 幂等：--reload 重复导入时不重复加 handler
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    log_path = os.path.join(LOG_DIR, "agent.log")
    already = any(
        isinstance(h, RotatingFileHandler)
        and getattr(h, "baseFilename", "") == os.path.abspath(log_path)
        for h in root.handlers
    )
    if already:
        return
    fh = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s"
    ))
    root.addHandler(fh)


_setup_logging()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动时建库表"""
    await init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="BookKB - 书籍知识库助手", version="0.1.0", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 初始化核心组件
    print("Initializing LLM...")
    llm = configure_llm()

    print("Initializing Embedding...")
    configure_embedding()

    print(f"Initializing Index Manager... ({CHROMA_DIR})")
    index_manager = RAGIndexManager(
        persist_dir=CHROMA_DIR,
        collection_name="book_knowledge",
    )

    # 装配 Agent 工具（工具持有 index_manager 引用，空库时也能启动）
    print("Building Agent tools...")
    tools = [
        # create_book_search_tool(index_manager, llm),
        create_list_books_tool(index_manager),
        create_simple_book_search(index_manager, llm),

    ]
    book_agent = BookAgent(tools=tools, llm=llm)

    # 注册路由
    chat_router = create_chat_router(book_agent)
    doc_router = create_documents_router(index_manager)
    sessions_router = create_sessions_router(book_agent)
    app.include_router(chat_router)
    app.include_router(doc_router)
    app.include_router(sessions_router)

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "vectors": index_manager.chroma_collection.count(),
        }

    return app

app = create_app()

if __name__ == "__main__":
    # 传字符串而非 app 对象，reload 才能工作
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
