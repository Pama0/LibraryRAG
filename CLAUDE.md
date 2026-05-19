# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

这是一个 LLM 学习/实验项目，专注于 RAG（检索增强生成）和 Agent 开发，使用智谱 AI GLM 作为底层 LLM。

**核心功能：**
- **RAG 系统**：基于 LlamaIndex 的文档检索问答系统
- **Agent 开发**：支持工具调用的智能助手
- **评测框架**：使用 Ragas 进行 RAG 系统质量评估

## Project Structure

```
llmaLearn/
├── main.py                    # 主入口
├── app.py                     # 应用组装层（Agent + Tools）
├── run_eval.py                # 评测入口（项目根目录）
│
├── configs/                   # 配置模块
│   ├── llm.py                 # LLM 配置 (ZhipuAI GLM)
│   └── embedding.py           # Embedding 配置 (BGE-small-zh)
│
├── core/                      # 核心模块
│   ├── agent/
│   │   └── agent.py           # ReActAgent 封装
│   ├── rag/
│   │   ├── data_loader.py     # 文档加载与索引
│   │   └── indexer.py         # 索引管理
│   ├── tools/
│   │   └── tools.py           # 工具定义 (RAG Tools)
│   └── workflow/
│       ├── simple_rag.py      # 简单 RAG 工作流
│       └── multi_strategy_rag.py  # 多策略 RAG 工作流
│
├── evals/                     # 评测模块
│   ├── evals.py               # 评测逻辑 (@experiment)
│   ├── run_eval.py            # 评测运行（需用 -m 运行）
│   ├── generate_testset.py    # 测试集生成
│   └── dataset/
│       └── testset.csv        # 测试数据
│
├── data/                      # RAG 知识库文档
└── temp_data/                 # 临时测试代码
```

## Development Setup

### Environment
- Python 3.12+ with virtual environment (`.venv`)
- Activate: `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\activate` (PowerShell)

### Key Dependencies
- `llama-index` - RAG 框架
- `llama-index-llms-openai-like` - OpenAI 兼容接口（用于 ZhipuAI）
- `llama-index-embeddings-huggingface` - 本地 Embedding 模型
- `llama-index-vector-stores-chroma` - ChromaDB 向量存储
- `ragas` - RAG 评测框架
- `chromadb` - 向量数据库

### Configuration
- API keys stored in `.env`:
  ```
  ZHIPU_API_KEY=your_api_key
  ```
- Use `python-dotenv` to load environment variables

## Running the Code

```bash
# 运行主应用（Agent 对话）
python main.py

# 运行评测
python run_eval.py

# 或以模块方式运行子模块
python -m evals.run_eval
```

## Architecture

### 组件分层

```
┌─────────────────────────────────────────────────────────────┐
│                    app.py (组装层)                           │
│         初始化配置 → 创建工具 → 组装 Agent                    │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Agent 层       │  │    Tools 层      │  │  Workflow 层    │
│   MyAgent       │  │  FunctionTool   │  │  SimpleRAG      │
│  (ReActAgent)   │  │  - simple_rag   │  │  MultiStrategy  │
└─────────────────┘  │  - multi_search │  └─────────────────┘
                     └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG 层                                    │
│  data_loader → VectorStoreIndex → ChromaDB                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Configs 层                                │
│  LLM: OpenAILike (ZhipuAI GLM-4-flash)                      │
│  Embedding: HuggingFace (BAAI/bge-small-zh-v1.5)            │
└─────────────────────────────────────────────────────────────┘
```

### RAG 工作流

**SimpleRagWorkflow**: 基础检索
```
Query → VectorStoreIndex.query() → Response
```

**MultiStrategyRAGWorkflow**: 多策略并行 + 评判
```
Query → 判断质量 → [坏查询则改进]
                      ↓
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    Naive(top_k=5)  HighTopK(20)  Rerank(20+LLM)
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              LLM 评判选择最佳结果
```

## Evaluation Module

### 测试数据格式 (testset.csv)

| 列名 | 说明 |
|-----|------|
| `user_input` | 测试问题 |
| `reference_contexts` | 参考上下文 |
| `reference` | 标准答案 |
| `persona_name` | 测试角色 |
| `query_style` | 查询风格 |
| `query_length` | 查询长度 |

### 评测 LLM 要求

**重要**: Ragas 评测需要独立的 LLM 实例，不能复用 RAG 的 LLM。

```python
# RAG 用 LLM (LlamaIndex 接口)
from llama_index.llms.openai_like import OpenAILike
rag_llm = OpenAILike(model="glm-4-flash", ...)

# 评测用 LLM (Ragas 接口 - 支持 agenerate + response_model)
from ragas.llms import llm_factory
import openai
client = openai.OpenAI(base_url="...", api_key="...")
eval_llm = llm_factory("glm-4-flash", client=client)
```

### 运行评测

```bash
# 方式 1: 运行根目录入口脚本（推荐）
python run_eval.py

# 方式 2: 以模块方式运行
python -m evals.run_eval
```

### 生成测试集

```bash
cd evals
python -m generate_testset
```

## Important Patterns

### 模块导入规则

```python
# ✅ 正确: 在项目根目录运行的脚本使用绝对导入
from configs.llm import configure_llm
from core.rag.data_loader import load_and_process_data
from evals.evals import evaluate_rag

# ✅ 正确: 子模块使用相对导入
from .evals import evaluate_rag

# ❌ 错误: 在子目录直接运行脚本
# python evals/run_eval.py  → 会导致 ModuleNotFoundError
```

### Agent 工具创建模式

```python
# 工具在 app.py 组装层创建，注入到 Agent
from core.tools.tools import create_simple_rag_tool, create_multi_strategy_rag_tool

simple_tool = create_simple_rag_tool(query_engine, name="simple_rag")
multi_tool = create_multi_strategy_rag_tool(index, llm)

agent = MyAgent(tools=[simple_tool, multi_tool], llm=llm, ...)
```

## Technical Decisions

| Domain | Choice | Reason |
|--------|--------|--------|
| LLM Provider | ZhipuAI GLM-4-flash | 国产模型，中文能力强，成本低 |
| Embedding | BAAI/bge-small-zh-v1.5 | 中文向量模型，体积小，效果好 |
| Vector Store | ChromaDB | 轻量级，持久化，开发友好 |
| RAG Framework | LlamaIndex | 完整的 RAG 工具链 |
| Evaluation | Ragas | 专业的 RAG 评测框架 |
| Agent | ReActAgent | 支持工具调用，推理透明 |

## Code Style

- Use `async/await` for all I/O operations
- Pydantic models for structured data
- Type hints on all function signatures
- Chinese comments acceptable for this project
