"""评测侧配置：judge LLM / embedding / 路径。

评测侧用 ragas 原生 llm_factory（instructor 结构化输出，collections 指标必需），
沿用 legacy 的智谱 GLM，与被测系统（项目 DeepSeek）解耦。
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────────────────────
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
DATASET_DIR = os.path.join(EVAL_DIR, "dataset")
TESTSET_PATH = os.path.join(DATASET_DIR, "testset.jsonl")
TESTSET_DRAFT_PATH = os.path.join(DATASET_DIR, "testset.draft.jsonl")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
# chroma 用绝对路径锚定项目根，避免脚本因工作目录不同读到错误/空库
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
CHROMA_COLLECTION = "book_knowledge"

# ── 评测侧模型 ────────────────────────────────────────
EVAL_LLM_MODEL = "deepseek-v4-flash"
EVAL_LLM_BASE_URL = "https://api.deepseek.com/v1"
EVAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def make_eval_llm():
    """评测 judge LLM：llm_factory + DeepSeek OpenAI 兼容端点。

    deepseek-v4-flash 是思考模型，默认会把输出预算烧在 reasoning_content 上、
    content 返回空，导致 instructor 结构化解析失败（IncompleteOutputException）。
    必须关闭 thinking（同 configs/llm.py 的处理），并放大 max_tokens（ragas 默认仅 1024）。
    """
    import openai
    from ragas.llms import llm_factory

    client = openai.AsyncOpenAI(
        base_url=EVAL_LLM_BASE_URL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    return llm_factory(
        EVAL_LLM_MODEL,
        client=client,
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=2048,
    )


def make_eval_embeddings():
    """评测 embedding：ragas HuggingFaceEmbeddings。"""
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=EVAL_EMBED_MODEL)
