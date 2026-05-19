import os

from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
zhipu_api_key = os.getenv('ZHIPU_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
def configure_llm():
    """配置 LLM 并设置全局参数"""
    llm = OpenAILike(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        api_key=deepseek_api_key,
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=True,
    )
    Settings.llm = llm
    return llm
