import logging

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

logger = logging.getLogger(__name__)
#初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
def configure_embedding():
    logger.info("开始配置embedding模型")
    """配置文本嵌入模型"""
	# 指定了一个预训练的sentence-transformer模型的路径
    model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cpu"  # 如果你有 NVIDIA 显卡，可以改为 "cuda"
)
    # 必须设置全局 Embedding
    Settings.embed_model = model
    logger.info("embedding模型配置完成")
    return model
