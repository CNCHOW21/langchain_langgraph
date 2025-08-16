# @Time    : 2025/8/5 19:10
# @Author  : liuzhou
# @File    : 2_simple_RAG.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.llms.openai_like import OpenAILike

"""
创建
"""
# 加载env环境
load_dotenv()

# 设置生成模型
Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv('DASHSCOPE_API_KEY'))

# LlamaIndex默认使用的大模型被替换为百炼
# Settings.llm = OpenAILike(
#     model="qwen-max",
#     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     is_chat_model=True
# )

# 设置向量模型
Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2)

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 创建向量索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 提问
response = query_engine.query("deepseek V3有多少参数？")

print(response)