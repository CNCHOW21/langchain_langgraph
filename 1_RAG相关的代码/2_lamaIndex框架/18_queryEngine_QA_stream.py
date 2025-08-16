# @Time    : 2025/8/6 14:05
# @Author  : liuzhou
# @File    : queryEngine_QA.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

# 加载环境
load_dotenv()

# 配置模型
Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv('DASHSCOPE_API_KEY'))
Settings.embed_model = DashScopeEmbedding(DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2, os.getenv("DASHSCOPE_API_KEY"))

# 加载文档
documents = SimpleDirectoryReader("./data", required_exts=[".pdf"]).load_data()

# 文档切分
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=128)
nodes = splitter.get_nodes_from_documents(documents)

# 创建index
index = VectorStoreIndex(nodes)

# 直接创建查询引擎
query = index.as_query_engine(streaming=True)
response = query.query("deepseek v3数学能力怎么样?")
print(response)