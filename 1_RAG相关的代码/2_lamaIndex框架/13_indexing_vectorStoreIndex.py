# @Time    : 2025/8/5 23:20
# @Author  : liuzhou
# @File    : 13_indexing_vectorStoreIndex.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels, DashScopeEmbedding

load_dotenv()

Settings.embed_model = DashScopeEmbedding(DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2, os.getenv("DASHSCOPE_API_KEY"))

# 加载文档
documents = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"]).load_data()

# 切分文档
splitter = TokenTextSplitter(chunk_size=512,chunk_overlap=128)
nodes = splitter.get_nodes_from_documents(documents)

# 构建index，默认在内存中
index = VectorStoreIndex(nodes)

# 获取retriever
retrieve = index.as_retriever(
                similarity_top_k=2
            )

# 检索
results = retrieve.retrieve("deepseek v3数学能力怎么样？")

print(results[0].text)