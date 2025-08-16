# @Time    : 2025/8/6 00:14
# @Author  : liuzhou
# @File    : 14_indexing_vectorStoreIndex_qdrant.py
# @software: PyCharm
import os

from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from qdrant_client.models import Distance

Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2, api_key=os.getenv("DASHSCOPE_API_KEY"))

documents = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"]).load_data()

parser = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )

nodes = parser.get_nodes_from_documents(documents)

client = QdrantClient(location=":memory:")
collection_name = "demo"
colletion = client.create_collection(
    collection_name,
    vectors_config=VectorParams(size=1536,distance=Distance.COSINE))

vector_store = QdrantVectorStore(client = client, collection_name=collection_name)
# storage: 指定存储空间
storage_context = StorageContext.from_defaults(vector_store = vector_store)
# 创建 index：通过 Storage Context 关联到自定义的 Vector Store
index = VectorStoreIndex(nodes, storage_context=storage_context)
# 获取 retriever
vector_retriever = index.as_retriever(similarity_top_k=2)
# 检索
results = vector_retriever.retrieve("deepseek v3数学能力怎么样")

for i, result in enumerate(results):
    print(f"[{i}] {result.text}\n")