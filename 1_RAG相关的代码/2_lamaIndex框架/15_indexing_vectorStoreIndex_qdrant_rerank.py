# @Time    : 2025/8/6 00:14
# @Author  : liuzhou
# @File    : 14_indexing_vectorStoreIndex_qdrant.py
# @software: PyCharm
import os

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv('DASHSCOPE_API_KEY'))
Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2, api_key=os.getenv("DASHSCOPE_API_KEY"))

documents = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"]).load_data()

parser = TokenTextSplitter(
        chunk_size=100,
        chunk_overlap=10
    )

nodes = parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)

retrieve = index.as_retriever(
                similarity_top_k=10
            )

retrieve_res = retrieve.retrieve("deepseek v3数学能力怎么样？")

llm_rerank = LLMRerank(top_n=2)

rerank_nodes = llm_rerank.postprocess_nodes(retrieve_res, query_str="deepseek v3有多少参数?")


print(f"[0] {nodes[0]}")
print(f"[1] {nodes[1]}")

print("="*100)

for i,node in enumerate(rerank_nodes):
    print(f"[{i}] {node}")