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


# 创建检索
retrieve_orig = index.as_retriever(similarity_top_k=5)

# 重排模型
rerank = LLMRerank(top_n = 2)

# retrieve_res = retrieve_orig.retrieve("deepseek v3数学能力怎么样?")
# nodes_rerank = rerank.postprocess_nodes(retrieve_res, query_str = "deepseek v3数学能力怎么样?")
# print(nodes_rerank)

# 创建查询引擎
qa_engine = RetrieverQueryEngine.from_args(
    retriever=retrieve_orig,
    node_postprocessors=[rerank],
    response_mode=ResponseMode.REFINE
)

# 执行查询
response = qa_engine.query("deepseek v3数学能力怎么样?")

print(response)