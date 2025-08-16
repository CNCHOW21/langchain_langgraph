# @Time    : 2025/8/6 14:05
# @Author  : liuzhou
# @File    : 20_queryEngine_chat_stream.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from exceptiongroup import catch
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.node_parser import TokenTextSplitter
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
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_QUESTION,
    verbose=True  # 显示 debug 信息
)

# 流式输出
# streaming_response1 = chat_engine.stream_chat("deepseek v3数学能力怎么样?")
# for token in streaming_response1.response_gen:
#     print(token, end="", flush=True)

# streaming_response2 = chat_engine.stream_chat("代码能力呢？")
# for token in streaming_response2.response_gen:
#     print(token, end="", flush=True)

while True:
    query_str = input("\nQ:")
    if query_str == "q":
        break
    streaming_response1 = chat_engine.stream_chat(query_str)
    for token in streaming_response1.response_gen:
        print(token, end="", flush=True)