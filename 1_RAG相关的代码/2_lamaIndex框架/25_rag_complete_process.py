# @Time    : 2025/8/6 18:11
# @Author  : liuzhou
# @File    : 25_rag_complete_process.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# 加载环境
load_dotenv(verbose=True)

# 加载模型
Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv("DASHSCOPE_API_KEY"))
Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
                                          api_key=os.getenv("DASHSCOPE_API_KEY"))

# 加载文档
documents = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"]).load_data()

# 切分文档
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
nodes = splitter.get_nodes_from_documents(documents)

# 创建collection
COLLECTION_NAME = "demo"
EMBEDDING_DIM = 1536
client = QdrantClient(path="./qdrant")

if client.collection_exists(collection_name = COLLECTION_NAME) == False:
    client.create_collection(collection_name=COLLECTION_NAME,
                             vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))

# 创建vector_store
vector_store = QdrantVectorStore(collection_name= COLLECTION_NAME, client=client)

# 创建index , 指定 Vector Store 的 Storage 用于 index
storage_context = StorageContext.from_defaults(vector_store = vector_store)
index = VectorStoreIndex.from_documents(documents = documents, storage_context=storage_context)

# 定义重排序模型
re_rank = LLMRerank(top_n=2)

# 阈值控制
similary_score = SimilarityPostprocessor(similarity_cutoff=0.6)

# 定义RAG_fusion检索器
fusion_retrieve = QueryFusionRetriever(
                    retrievers = [index.as_retriever()],
                    similarity_top_k = 5, # 检索召回 top k 结果
                    num_queries = 3, # 生成 query 数
                    use_async = False,
                    query_gen_prompt="" #自定义提示词
                )

# 构建单轮query engine
query_engine = RetrieverQueryEngine.from_args(
    fusion_retrieve,
    node_postprocessors=[re_rank,similary_score],
    vector_store_query_mode="hybrid",  # 混合检索模式
    alpha=0.5,  # 语义和关键字权重平衡
    response_synthesizer=get_response_synthesizer(
        response_mode = ResponseMode.REFINE
    )
)

# 对话引擎
chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine = query_engine,
                # condense_question_prompt=""
            )


while True:
    question = input("Q:")
    if question.strip() == '':
        break
    response = chat_engine.chat(question)
    print(f"AI:{response}")