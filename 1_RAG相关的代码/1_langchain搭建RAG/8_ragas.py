# @Time    : 2025/8/8 02:21
# @Author  : liuzhou
# @File    : 8_ragas.py
# @software: PyCharm

import os
from datasets import Dataset
from dotenv import load_dotenv
# PDF文档加载器
from langchain.document_loaders import PyPDFLoader
# 父文档检索器（小块检索、大块返回，提升精度）
from langchain.retrievers import ParentDocumentRetriever
# 向量数据库Chroma
from langchain_chroma import Chroma
# 阿里通义千问 嵌入模型
from langchain_community.embeddings import DashScopeEmbeddings
# 阿里通义千问 LLM
from langchain_community.llms.tongyi import Tongyi
# 输出解析器：把模型返回转成字符串
from langchain_core.output_parsers import StrOutputParser
# 提示词模板
from langchain_core.prompts import ChatPromptTemplate
# LCEL 并行组装字典
from langchain_core.runnables import RunnableMap
# 内存文档存储：存父文档
from langchain_core.stores import InMemoryStore
# 文本递归分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 .env 环境变量（API_KEY）
load_dotenv()

# 加载本地PDF考核办法文档
docs = PyPDFLoader("./data/浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf").load()

# 从环境变量获取通义千问API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化通义千问大模型
llm = Tongyi(
    model_name="qwen-max",      # 使用千问max模型
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 初始化文本嵌入向量模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 阿里官方嵌入模型
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 父文档分割器：分割成较大块（最终返回给LLM的上下文）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

# 子文档分割器：分割成较小块（用于向量检索匹配）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

# 初始化Chroma向量库
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings
)

# 内存存储：用来保存完整的父文档
store = InMemoryStore()

# 创建【父文档检索器】
# 检索时：用小块向量检索 → 找到后返回对应的大块父文档
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,    # 向量库（存子块）
    docstore=store,             # 文档存储（存父块）
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2}      # 每次检索返回2条最相关内容
)

# 将PDF全文加入检索器（自动切块、建向量、存父文档）
retriever.add_documents(docs)

# 定义问答Prompt模板（英文模板，要求简洁回答）
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# 构建提示词模板对象
prompt = ChatPromptTemplate.from_template(template)

# ====================== 构建 RAG 问答链 ======================
chain = RunnableMap({
    # 从输入问题中检索相关上下文
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    # 直接透传问题
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

# 测试问题列表
questions = [
    "客户经理被投诉了，投诉一次扣多少分？",
    "客户经理每年评聘申报时间是怎样的？",
    "客户经理在工作中有不廉洁自律情况的，发现一次扣多少分？",
    "客户经理不服从支行工作安排，每次扣多少分？",
    "客户经理需要什么学历和工作经验才能入职？",
    "个金客户经理职位设置有哪些？"
]

# 标准答案（真实标签）
ground_truths = [
    "每投诉一次扣2分",
    "每年一月份为客户经理评聘的申报时间",
    "在工作中有不廉洁自律情况的每发现一次扣50分",
    "不服从支行工作安排，每次扣2分",
    "须具备大专以上学历，至少二年以上银行工作经验",
    "个金客户经理职位设置为：客户经理助理、客户经理、高级客户经理、资深客户经理"
]

# 用于保存模型生成的答案
answers = []
# 用于保存检索到的上下文内容（RAGAS评测需要）
contexts = []

# 批量推理：对每个问题执行RAG问答
for query in questions:
    # 调用RAG链获取答案
    answers.append(chain.invoke({"question": query}))
    # 获取检索到的原文上下文
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# 把问题、答案、检索内容、标准答案构造成字典
data = {
    "user_input": questions,        # 用户问题
    "response": answers,            # 模型生成答案
    "retrieved_contexts": contexts, # 检索到的上下文
    "reference": ground_truths      # 标准答案
}

# 转换成RAGAS需要的Dataset格式
dataset = Dataset.from_dict(data)

# ====================== RAG 效果评测（RAGAS） ======================
from ragas import evaluate
# 引入评测指标
from ragas.metrics import (
    faithfulness,       # 忠实度：答案是否符合上下文
    answer_relevancy,   # 答案相关性：答案是否切题
    context_recall,     # 上下文召回率：是否找到正确原文
    context_precision,  # 上下文精确率：检索内容是否有用
)

# 执行RAG评测
result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    embeddings=embeddings  # 用于计算相关性的嵌入模型
)

# 转成Pandas DataFrame
df = result.to_pandas()

# 导出成HTML文件，方便查看完整评测表格
df.to_html('output.html', index=False)