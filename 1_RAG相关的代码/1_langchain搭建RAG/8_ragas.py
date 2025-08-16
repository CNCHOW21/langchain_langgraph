# @Time    : 2025/8/8 02:21
# @Author  : liuzhou
# @File    : 8_ragas.py
# @software: PyCharm
import os

from datasets import Dataset
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

docs = PyPDFLoader("./data/浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf").load()

# 初始化大语言模型
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
llm = Tongyi(
    model_name="qwen-max",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

# 创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=embeddings
)
# 创建内存存储对象
store = InMemoryStore()
# 创建父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2}
)

# 添加文档集
retriever.add_documents(docs)

# 创建prompt模板
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建 chain（LCEL langchain 表达式语言）
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

questions = [
    "客户经理被投诉了，投诉一次扣多少分？",
    "客户经理每年评聘申报时间是怎样的？",
    "客户经理在工作中有不廉洁自律情况的，发现一次扣多少分？",
    "客户经理不服从支行工作安排，每次扣多少分？",
    "客户经理需要什么学历和工作经验才能入职？",
    "个金客户经理职位设置有哪些？"
]

ground_truths = [
    "每投诉一次扣2分",
    "每年一月份为客户经理评聘的申报时间",
    "在工作中有不廉洁自律情况的每发现一次扣50分",
    "不服从支行工作安排，每次扣2分",
    "须具备大专以上学历，至少二年以上银行工作经验",
    "个金客户经理职位设置为：客户经理助理、客户经理、高级客户经理、资深客户经理"
]

answers = []
contexts = []

# Inference
for query in questions:
    answers.append(chain.invoke({"question": query}))
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# 评测结果
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    embeddings=embeddings
)

df = result.to_pandas()

df.to_html('output.html', index=False)