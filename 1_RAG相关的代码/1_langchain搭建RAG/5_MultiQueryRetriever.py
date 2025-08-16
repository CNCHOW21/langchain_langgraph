# @Time    : 2025/8/7 21:27
# @Author  : liuzhou
# @File    : 5_MultiQueryRetriever.py
# @software: PyCharm

import os

from dotenv import load_dotenv
from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi

load_dotenv()
# 初始化大语言模型
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
llm = Tongyi(
    model_name="qwen-max",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2", # v3会报错。。。
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 加载向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
vectorstore = FAISS.load_local("./vector_db", embeddings, allow_dangerous_deserialization=True)

# 创建MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 示例查询
query = "客户经理的考核标准是什么？"
# 执行查询
results = retriever.get_relevant_documents(query)

# 打印结果
print(f"查询: {query}")
print(f"找到 {len(results)} 个相关文档:")
for i, doc in enumerate(results):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)