from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from .config import Config


def get_tools(llm_embedding):
    """
    创建并返回工具列表

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储

    Returns:
        list: 工具列表
    """

    # 创建 Chroma 向量存储实例
    vectorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )
    # 将向量存储转换为检索器
    retriever = vectorstore.as_retriever()
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是健康档案查询工具，搜索并返回有关用户的健康档案信息。"
    )

    # 自定义 multiply 工具
    # @tool
    # def multiply(a: float, b: float) -> float:
    #     """这是计算两个数的乘积的工具，返回最终计算结果"""
    #     return a * b

    # 自定义 search 工具，不要使用search作为名称
    @tool
    def search_web(question: str) -> str:
        """这是一个用于在网络上搜索信息的工具，可以用来查找任何互联网上的内容。
        arg question: 问题描述
        """
        print(f"=====调用web搜索工具，查询内容是：{question}")
        search_tool = TavilySearchResults()
        res = search_tool.invoke(question)
        return res[0].get("content")


    # 返回工具列表
    return [retriever_tool, search_web]