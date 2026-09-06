import json

import requests
from langchain.retrievers import MultiQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

from .config import Config
from .qwen_agent_run import run_agent


def get_tools(llm_embedding, llm_chat):
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
    # retriever = vectorstore.as_retriever()

    # 创建MultiQueryRetriever查询扩展
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm_chat
    )
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是健康档案查询工具，搜索并返回有关用户的健康档案信息。"
    )

    # 自定义 search 工具，不要使用search作为名称
    @tool
    def search_web(question: str) -> str:
        """这是一个用于在网络上搜索信息的工具，可以用来查找任何互联网上的内容。
        """
        search_tool = TavilySearchResults()
        res = search_tool.invoke(question)
        return res[0].get("content")


    # 自定义Qwen-Agent工具，测试是否能取代langchain+RAG
    @tool
    def qwen_agent_query(question: str) -> str:
        """这是一个用于查询个人信息的工具，可以用来查找个人的offer信息。
        """
        response = run_agent(question)
        return response

    # 远程调用第三方api
    @tool
    def query_huangli(year: str, month: str, day: str) -> str:
        """这是一个用于查询黄历的工具，按年月日查询农历、星座、生肖、胎神、
        喜神、五行、冲、煞、吉日、值日天神、凶神、吉神宜趋、财神、喜神、福神、
        岁次、宜、忌、星期等黄历信息，数据范围1900-2100年
                """
        url = 'http://gwgp-gjjifhqt3mv.n.bdcloudapi.com/huangli/date'
        params = {}
        params['year'] = year
        params['month'] = month
        params['day'] = day

        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'X-Bce-Signature': 'AppCode/83af7821f4db4b68a990056b89e6da8a'
        }
        r = requests.request("GET", url, params=params, headers=headers)
        result= r.content.decode('utf-8')
        return result

    # 根据城市天气生成图片，调用dify接口
    @tool
    def create_city_image(city: str, weather: str) -> str:
        """这是一个根据城市，天气生成图片的工具，可以用来生成城市的天气图片。
                """
        params = {}
        params['city'] = city
        params['weather'] = weather
        url = 'http://localhost/v1/chat-messages'
        headers = {
            'Authorization': f'Bearer app-TgLrmlIztqRaM7vGDSExAwFH',  # 替换 {your_api_key} 为你实际的 API 密钥
            'Content-Type': 'application/json'
        }
        data = {
            "inputs": params,
            "query": "开始",
            "response_mode": "streaming",
            "conversation_id": "",
            "user": "abc-123",
            "files": [
                {
                    "type": "image",
                    "transfer_method": "remote_url",
                    "url": "https://cloud.dify.ai/logo/logo-site.png"
                }
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = ""
        if response.status_code == 200:
            data = response.text
            for line in data.split('data: '):
                if not line.strip():
                    continue
                if not line.strip().endswith("}"):
                    continue
                record = json.loads(line)
                # 查找 event 等于 workflow_finished 的记录
                if record['event'] == 'workflow_finished':
                    outputs = record['data']['outputs']
                    print("Found the required 'outputs':", outputs)
                    result = outputs
                    break
        return result

    # 返回工具列表
    return [retriever_tool, search_web, qwen_agent_query, query_huangli, create_city_image]
