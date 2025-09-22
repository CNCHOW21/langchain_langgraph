# @Time    : 2025/9/21 16:02
# @Author  : liuzhou
# @File    : mcp_server_test.py
# @software: PyCharm
import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from utils.config import Config
from .ragAgent import get_llm

load_dotenv()
# 获取高德地图 API Key
AMAP_MAPS_API_KEY = os.getenv('AMAP_MAPS_API_KEY')

client = MultiServerMCPClient(
    {
        # 高德地图MCP Server
        "amap-amap-sse": {
            "url": "https://mcp.amap.com/sse?key="+AMAP_MAPS_API_KEY,
            "transport": "sse",
        }
    }
)

async def call_agent(quesiton : str):
    tools = await client.get_tools()
    llm_chat,llm_embedding = get_llm(Config.LLM_TYPE)
    agent = create_react_agent(
        llm_chat,
        tools,
    )
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": quesiton}]}
    )
    messages = response["messages"]
    # 找到最后一个 AIMessage
    content = ""
    last_aimessage = None
    for message in messages:
        if isinstance(message, AIMessage):
            last_aimessage = message
    return last_aimessage.content

def call_mcp_tools(question: str):
    return asyncio.run(call_agent(question))

if __name__ == '__main__':
    print(call_mcp_tools("武汉一日游计划，最好是提供具体的公共交通"))