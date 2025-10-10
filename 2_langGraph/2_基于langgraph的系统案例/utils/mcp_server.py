# @Time    : 2025/9/20 00:58
# @Author  : liuzhou
# @File    : mcp_server.py
# @software: PyCharm
import asyncio
import os
import traceback

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from .llms import get_llm
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
# 获取高德地图 API Key
AMAP_MAPS_API_KEY=os.getenv('AMAP_MAPS_API_KEY')
client = MultiServerMCPClient({
    # 高德地图MCP Server
    "amap-amap-sse": {
        "url": "https://mcp.amap.com/sse?key="+AMAP_MAPS_API_KEY,
        "transport": "sse",
    },
    # "mysql_mcp_server": {
    #     "url": "http://127.0.0.1:8888/mcp",
    #     "transport": "streamable_http",
    # }
})

async def get_mcp_tools():
    # 实例化MCP Server客户端
    # 从MCP Server中获取可提供使用的全部工具
    tools = await client.get_tools()
    return tools

# 定义并运行agent
async def get_mcp(question: str,llm_chat):
    tools = await get_mcp_tools()
    # 基于内存存储的short-term
    checkpointer = InMemorySaver()
    system_message = SystemMessage(content=(
        "你是一个AI助手，使用高德地图工具获取信息。"
    ))
    agent = create_react_agent(
        llm_chat,
        tools,
        prompt=system_message,
        checkpointer=checkpointer
    )
    try:
        config = {"configurable": {"thread_id": "1"}}
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)]}, config
        )
        return response

    #     messages = response["messages"]
    #     # 找到最后一个 AIMessage
    #     content = ""
    #     last_aimessage = None
    #     for message in messages:
    #         if isinstance(message, AIMessage):
    #             last_aimessage = message
    #     return last_aimessage.content
    except Exception as e:
        traceback.print_exc()

def call_mcp_tools(question: str,llm_chat):
    response = asyncio.run(get_mcp(question,llm_chat))
    return response

if __name__ == '__main__':
    question = "武汉今天天气怎么样？"
    print(call_mcp_tools(question))



