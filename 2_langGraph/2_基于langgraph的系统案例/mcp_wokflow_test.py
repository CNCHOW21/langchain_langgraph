# @Time    : 2025/9/21 18:42
# @Author  : liuzhou
# @File    : mcp_wokflow_test.py
# @software: PyCharm
import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from utils.config import Config
from ragAgent import get_llm
load_dotenv()
# 获取高德地图 API Key
AMAP_MAPS_API_KEY = os.getenv('AMAP_MAPS_API_KEY')
# Set up MCP client
client = MultiServerMCPClient(
    {
        # 高德地图MCP Server
        "amap-amap-sse": {
            "url": "https://mcp.amap.com/sse?key=" + AMAP_MAPS_API_KEY,
            "transport": "sse",
        }
    }
)

async def get_llm_with_tool():
    tools = await client.get_tools()
    llm_chat,llm_embedding = get_llm(Config.LLM_TYPE)
    model_with_tools = llm_chat.bind_tools(tools)
    # Create ToolNode
    tool_node = ToolNode(tools)
    return model_with_tools,tool_node


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Define call_model function
async def call_model(state: MessagesState):
    messages = state["messages"]
    model_with_tools,tool_node = await get_llm_with_tool()
    response = await model_with_tools.ainvoke(messages)
    print(f"========================={response}")
    return {"messages": [response]}

# Build the graph
model_with_tools,tool_node = asyncio.run(get_llm_with_tool())
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    should_continue,
)
builder.add_edge("tools", "call_model")

# Compile the graph
graph = builder.compile()

async def graph_invoke():
    # Test the graph
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "武汉今天天气怎么样？"}]}
    )
    print(response)

if __name__ == '__main__':
   asyncio.run(graph_invoke())

