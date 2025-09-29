# @Time    : 2025/9/26 14:08
# @Author  : liuzhou
# @File    : model_chain_agent.py
# @software: PyCharm
import asyncio

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.config import Config
from utils.llms import get_llm
from utils.mcp_server import get_mcp_tools

# 创建聊天模型
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个专业的IT顾问，使用高德MCP服务，一步一步完成"),
    ("human", "{input}")
])

# 获取MCP工具集
mcp_tools = asyncio.run(get_mcp_tools())
# 绑定工具
llm_chat_with_tool = llm_chat.bind_tools(mcp_tools)
# 创建处理链
chain = prompt | llm_chat_with_tool


async def main():
    # 直接调用llm_chat_with_tool看看是否有输出
    response_llm = await llm_chat_with_tool.ainvoke("武汉名湖豪庭附近10公里的充电桩")
    print(f"直接调用llm_chat_with_tool: {response_llm}")
    # 手动使用StrOutputParser进行解析
    parser = StrOutputParser()
    try:
        parsed_response = parser.parse(response_llm)
        print(f"手动解析后的响应: {parsed_response}")
    except Exception as e:
        print(f"手动解析时发生错误: {e}")
    # 使用不带StrOutputParser的链式调用
    response_chain = await (prompt | llm_chat_with_tool).ainvoke("武汉名湖豪庭附近10公里的充电桩")
    print(f"使用无StrOutputParser的chain.ainvoke: {response_chain}")

if __name__ == '__main__':
    asyncio.run(main())

