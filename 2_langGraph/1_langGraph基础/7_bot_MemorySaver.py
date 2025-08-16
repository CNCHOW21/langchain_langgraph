# @Time    : 2025/8/9 17:51
# @Author  : liuzhou
# @File    : 1_context_bot.py
# @software: PyCharm
import json
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# # 定义state
# class State(TypedDict):
#     # 状态变量messages类型是list，更新方式是add_messages
#     # add_messages是内置的一个办法，将新的消息追加在原列表后面
#     messages: Annotated[list, add_messages]

# MessagesState是一个State内置对象
# add_messages是内置的一个方法，将新的消息列表追加到原列表后面
# 创建Graph，状态机
graph_builder  = StateGraph(MessagesState)

# 定义模型
llm = init_chat_model("gpt-4o", model_provider="openai")

# 定义一个执行节点
# 输入是State，输出是系统回复
def chatbot(state: MessagesState):
    # 调用大模型，并返回消息（列表）
    return {"messages": [llm.invoke(state["messages"])]}

# 定义工作流，包含节点和边
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 为了添加持久性，我们需要在编译图表时传递检查点，使用MemorySaver就可以记住以前的消息
graph = graph_builder.compile(checkpointer=MemorySaver())

# 可视化展示工作流，在pycharm中无法展示，参考4_matplotlib_bot.py
# try:
#     display(Image(data=graph.get_graph().draw_png()))
# except Exception as e:
#     print(f"运行时异常：{e}")

# langGraph消息的共享是通过线程实现的
config = {"configurable":{"thread_id": "userid_654643511"}}

def stream_graph_updates(user_input: str, configs : json):
    # 向graph传入一条消息
    for event in graph.stream({"messages":[{"role":"user","content":user_input}]}, configs):
        for value in event.values():
            if "messages" in value and isinstance(value["messages"][-1], AIMessage):
                print("Assistant:", value["messages"][-1].content)

def run():
    global config  # 声明使用全局变量
    while True:
        user_input = input("User : ")
        if user_input.strip() == '':
            break
        if user_input.strip() == 'x':
            config["configurable"]["thread_id"] = "userid_654643512"# 通过线程id区分是否是同一个用户
        stream_graph_updates(user_input, config)

if __name__ == '__main__':
    run()