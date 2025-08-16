from typing import Annotated
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

"""
使用 matplotlib 显示图片
"""

# 定义state
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 创建Graph，状态机
graph_builder = StateGraph(State)

# 定义模型
llm = init_chat_model("gpt-4o", model_provider="openai")


# 定义一个执行节点
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 定义工作流
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# 查看工作流结构的替代方法
print("=== LangGraph 工作流结构 ===")
print("节点:")
for node_name, node_data in graph.get_graph().nodes.items():
    print(f"  - {node_name}")

print("\n连接:")
for edge in graph.get_graph().edges:
    print(f"  - {edge.source} -> {edge.target}")

# 尝试显示图片
try:
    # 获取 Mermaid 格式的图片
    img_data = graph.get_graph().draw_mermaid_png()

    # 保存并显示
    with open("workflow.png", "wb") as f:
        f.write(img_data)

    # 使用 matplotlib 显示
    img = plt.imread("workflow.png")
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("LangGraph Workflow")
    plt.show()

    print("工作流图片已显示")

except Exception as e:
    print(f"无法显示图片: {e}")
    print("但工作流功能正常，可以继续使用")
