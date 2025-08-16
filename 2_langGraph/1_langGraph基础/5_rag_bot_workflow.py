# @Time    : 2025/8/9 23:39
# @Author  : liuzhou
# @File    : 5_rag_bot_workflow.py
# @software: PyCharm
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from matplotlib import pyplot as plt
from typing_extensions import Annotated

load_dotenv()
# 定义模型
llm = init_chat_model("gpt-4o", model_provider="openai")

# 加载文档
loader = PyMuPDFLoader("./data/deepseek-v3-1-4.pdf")
pages = loader.load_and_split()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=200,
                    length_function=len,
                    add_start_index=True
                )

texts = text_splitter.create_documents(
    [page.page_content for page in pages[:2]]
)

# 灌库
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
db = FAISS.from_documents(texts, embeddings)

# 检索top-5结果
retriever = db.as_retriever(search_kwargs={"k":5})

# prompt模块
template = """请根据对话历史和下面提供的信息回答上面用户提出的问题：
{query}
"""
prompt = ChatPromptTemplate.from_messages(
    [HumanMessagePromptTemplate.from_template(template)]
)

# 定义state
class State(TypedDict):
    # 状态变量messages类型是list，更新方式是add_messages
    # add_messages是内置的一个办法，将新的消息追加在原列表后面
    messages: Annotated[list, add_messages]

# 定义检索节点
def retrieval(state: State):
    user_query = ''
    if len(state["messages"]) > 0:
        # 获取最后一轮用户输入
        user_query = state["messages"][-1]
    else:
        return {"messages": []}

    # 检索
    docs = retriever.invoke(str(user_query))

    # 填prompt模板
    messages = prompt.invoke("\n".join([doc.page_content for doc in docs])).messages
    return {"messages": messages}

# 定义一个聊天机器人
# 输入是State，输出是系统回复
def chatbot(state: State):
    # 调用大模型，并返回消息（列表）
    return {"messages": [llm.invoke(state["messages"])]}

# 创建状态机
graph_builder = StateGraph(State)

# 添加节点和边
graph_builder.add_node("retrieval", retrieval)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "retrieval")
graph_builder.add_edge("retrieval", "chatbot")
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

