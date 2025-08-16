# @Time    : 2025/8/10 19:40
# @Author  : liuzhou
# @File    : 8_inmemorystore.py
# @software: PyCharm
import uuid

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()

in_memory_store = InMemoryStore(
    index = {
        "embed":DashScopeEmbeddings(model = "text-embedding-v2"),
        "dims":1536
    }
)


model = init_chat_model("deepseek-chat", model_provider="deepseek")

# langGraph消息的共享是通过线程实现的
# config = {"configurable":{"thread_id": "userid_654643511"}}

def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Liuzhou"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}

builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=MemorySaver(), store=in_memory_store) # 对话记录存起来，存到memory_store

config = {"configurable":{"thread_id": "1","user_id": "1"}}
input_message = {"role":"user", "content": "hi! Remember: my name is Liuzhou"}

for chunk in graph.stream({"messages":[input_message]}, config, stream_mode="values"):
    chunk = chunk["messages"][-1].pretty_print()

# config = {"configurable":{"thread_id": "1","user_id": "1"}}
# input_message = {"role":"user", "content": "what's my name?"}
# for chunk in graph.stream({"messages":[input_message]}, config, stream_mode="values"):
#     chunk = chunk["messages"][-1].pretty_print()

# 线程改变，用户id不变
# config = {"configurable":{"thread_id": "2","user_id": "1"}}
# input_message = {"role":"user", "content": "what's my name?"}
# for chunk in graph.stream({"messages":[input_message]}, config, stream_mode="values"):
#     chunk = chunk["messages"][-1].pretty_print()


# 线程改变，用户id改变
config = {"configurable":{"thread_id": "3","user_id": "2"}}
input_message = {"role":"user", "content": "what's my name?"}
for chunk in graph.stream({"messages":[input_message]}, config, stream_mode="values"):
    chunk = chunk["messages"][-1].pretty_print()

