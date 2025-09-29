# @Time    : 2025/9/26 14:03
# @Author  : liuzhou
# @File    : model_backup.py
# @software: PyCharm

from langchain_openai import ChatOpenAI

# 主模型 + 备用模型
reliable_chat = ChatOpenAI(model="gpt-4").with_fallbacks([
    ChatOpenAI(model="gpt-3.5-turbo"),
    ChatOpenAI(model="gpt-3.5-turbo-16k")
])

# 当gpt-4不可用时自动回退
response = reliable_chat.invoke("请解释量子力学的基本原理")
