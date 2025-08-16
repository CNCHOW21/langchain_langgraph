# @Time    : 2025/8/14 00:16
# @Author  : liuzhou
# @File    : test.py
# @software: PyCharm
import json

from langchain_core.messages import AIMessage

message = AIMessage(content="123")
# print(message)
# print(message.model_dump())

ai_message = message.model_dump()
if hasattr(ai_message, "content"):
    print(ai_message.content)