# @Time    : 2025/8/6 16:56
# @Author  : liuzhou
# @File    : 22_prompt_ChatPromptTemplate.py
# @software: PyCharm
from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import MessageRole, ChatMessage
"""
ChatPromptTemplate定义多轮模板
"""
chat_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="你叫{name}，必须按照用户提供的上下文回答问题。",
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "已知上下文：\n{content}\n问题：{question}"),
        )
]

text_qa_template = ChatPromptTemplate(chat_qa_msgs)

prompt_str = text_qa_template.format(name="刘舟", content="这是一个测试", question="这是什么")

print(prompt_str)