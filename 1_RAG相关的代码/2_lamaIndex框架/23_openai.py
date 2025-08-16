# @Time    : 2025/8/6 17:08
# @Author  : liuzhou
# @File    : 23_openai.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

load_dotenv()
"""
调用openAI的模型
"""
llm = OpenAI(
    temperature=0,
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"))

prompt = PromptTemplate("讲一个关于{name}的笑话")
prompt_str = prompt.format(name="小明")
response = llm.complete(prompt_str)
print(response)