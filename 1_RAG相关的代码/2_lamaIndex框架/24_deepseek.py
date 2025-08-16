# @Time    : 2025/8/6 17:18
# @Author  : liuzhou
# @File    : 24_deepseek.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from llama_index.llms.deepseek import DeepSeek

load_dotenv()

llm = DeepSeek(model="deepseek-chat",api_key=os.getenv("DEEPSEEK_API_KEY"))
response = llm.complete("每天通勤多长时间最合适？")
print(response)

