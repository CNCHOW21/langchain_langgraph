# @Time    : 2025/8/6 16:45
# @Author  : liuzhou
# @File    : prompt_PromptTemplate.py
# @software: PyCharm
from llama_index.core import PromptTemplate

prompt= PromptTemplate("写一个关于{name}的笑话")

prompt_str = prompt.format(name="liuzhou")

print(prompt_str)