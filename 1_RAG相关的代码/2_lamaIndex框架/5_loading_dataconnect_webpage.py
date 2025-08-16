# @Time    : 2025/8/5 20:42
# @Author  : liuzhou
# @File    : 6_load_webpage.py
# @software: PyCharm
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://edu.guangjuke.com/tx/"]
)

print(documents[0].text)