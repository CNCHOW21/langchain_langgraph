# @Time    : 2025/8/5 22:25
# @Author  : liuzhou
# @File    : 10_chunking_HTMLNodeParser.py
# @software: PyCharm
from llama_index.core.node_parser import HTMLNodeParser
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=False).load_data(
    ["https://edu.guangjuke.com/tx/"]
)

# 默认解析 ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "section"]
parser = HTMLNodeParser(tags=["span"]) # 可以自定义解析哪些标签
nodes = parser.get_nodes_from_documents(documents)

print(f"========================{len(nodes)}")
for node in nodes:
    print(node.text)
    print("="*100)