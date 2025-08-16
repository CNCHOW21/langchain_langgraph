# @Time    : 2025/8/5 22:25
# @Author  : liuzhou
# @File    : 10_chunking_HTMLNodeParser.py
# @software: PyCharm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HTMLNodeParser, MarkdownNodeParser, TokenTextSplitter, JSONNodeParser
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleDirectoryReader("./data", required_exts=[".json"]).load_data()

parser = JSONNodeParser(include_metadata=True,include_prev_next_rel=True)

nodes = parser.get_nodes_from_documents(documents)

print(f"========================{len(nodes)}")
for node in nodes:
    print(node.text)
    print("="*100)