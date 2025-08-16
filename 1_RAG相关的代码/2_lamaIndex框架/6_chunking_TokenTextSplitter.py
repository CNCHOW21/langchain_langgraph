# @Time    : 2025/8/5 21:15
# @Author  : liuzhou
# @File    : 6_textsplitter.py
# @software: PyCharm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from convert_json import show_json

documents = SimpleDirectoryReader("./data").load_data()

# TokenTextSplitter 按指定 token 数切分文本
# 精准适配 LLM 限制，直接基于 token 计数，避免因 token 计算规则（如空格、标点的处理）导致的长度误判。
node_parse = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=128
)

nodes = node_parse.get_nodes_from_documents(
    documents, show_progress=True
)

show_json(nodes[0].json())