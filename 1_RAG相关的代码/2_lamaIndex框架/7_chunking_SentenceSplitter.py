# @Time    : 2025/8/5 21:15
# @Author  : liuzhou
# @File    : 6_textsplitter.py
# @software: PyCharm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from convert_json import show_json

documents = SimpleDirectoryReader("./data").load_data()

# SentenceTextSplitter 按句子的完整度进行切分
# 在切分指定长度的 chunk 同时尽量保证句子边界不被切断；
node_parse = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=128
)

nodes = node_parse.get_nodes_from_documents(
    documents, show_progress=True
)

show_json(nodes[0].json())