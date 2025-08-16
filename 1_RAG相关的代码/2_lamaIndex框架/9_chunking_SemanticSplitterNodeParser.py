# @Time    : 2025/8/5 21:15
# @Author  : liuzhou
# @File    : 6_textsplitter.py
# @software: PyCharm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels, DashScopeEmbedding

from convert_json import show_json

documents = SimpleDirectoryReader("./data",required_exts=[".pdf"]).load_data()

# SemanticSplitterNodeParser：根据语义相关性对将文本切分为片段。
node_parse = SemanticSplitterNodeParser(
    embed_model= DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2),
)

nodes = node_parse.get_nodes_from_documents(
    documents, show_progress=True
)
print(f"==========划分的节点数量为：{len(nodes)}")
show_json(nodes[0].json())
print("="*100)
show_json(nodes[1].json())