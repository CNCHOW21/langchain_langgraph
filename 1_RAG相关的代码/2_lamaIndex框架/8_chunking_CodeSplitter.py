# @Time    : 2025/8/5 21:15
# @Author  : liuzhou
# @File    : 6_textsplitter.py
# @software: PyCharm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter
from convert_json import show_json

# (llamaindex-new) C:\Users\Administrator>pip install tree_sitter
# (llamaindex-new) C:\Users\Administrator>pip install tree_sitter_language_pack
documents = SimpleDirectoryReader("./data",required_exts=[".py"]).load_data()

# CodeSplitter 根据 AST（编译器的抽象句法树）切分代码，保证代码功能片段完整；
node_parse = CodeSplitter(
    language="python",
    chunk_lines=2,
    chunk_lines_overlap=1
)

nodes = node_parse.get_nodes_from_documents(
    documents, show_progress=True
)

show_json(nodes[0].json())