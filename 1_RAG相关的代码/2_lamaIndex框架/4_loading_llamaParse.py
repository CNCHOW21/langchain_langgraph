# @Time    : 2025/8/5 20:30
# @Author  : liuzhou
# @File    : 5_llamaParse.py
# @software: PyCharm

"""
通过llamaParse解析复杂pdf文件
"""
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader

load_dotenv()

parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf":parser}

documents = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"], file_extractor=file_extractor).load_data()

print(documents[0].text)