# @Time    : 2025/8/5 20:03
# @Author  : liuzhou
# @File    : 3_read_file.py
# @software: PyCharm
from llama_index.core import SimpleDirectoryReader

"""
解析特定格式的的文件
"""

import json
from pydantic.v1 import BaseModel

# 转换json格式
def show_json(data):
    """用于展示json数据"""
    if isinstance(data, str):
        obj = json.loads(data)
        print(json.dumps(obj, indent=4, ensure_ascii=False))
    elif isinstance(data, dict) or isinstance(data, list):
        print(json.dumps(data, indent=4, ensure_ascii=False))
    elif issubclass(type(data), BaseModel):
        print(json.dumps(data.dict(), indent=4, ensure_ascii=False))

def show_list_obj(data):
    """用于展示一组对象"""
    if isinstance(data, list):
        for item in data:
            show_json(item)
    else:
        raise ValueError("Input is not a list")


# 读取pdf文档
reader = SimpleDirectoryReader(
            input_dir="./data",
            recursive=False,
            required_exts=[".pdf"]
        )
documents = reader.load_data()

# print(documents[0].text)
show_json(documents[0].json())