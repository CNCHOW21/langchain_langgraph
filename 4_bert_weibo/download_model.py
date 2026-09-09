# @Time    : 2025/5/26 21:53
# @Author  : liuzhou
# @File    : download_model.py
# @software: PyCharm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google-bert/bert-base-chinese"
cache_dir = "model"
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("下载完成")