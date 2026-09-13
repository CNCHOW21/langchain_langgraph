import requests
import json

url = "http://192.168.1.200:8001/v1/embeddings"

payload = {
    "input": "RAG是什么",
    "model": "bge-m3"
}
headers = {"Content-Type": "application/json"}

resp = requests.post(url, data=json.dumps(payload), headers=headers)
result = resp.json()

# 打印向量长度
embedding = result["data"][0]["embedding"]
print(f"向量长度：{len(embedding)}")
print("向量前10位：", embedding[:10])
