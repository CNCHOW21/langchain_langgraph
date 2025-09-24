# @Time    : 2025/9/24 16:13
# @Author  : liuzhou
# @File    : store_test.py
# @software: PyCharm

import psycopg
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

# pool = psycopg.Pool("postgresql://liuzhou:liuzhou@localhost:5432/postgres")
# store = PostgresStore(conn=pool, index_config={"index_type": "hnsw"})
llm_embedding = OpenAIEmbeddings(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-8135aedf8bf34fbc9ef1d2894a30c6e1",
    model="text-embedding-v1",
    deployment="text-embedding-v1"
)

# 测试嵌入一个文本
try:
    res = llm_embedding.embed_query("Alice is 30 years old.")
    print("Success! Embedding length:", len(res))  # 应该输出 1536
except Exception as e:
    print("❌ Embedding failed:", str(e))

connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
# 创建数据库连接池，最大连接数20,最小保持2个活跃连接,从池中获取连接的最大等待时间10秒
db_connection_pool = ConnectionPool(conninfo="postgresql://liuzhou:liuzhou@localhost:5432/postgres", max_size=20, min_size=2, kwargs=connection_kwargs,
                                    timeout=10)
store = PostgresStore(db_connection_pool, index={"dims": 1536, "embed": llm_embedding})

store.setup()

if __name__ == '__main__':
    value_str = "name: Alice, age: 30"  # 或 json.dumps(...)
    store.put(
        namespace=("users", "user1"),
        key="profile",
        value=value_str  # ← 必须是 str 或 list[str]
    )

