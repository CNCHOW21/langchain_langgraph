from openai import OpenAI

# 连接本地 vLLM
client = OpenAI(
    base_url="http://192.168.1.200:8000/v1",
    api_key="none"  # vLLM 不需要真实密钥
)

# 对话历史，用于多轮上下文
messages = []

print("多轮流式对话已启动，输入 exit 退出")
print("-" * 40)

while True:
    user_input = input("用户：")

    # 退出条件
    if user_input.strip().lower() == "exit":
        print("对话结束")
        break

    # 把用户问题加入历史
    messages.append({"role": "user", "content": user_input})

    # 调用 vLLM 流式接口
    try:
        stream = client.chat.completions.create(
            model="/storage/models/Qwen/Qwen2___5-7B-Instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True  # 开启流式输出
        )

        print("AI：", end="", flush=True)
        full_answer = ""  # 存储完整回答，存入历史

        # 逐块读取流数据
        for chunk in stream:
            # 过滤空分片
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 存在增量文字则实时打印
            if delta.content:
                print(delta.content, end="", flush=True)
                full_answer += delta.content

        # 换行分隔下一轮输入
        print()
        # 将完整回答存入对话上下文，实现多轮记忆
        messages.append({"role": "assistant", "content": full_answer})

    except Exception as e:
        print(f"\n请求出错：{e}")
        print("请确认 vLLM 服务正常运行在 http://192.168.1.200:8000")