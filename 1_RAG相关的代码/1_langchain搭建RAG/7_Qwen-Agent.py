# @Time    : 2025/8/8 01:17
# @Author  : liuzhou
# @File    : Qwen-Agent.py
# @software: PyCharm
import os

from dotenv import load_dotenv
from qwen_agent.agents import Assistant

load_dotenv()

def get_file_list(folder_path):
    # 初始化文件列表
    file_list = []

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 将文件路径添加到列表
            file_list.append(file_path)
    return file_list

file_list = get_file_list("./data")

# 步骤 1：配置您所使用的 LLM
llm_config = {
    "model": 'qwen-max',
    'model_server': 'dashscope',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
    # 'model': 'Qwen2-7B-Chat',
    # 'model_server': 'http://localhost:8000/v1',  # base_url，也称为 api_base
    # 'api_key': 'EMPTY',
    # （可选） LLM 的超参数：
    'generate_cfg': {
        'top_p': 0.8
    }
}

# 步骤2：创建一个智能体，这里我们以'Assistant'智能体为例，它能够使用工具并读取文件
system_instruction = '你是一位HR，根据你的经验来准确回答用户提出的问题'

tools = [] # `code_interpreter` 是框架自带的工具，用于执行代码。

bot = Assistant(llm=llm_config,
                system_message=system_instruction,
                function_list=tools,
                files=file_list)

# 步骤3：作为聊天机器人运行智能体
messages = [] # 这里存储聊天历史
while True:
    query = input("\nQ：")
    if query == 'q':
        break
    # 将用户请求添加到聊天历史
    messages.append({'role':'user','content':query})
    response = []
    current_index = 0
    for response in bot.run(messages=messages):
        # 流式输出
        current_response = response[0]['content'][current_index:]
        current_index = len(response[0]['content'])
        print(current_response, end='')
    # 将机器人的回应添加到聊天历史
    messages.extend(response)

