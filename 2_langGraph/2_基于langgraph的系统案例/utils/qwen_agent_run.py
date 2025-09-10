# @Time    : 2025/8/31 14:48
# @Author  : liuzhou
# @File    : qwen-agent.py
# @software: PyCharm

import os

from dotenv import load_dotenv
from qwen_agent.agents import Assistant
from .llms import MODEL_CONFIGS

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

config = MODEL_CONFIGS['qwen']

# 步骤 1：配置您所使用的 LLM
llm_config = {
    "model": config['chat_model'],
    'model_server': config['base_url'],
    'api_key': config['api_key'],
    "timeout": 30,  # 添加超时配置（秒）
     "max_retries":2,  # 添加重试次数
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
system_instruction = '你是一位资料管理专家，请严格按照名字进行查询，如果没有这个名字的信息，直接回复没有相关信息。'

tools = [] # `code_interpreter` 是框架自带的工具，用于执行代码。

bot = Assistant(llm=llm_config,
                system_message=system_instruction,
                function_list=tools,
                files=file_list)

def run_agent(query: str)-> str:
    messages = [] # 这里存储聊天历史
    response = ''
    messages.append({'role':'user','content':query})
    for response in bot.run(messages=messages):
        response = response[0]['content']
        # 将机器人的回应添加到聊天历史
        # messages.extend(response)
    return response

# if __name__ == '__main__':
#     print(run_agent("刘舟的工资多少？"))