# @Time    : 2025/8/7 21:59
# @Author  : liuzhou
# @File    : 6_Qwen-Agent.py
# @software: PyCharm
import io
import logging
import os
from dotenv import load_dotenv
from qwen_agent.agents import Assistant

load_dotenv()
# 创建一个自定义的日志处理器来捕获日志输出
class LogCapture:
    def __init__(self):
        self.log_capture_string = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture_string)
        self.log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(self.log_formatter)

        # 获取qwen-agent 的日志记录器
        self.logger = logging.getLogger("qwen_agent_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)

        # 也可以捕获根日志记录器的输出
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        self.root_logger.addHandler(self.log_handler)

    def get_log(self):
        return self.log_capture_string.getvalue()

    def clear_log(self):
        self.log_capture_string.truncate(0)
        self.log_capture_string.seek(0)

# 初始化日志捕获器
log_capture = LogCapture()

# 清除之前的日志
log_capture.clear_log()

# 步骤1：配置LLM
llm_cfg = {
    "model":"qwen-max",
    "model_server":"dashscope",
    "apd_key":os.getenv("DASHSCOPE_API_KEY"),
    "generate_cfg":{
        "top_p":0.8
    }
}

# 步骤2：创建一个智能体，这里我们以‘Assistant’智能体为例，它能够使用工具并读取文件
system_instruction = ''
tools = []
files = ['./data/浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf']  # 给智能体一个 PDF 文件阅读。

# 创建智能体
bot = Assistant(
    llm=llm_cfg,
    system_message=system_instruction,
    function_list=tools,
    files=files
)

# 步骤 3：作为聊天机器人运行智能体。
messages = []
query = "客户经理被客户投诉一次，扣多少分？"
# 将用户请求添加到聊天历史
messages.append({'role':'user', 'content':query})
response = []
current_index = 0

# 运行智能体
for response in bot.run(messages=messages):
    # 在第一次响应时，分析日志以查找召回的文档内容
    if current_index == 0:
        # 获取日志
        log_content = log_capture.get_log()

        print("\n============从日志提取的检索信息============")
        # 查找与检索相关的日志行
        retrieve_logs = [line for line in log_content.split('\n')
                         if any(keyword in line.lower() for keyword in
                               ['retriev', 'search', 'chunk', 'document', 'ref', 'token'])]

        # 打印检索相关的日志
        for log_line in retrieve_logs:
            print(log_line)

        # 尝试从日志中提取文档内容
        # 通常在日志中会有类似 "retrieved document: ..." 或 "content: ..." 的行
        content_logs = [line for line in log_content.split('\n')
                        if any(keyword in line.lower() for keyword in
                               ['content', 'text', 'document', 'chunk'])]

        print("\n==============可能包含文档内容的日志===========")
        for content_log in content_logs:
            print(content_log)

        print("=======================================\n")

    current_response = response[0]['content'][current_index:]
    current_index = len(response[0]['content'])
    print(current_response, end='')

# 将机器人的回应添加到聊天历史。
messages.extend(response)

# 运行结束后，分析完整的日志
print("\n\n===== 运行结束后的完整日志分析 =====")
log_content = log_capture.get_log()

# 尝试从日志中提取更多信息
print("\n1. 关键词提取:")
keyword_logs = [line for line in log_content.split('\n') if 'keywords' in line.lower()]
for log_line in keyword_logs:
    print(log_line)

print("\n2. 文档处理:")
doc_logs = [line for line in log_content.split('\n') if 'doc' in line.lower() or 'chunk' in line.lower()]
for log_line in doc_logs:
    print(log_line)

print("\n3. 检索相关:")
retrieval_logs = [line for line in log_content.split('\n') if 'retriev' in line.lower() or 'search' in line.lower() or 'ref' in line.lower()]
for log_line in retrieval_logs:
    print(log_line)

print("\n4. 可能包含文档内容的日志:")
content_logs = [line for line in log_content.split('\n') if 'content:' in line.lower() or 'text:' in line.lower()]
for log_line in content_logs:
    print(log_line)

print("===========================\n")