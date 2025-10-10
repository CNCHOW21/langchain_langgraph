# 这是一个示例 Python 脚本。
import json
import requests


# 按 Alt+Shift+X 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+Shift+B 切换断点。


def create_city_image(city: str, weather: str) -> str:
    """这是一个根据城市，天气生成图片的工具，可以用来生成城市的天气图片。
            """
    params = {}
    params['city'] = city
    params['weather'] = weather
    url = 'http://localhost/v1/chat-messages'
    headers = {
        'Authorization': f'Bearer app-TgLrmlIztqRaM7vGDSExAwFH',  # 替换 {your_api_key} 为你实际的 API 密钥
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": params,
        "query": "开始",
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "abc-123",
        "files": [
            {
                "type": "image",
                "transfer_method": "remote_url",
                "url": "https://cloud.dify.ai/logo/logo-site.png"
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = ""
    if response.status_code == 200:
        data = response.text
        for line in data.split('data: '):
            if not line.strip():
                continue
            if not line.strip().endswith("}"):
                continue
            record = json.loads(line)
            # 查找 event 等于 workflow_finished 的记录
            if record['event'] == 'workflow_finished':
                outputs = record['data']['outputs']
                print("Found the required 'outputs':", outputs)
                result = outputs
                break
    return result


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # print_hi('PyCharm')
    create_city_image('武汉', '大风')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助


# 1. 先跑通一个简单例子
import requests
# response = requests.get('https://api.github.com')
# print(response.json())

# 2. 逐步探索功能
# help(requests)  # 查看帮助
# dir(requests)   # 查看可用方法

