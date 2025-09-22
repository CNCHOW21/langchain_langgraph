# @Time    : 2025/8/14 00:16
# @Author  : liuzhou
# @File    : test.py
# @software: PyCharm
import json

import requests
from langchain_core.messages import AIMessage

message = AIMessage(content="123")
# print(message)
# print(message.model_dump())

ai_message = message.model_dump()
if hasattr(ai_message, "content"):
    print(ai_message.content)


def query_huangli(year: str, month: str, day: str) -> str:
    # 83af7821f4db4b68a990056b89e6da8a
    url = 'http://gwgp-gjjifhqt3mv.n.bdcloudapi.com/huangli/date'
    params = {}
    params['year'] = year
    params['month'] = month
    params['day'] = day

    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'X-Bce-Signature': 'AppCode/83af7821f4db4b68a990056b89e6da8a'
    }
    r = requests.request("GET", url, params=params, headers=headers)
    print(r.content.decode('utf-8'))

if __name__ == '__main__':
    query_huangli('1989','11','23')