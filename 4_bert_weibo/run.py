import torch
from transformers import BertTokenizer

from define_increment_model import InceamentModel

#定义设备信息
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

token = BertTokenizer.from_pretrained(r'E:\1TB硬盘备份\program\pycharm-project\liuzzzzzz\4_bert_weibo\model\google-bert\bert-base-chinese\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')
names = ["喜欢","厌恶","开心","悲伤","愤怒","惊讶","害怕","无"]
model = InceamentModel().to(device)

def collate_fn(data):
    sents = []
    sents.append(data)
    #编码
    data = token.batch_encode_plus(
        sents,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt', #返回pytorch的增量
        return_length=True,
    )
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    return input_ids, attention_mask, token_type_ids


def test():
    #加载训练参数
    model.load_state_dict(torch.load('trained_result/best_model.pth',map_location=device))
    #开启测试模式
    model.eval()

    while True:
        data = input("请输入测试数据(输入‘q’退出)：")
        if data == 'q':
            print("测试结束")
            break
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), \
            token_type_ids.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            out = out.argmax(dim=1)
            print(f"out:{out}")
            print("模型判定：",names[out],"\n")

if __name__ == '__main__':
    test()