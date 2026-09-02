import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from Mydata import MyDataset
from net import Model

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次
EPOCH= 30000

token = BertTokenizer.from_pretrained(r"G:\develop\pycharm-project\huggingface_my_weibo_csv\model\google-bert\bert-base-chinese\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    sents = [i[0] for i in data] # 文本的内容
    label = [i[1] for i in data] # 标志，0或者1
    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 需要分词的文本
        truncation=True, # 当文本长度超过 max_length 时自动截断，避免溢出 BERT 模型最长 512 的限制（适用于中文长文本）
        max_length=1024, # 统一将文本分词后长度固定为 512 个 token（BERT 模型的输入上限）
        padding='max_length', # 若文本分词后长度不足 512，用[PAD]填充至该长度
        return_tensors='pt',  # 返回 PyTorch 张量格式（如 torch.Tensor）
        return_length=True, # 在返回结果中包含原始分词长度（不含填充部分）
    )
    input_ids = data['input_ids'] # 二维张量，形状为 (batch_size, 512)，每个句子被转换成的词表索引值（如 [101, 2345, 103, ...]）
    attention_mask = data['attention_mask'] # 与 input_ids 同形状的 0/1 张量，标识实际内容为 1，填充部分为 0
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, labels

# 创建数据集
train_dataset = MyDataset("train")
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=10,# *batch_size一次性取多少数据，取决于GPU的大小
                          shuffle=True, # shuffle=True把数据集打乱
                          drop_last=True, #drop_last=True舍弃最后一个批次的数据，防止形状出错
                          collate_fn=collate_fn
                          )

# 创建验证数据集
val_dataset = MyDataset("validation")
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=10,# *batch_size一次性取多少数据，取决于GPU的大小，批次至少大于1
                          shuffle=True, # shuffle=True把数据集打乱
                          drop_last=True, #drop_last=True舍弃最后一个批次的数据，防止形状出错
                          collate_fn=collate_fn
                          )

if __name__ == '__main__':
    # 开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    # 定义优化器
    optimizer = AdamW(model.parameters())
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    #初始化最佳验证准确率
    best_val_acc = 0.0

    #加载训练参数，从第第二轮开始训练
    # model.load_state_dict(torch.load('params/2_bert.pth'))
    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 将数据存放到DEVICE上
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            # 前向计算（将数据输入模型，得到输出）
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # 根据输出，计算损失
            loss = loss_func(out, labels)
            # 根据损失，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #每隔5个批次输出训练信息
            if i%5==0:
                out = out.argmax(dim=1)
                acc = (out==labels).sum().item()/len(labels) #精度
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")

        # 验证模型（判断是否过拟合）
        # 设置为评估模式
        model.eval()
        # 不需要模型参与训练
        with torch.no_grad():
            val_acc = 0.0
            val_loss = 0.0
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
                # 将数据存放到DEVICE上
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                    DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
                # 前向计算（将数据输入模型，得到输出）
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # 根据输出，计算损失
                val_loss += loss_func(out, labels)
                out = out.argmax(dim=1)
                val_acc+=(out==labels).sum().item()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"验证集：loss:{val_loss}, acc:{val_acc}")

        # 每训练完一轮，保存一次参数
        # torch.save(model.state_dict(), f'params/{epoch}_bert.pth')
        # print(epoch, "参数保存成功！")

        # 根据验证准确率保存最优参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "trained_result/best_model.pth")
            print(f"Epoch:{epoch}:保存最优参数：acc:{best_val_acc}")

        # 保存最后一轮的参数
        torch.save(model.state_dict(), "trained_result/model.pth")
        print(epoch,f"Epoch:{epoch}最后一轮参数保存成功！")

