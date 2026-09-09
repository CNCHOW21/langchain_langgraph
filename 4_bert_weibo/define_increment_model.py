# 基于 BERT 的中文文本 8 分类模型实现。使用预训练 BERT 提取语义特征，通过线性层进行分类。
import torch
from torch import nn
from transformers import BertModel

# 定义设备信息
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#加载模型
pre_model = BertModel.from_pretrained(r'E:\1TB硬盘备份\program\pycharm-project\liuzzzzzz\4_bert_weibo\model\google-bert\bert-base-chinese\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f').to(device)
#input_ids将文本转换为词表索引序列，每个token被映射为整数ID（如 [CLS]=101，单词"你好"=7592等）4
#attention_mask标识哪些token是真实数据（需参与计算），哪些是填充的无效数据（需被忽略）
#token_type_ids在包含多个文本序列的输入中，区分不同序列的归属（如句子对任务中的第一句和第二句）
class InceamentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 8)  # 线性分类层：nn.Linear(768,8)，将 BERT 输出转换为类别概率，8分类
    def forward(self, input_ids, attention_mask, token_type_ids):#forward 的参数有 input_ids、attention_mask 和 token_type_ids，这些都是 BERT 的标准输入
        with torch.no_grad():#冻结 BERT 参数：通过 torch.no_grad() 禁止梯度回传，仅训练线性层 [迁移学习典型方法]
            out = pre_model(input_ids, attention_mask, token_type_ids)
        # 增量模型参与训练
        out = self.fc(out.last_hidden_state[:,0]) #解冻分类层参数（默认已启用梯度），last_hidden_state[:,0]表示 从模型最后一个隐藏层中提取所有样本的CLS标记（Classification Token）对应的嵌入向量
        return out