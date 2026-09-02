from transformers import BertModel,BertConfig
import torch

# 获取设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载预训练模型
# train_model = BertModel.from_pretrained(r'G:\develop\pycharm-project\huggingface_my_weibo_csv\model\google-bert\bert-base-chinese\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f').to(DEVICE)

# 重置模型输入最大长度，将position_embeddings设置成1024
# train_model.embeddings.position_embeddings = torch.nn.Embedding(1024, 768).to(DEVICE)

# max_position_embeddings修改成1024
config = BertConfig.from_pretrained(r'G:\develop\pycharm-project\huggingface_my_weibo_csv\model\google-bert\bert-base-chinese\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')
config.max_position_embeddings = 1024
print(config)

# 将修改的配置文件加载模型内部，使用配置文件初始化模型
train_model = BertModel(config).to(DEVICE)
print(train_model)

#定义下游任务
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 10)  # 10分类
    def forward(self, input_ids, attention_mask, token_type_ids):
        #冻结预训练的权重，只训练增量的权重
        #max_length参数变了，需要重新训练参数，不能冻结先验了
        # with torch.no_grad():
        # 全量微调
        out = train_model(input_ids, attention_mask, token_type_ids)
        out = self.fc(out.last_hidden_state[:,0])
        return out
