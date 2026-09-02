from datasets import load_dataset
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, split):
        # 从磁盘加载数据
        self.news_dataset = load_dataset(path="csv", data_files=f"data/news/{split}.csv",split="train")
    def __len__(self):
        return len(self.news_dataset)
    def __getitem__(self, item):
        text = self.news_dataset[item]["text"]
        label = self.news_dataset[item]["label"]
        return text, label

if __name__ == "__main__":
    dataset = MyDataset("test")
    print(dataset[0])
    # for data in dataset:
    #     print(data)