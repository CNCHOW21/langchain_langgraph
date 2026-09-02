from datasets import load_dataset

news_dataset = load_dataset(path="csv", data_files="data/news/train.csv",split="train")

print(news_dataset[0]["text"])

