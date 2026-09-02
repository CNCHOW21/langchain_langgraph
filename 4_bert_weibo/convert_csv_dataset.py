# 将本地的arrow文件转换成csv文件
from datasets import Dataset

train_dataset = Dataset.from_file(r"G:\develop\pycharm-project\huggingface_my_weibo_csv\data\Weibo\weibo-train.arrow")
test_dataset = Dataset.from_file(r"G:\develop\pycharm-project\huggingface_my_weibo_csv\data\Weibo\weibo-test.arrow")
val_dataset = Dataset.from_file(r"G:\develop\pycharm-project\huggingface_my_weibo_csv\data\Weibo\weibo-validation.arrow")

train_dataset.to_pandas().to_csv("csv_data/Weibo/weibo-train.csv", index=False)
test_dataset.to_pandas().to_csv("csv_data/Weibo/weibo-test.csv", index=False)
val_dataset.to_pandas().to_csv("csv_data/Weibo/weibo-validation.csv", index=False)