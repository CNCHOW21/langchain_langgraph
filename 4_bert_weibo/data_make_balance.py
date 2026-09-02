#手动重采样，保持数据均衡
# pip install pandas imbalanced-learn
import  pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler # 随机采样

# 读取CSV文件
csv_file_path = "csv_data/Weibo/weibo-train.csv"
csv_file_path = "csv_data/Weibo/weibo-test.csv"
csv_file_path = "csv_data/Weibo/weibo-validation.csv"
df = pd.read_csv(csv_file_path)

#定义重采样策略
# 如果想要过采样，使用RandomOverSampler , 把少的变成多的，造出来的数据不符合实际情况，推荐使用欠采样
# 如果想要欠采样，使用RandomUnderSampler, 把多的变成少的
# random_state控制随机数生成器的种子，一般给42
rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)

#将特征和标签分开
x = df[["text"]] # 维度，二维
y = df[["label"]]

#应用重采样
x_resampled,y_resampled = rus.fit_resample(x, y)
#合并特征和标签，创建系的DataFrame
df_resampled = pd.concat([x_resampled, y_resampled], axis=1) # axis轴为1

#保存均衡数据到新的csv文件中
# df_resampled.to_csv("weibo-train-new.csv", index=False)
# df_resampled.to_csv("weibo-test-new.csv", index=False)
df_resampled.to_csv("weibo-validation-new.csv", index=False)