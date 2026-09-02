# 统计各个类别的比例，查看各个类别的比例是否均衡
import pandas as pd
from sympy.integrals.meijerint_doc import category

# df = pd.read_csv("csv_data/Weibo/weibo-train.csv")
# df = pd.read_csv("weibo-train-new.csv")
df = pd.read_csv("data/news/train.csv")
#统计每个类别的数据量
category_count = df['label'].value_counts()
print(category_count)

#统计的各个类别数据量分布不均衡，数据的质量太差，训练的效果会很差
#1.补全数据
#2.舍弃比例过大的数据

#统计每个类别的比值
total_data = len(df)
category_ratio = (category_count / total_data)*100
print(category_ratio)

# 可以将比例控制在1.5之间，比如将7的数据截取到比例为1.5