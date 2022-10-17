from re import T
import sys
import pandas as pd


ratio = float(sys.argv[1])
df = pd.read_csv("haplo.csv", header=None)
df.sample(frac=1).reset_index(drop=True)
print(df)
bound = int(df.shape[0] * ratio)
print(f"样本数：{df.shape[0]}")
new_train = df[0: bound]
new_test = df[bound:]
print(f"训练集样本数：{new_train.shape[0]}")
print(f"测试集样本数：{new_test.shape[0]}")
new_train.to_csv("hap_train.csv", index=None, header=None)
new_test.to_csv("hap_test.csv", index=None, header=None)
