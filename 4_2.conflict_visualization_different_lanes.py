#对相邻车道的碰撞风险和
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取最新上传的数据
df = pd.read_csv("ttc_lttb_final_split_by_type_yaw.csv")

# 设置绘图风格
sns.set(style="whitegrid")

# 创建图形区域
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

# 第一张图：TTC 按类型区分
sns.histplot(data=df, x="TTC", hue="type", bins=50, kde=True, ax=axes[0], palette="Set2", multiple="stack")
axes[0].set_title("Distribution of TTC (Time To Collision) by Type")
axes[0].set_xlabel("TTC (seconds)")
axes[0].set_ylabel("Frequency")

# 第二张图：LTTB 按类型区分
sns.histplot(data=df, x="LTTB", hue="type", bins=50, kde=True, ax=axes[1], palette="Set2", multiple="stack")
axes[1].set_title("Distribution of LTTB (Latest Time To Brake) by Type")
axes[1].set_xlabel("LTTB (seconds)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
