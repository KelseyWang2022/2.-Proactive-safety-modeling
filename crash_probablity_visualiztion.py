import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 读取数据
ttc_data = pd.read_csv("TTC_results.csv")
lttb_data = pd.read_csv("merged_filtered_data(april_may_june).csv")

# 合并数据
data = pd.concat([
    ttc_data[['LTTB', 'crash']],
    lttb_data[['LTTB', 'crash']]
], ignore_index=True)

# 删除缺失值
data = data.dropna()

# 特征和目标
X = data[['LTTB']]
y = data['crash']

# 拆分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42)
rf.fit(X_train, y_train)

# 在完整数据上预测概率
lttb_range = np.linspace(0, 2, 200).reshape(-1, 1)
prob_rf = rf.predict_proba(lttb_range)[:, 1]

# 可视化（仅保留原始结果）
plt.figure(figsize=(10, 6))
plt.plot(lttb_range, prob_rf, label='P(Crash | LTTB)', color='blue', linewidth=2)
plt.xlabel("LTTB")
plt.ylabel("P(Crash | LTTB)")
plt.title("Accident Probability Curve Based on LTTB ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
