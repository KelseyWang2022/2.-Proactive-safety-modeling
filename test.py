import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# 读取数据
ttc_data = pd.read_csv("TTC_results.csv")
lttb_data = pd.read_csv("merged_filtered_data(april_may_june).csv")

# 合并并清洗数据
data = pd.concat([
    ttc_data[['LTTB', 'crash']],
    lttb_data[['LTTB', 'crash']]
], ignore_index=True)
data = data.dropna()

# 特征与标签
X = data[['LTTB']]
y = data['crash']

# 划分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 拟合随机森林
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42)
rf.fit(X_train, y_train)

# 生成LTTB范围
lttb_range = np.linspace(0, 2, 200).reshape(-1, 1)
preds = []

# Bootstrap生成置信区间
n_bootstrap = 100
for _ in range(n_bootstrap):
    X_bs, y_bs = resample(X_train, y_train, replace=True, random_state=None)
    rf_bs = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42)
    rf_bs.fit(X_bs, y_bs)
    preds.append(rf_bs.predict_proba(lttb_range)[:, 1])

preds = np.array(preds)
mean_preds = preds.mean(axis=0)
lower_bound = np.percentile(preds, 2.5, axis=0)
upper_bound = np.percentile(preds, 97.5, axis=0)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(lttb_range, mean_preds, label='Mean P(Crash | LTTB)', color='blue')
plt.fill_between(lttb_range.ravel(), lower_bound, upper_bound, alpha=0.3, color='blue', label='95% CI')
plt.xlabel("LTTB")
plt.ylabel("P(Crash | LTTB)")
plt.title("Collision Probability Curve Based on LTTB with 95% CI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
