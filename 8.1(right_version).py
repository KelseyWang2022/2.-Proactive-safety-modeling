import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 读取数据
crash_df = pd.read_csv("TTC_results.csv")
non_crash_df = pd.read_csv("merged_filtered_data(april_may_june).csv")

# 添加标签
crash_df["crash"] = 1
non_crash_df["crash"] = 0

# 合并数据
combined_df = pd.concat([crash_df, non_crash_df], ignore_index=True)
combined_df = combined_df[combined_df["LTTB"].notna()]  # 去除缺失值

# EVT 拟合：仅使用非事故数据
threshold_evt = 2.0
excesses = threshold_evt - non_crash_df["LTTB"]
excesses = excesses[excesses > 0]
c_evt, loc_evt, scale_evt = genpareto.fit(excesses)

# 训练随机森林模型
X = combined_df[["LTTB"]]
y = combined_df["crash"]
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 定义阈值范围
thresholds = np.linspace(0.5, 2.5, 50)

# 计算融合概率 + 置信区间
fused_mean, lower_ci, upper_ci = [], [], []
n_bootstrap = 300

for t in thresholds:
    probs = []
    for _ in range(n_bootstrap):
        boot_df = combined_df.sample(frac=1, replace=True)
        # RF 条件概率
        rf_prob = rf.predict_proba([[t]])[0][1]
        # EVT 边缘概率
        p_evt = genpareto.cdf(threshold_evt - t, c_evt, loc_evt, scale_evt) if t < threshold_evt else 1.0
        # 贝叶斯融合
        prob = rf_prob * p_evt
        probs.append(prob)
    fused_mean.append(np.mean(probs))
    lower_ci.append(np.percentile(probs, 2.5))
    upper_ci.append(np.percentile(probs, 97.5))

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(thresholds, fused_mean, label="Bayesian Fused Crash Risk (LTTB)", color="darkred")
plt.fill_between(thresholds, lower_ci, upper_ci, color="salmon", alpha=0.3)
plt.xlabel("LTTB Threshold (s)")
plt.ylabel("Crash Risk Probability")
plt.title("Fused LTTB Crash Risk Curve with 95% Confidence Interval\n(Bayesian Fusion of EVT and RF)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
