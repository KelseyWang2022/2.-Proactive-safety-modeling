import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# 读取数据
non_crash_data = pd.read_csv("merged_filtered_data(april_may_june).csv")
crash_data = pd.read_csv("TTC_results.csv")

# 添加标签
non_crash_data["crash"] = 0
crash_data["crash"] = 1

# 合并数据并清理
combined_df = pd.concat([non_crash_data, crash_data], ignore_index=True)
combined_df = combined_df.dropna(subset=["LTTB"])

# 特征与标签
X = combined_df[["LTTB"]]
y = combined_df["crash"]

# 训练模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# 设置 LTTB 阈值区间
thresholds = np.linspace(combined_df["LTTB"].min(), combined_df["LTTB"].max(), 25)

# 储存概率和置信区间
means, lowers, uppers = [], [], []
n_bootstrap = 100  # 更快

# 遍历每个阈值
for t in thresholds:
    subset = combined_df[combined_df["LTTB"] <= t]
    if len(subset) < 10:
        means.append(np.nan)
        lowers.append(np.nan)
        uppers.append(np.nan)
        continue

    boot_preds = []
    for _ in range(n_bootstrap):
        sample = resample(subset)
        proba = rf_model.predict_proba(sample[["LTTB"]])[:, 1]
        boot_preds.append(np.mean(proba))

    mean = np.mean(boot_preds)
    ci_lower, ci_upper = np.percentile(boot_preds, [2.5, 97.5])

    means.append(mean)
    lowers.append(ci_lower)
    uppers.append(ci_upper)

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(thresholds, means, label="Bayesian Fused Crash Risk (LTTB)", color="darkred", linewidth=2)
plt.fill_between(thresholds, lowers, uppers, color="red", alpha=0.2, label="95% Confidence Interval")
plt.xlabel("LTTB Threshold (s)", fontsize=12)
plt.ylabel("Crash Risk Probability", fontsize=12)
plt.title("Fused LTTB Crash Risk Curve with 95% Confidence Interval\n(Bayesian Fusion of EVT and RF)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
