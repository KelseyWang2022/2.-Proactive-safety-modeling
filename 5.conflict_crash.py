
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, gaussian_kde, ks_2samp

# 读取数据
df_all = pd.read_csv("ttc_lttb_final_split_by_type_yaw(april_may_june).csv")
df_accident = pd.read_csv("TTC_results.csv")

# 确保 LTTB 是浮点型
df_all["LTTB"] = pd.to_numeric(df_all["LTTB"], errors="coerce")
df_accident["LTTB"] = pd.to_numeric(df_accident["LTTB"], errors="coerce")

# 筛选非事故数据
non_crash_df = df_all[~df_all["type"].str.contains("crash|accident", case=False, na=False)]

# 设置 EVT 阈值起点
custom_threshold = 0.57
extreme_lttb = non_crash_df[non_crash_df["LTTB"] < custom_threshold]["LTTB"].dropna()
excesses = custom_threshold - extreme_lttb

# EVT GPD 拟合
params = genpareto.fit(excesses)
quantiles = genpareto.ppf([0.95, 0.99], *params)
thresholds = custom_threshold - quantiles
print("EVT 阈值（LTTB）：")
print(f"95% quantile = {thresholds[0]:.3f} 秒")
print(f"99% quantile = {thresholds[1]:.3f} 秒")

# 事故 vs 非事故数据准备
lttb_crash = df_accident[df_accident["crash"] == 1]["LTTB"].dropna()
lttb_non_crash = df_accident[df_accident["crash"] == 0]["LTTB"].dropna()

# KDE 分布估计
kde_crash = gaussian_kde(lttb_crash)
kde_non_crash = gaussian_kde(lttb_non_crash)
x_vals = np.linspace(0, max(lttb_crash.max(), lttb_non_crash.max()), 200)
y_crash = kde_crash(x_vals)
y_non_crash = kde_non_crash(x_vals)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_non_crash, label="non-crash", color="blue", alpha=0.6)
plt.plot(x_vals, y_crash, label="crash", color="red", alpha=0.6)
plt.fill_between(x_vals, y_non_crash, color="blue", alpha=0.3)
plt.fill_between(x_vals, y_crash, color="red", alpha=0.3)
plt.axvline(thresholds[0], color="orange", linestyle="--", label="EVT 95% threshold")
plt.axvline(thresholds[1], color="purple", linestyle="--", label="EVT 99% threshold")
plt.xlabel("LTTB")
plt.ylabel("density estimation")
plt.title("LTTB kernel density estimation: crash vs non-crash")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# KS检验
ks_result = ks_2samp(lttb_crash, lttb_non_crash)
print("\\nKS检验结果：")
print(ks_result)
