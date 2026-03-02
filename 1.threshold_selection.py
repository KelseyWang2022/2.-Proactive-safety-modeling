# 导入必要的库并加载数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# 读取数据
df = pd.read_csv('merged_filtered_data(april_may_june).csv')
lttb_values = df["LTTB"].dropna()
TTC_values = df["TTC"].dropna()
#选择TTC值小于3的


# ----------------------------
# 1. 平均残差图（Mean Residual Life Plot）
# ----------------------------
# 定义一个函数计算残差
def mean_residual_life_plot(data, threshold_range):
    residuals = []
    for threshold in threshold_range:
        excesses = data[data > threshold] - threshold
        if len(excesses) > 0:
            residuals.append(np.mean(excesses))
        else:
            residuals.append(np.nan)
    return residuals

# 设置阈值范围
threshold_range = np.linspace(0, 10, 50)
mean_residuals = mean_residual_life_plot(lttb_values, threshold_range)
mean_residuals_ttc = mean_residual_life_plot(TTC_values, threshold_range)

# 绘制平均残差图
plt.figure(figsize=(10, 6))
plt.plot(threshold_range, mean_residuals_ttc , label="Mean Residual Life", color='blue')
plt.title("Mean Residual Life Plot")
plt.xlabel("Threshold")
plt.ylabel("Mean Residual Life")
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------
# 2. 阈值稳定性图（Threshold Stability Plot）
# ----------------------------
# 计算不同阈值下的分布稳定性
def threshold_stability_plot(data, threshold_range):
    stability = []
    for threshold in threshold_range:
        excesses = data[data > threshold] - threshold
        if len(excesses) > 0:
            stability.append(np.std(excesses))
        else:
            stability.append(np.nan)
    return stability

# 计算并绘制阈值稳定性图
stability_values = threshold_stability_plot(lttb_values, threshold_range)
stability_values_ttc = threshold_stability_plot(TTC_values, threshold_range)
stability_values_lttb = threshold_stability_plot(lttb_values, threshold_range)
plt.figure(figsize=(10, 6))
plt.plot(threshold_range, stability_values_ttc, label="Threshold Stability", color='red')
plt.title("Threshold Stability Plot")
plt.xlabel("Threshold")
plt.ylabel("Standard Deviation of Excesses")
plt.grid(True)
plt.legend()
plt.show()


# ----------------------------
#使用百分位数法计算阈值，选择5%的数据
def calculate_threshold(data, percentile=5):
    """计算给定数据的指定百分位数阈值"""
    return np.percentile(data, percentile)

# 计算LTTB和TTC的5%阈值
threshold_lttb = calculate_threshold(lttb_values, 5)
threshold_ttc = calculate_threshold(TTC_values, 5)
# 输出计算的阈值
print(f"LTTB 5% Threshold: {threshold_lttb}")
print(f"TTC 5% Threshold: {threshold_ttc}")
# 返回计算结果
mean_residuals, stability_values
