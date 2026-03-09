
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, beta, probplot, kstest
from statsmodels.distributions.empirical_distribution import ECDF

# 固定随机种子确保可重复性
np.random.seed(42)

# 读取非事故和事故 LTTB 数据
non_crash = pd.read_csv("merged_filtered_data(april_may_june).csv")["LTTB"].dropna()
crash = pd.read_csv("TTC_results.csv")["LTTB"].dropna()
non_crash = non_crash[non_crash < 100]
crash = crash[crash < 100]

# 设置初始阈值为 5
threshold = 2.0
excesses = threshold - non_crash[non_crash < threshold]
excesses = excesses[excesses > 0]

# GPD 拟合
c, loc, scale = genpareto.fit(excesses)

# KS检验（对GPD拟合的优劣进行定量检验）
D_stat, p_value = kstest(excesses, 'genpareto', args=(c, loc, scale))
print(f"KS test p-value: {p_value:.3f}")

# 计算贝叶斯定理中的各项
total_samples = len(non_crash) + len(crash)  # 总样本数量
total_crash_obs = len(crash)  # 事故样本数量
P_crash = total_crash_obs / total_samples  # P(crash)

# 计算P(LTTB < x)
x_vals = np.linspace(0.1, 5, 100)
p_lttb_less_x = []
for x in x_vals:
    p = (non_crash < x).mean()  # 所有非事故样本中，LTTB小于x的比例
    p_lttb_less_x.append(p)

# 计算P(LTTB < x | crash)
p_lttb_less_x_given_crash = []
for x in x_vals:
    p_crash_given_lttb = (crash < x).mean()  # 在事故样本中，LTTB小于x的比例
    p_lttb_less_x_given_crash.append(p_crash_given_lttb)

# 使用贝叶斯定理计算P(crash | LTTB < x)
p_crash_given_lttb = (np.array(p_lttb_less_x_given_crash) * P_crash) / np.array(p_lttb_less_x)

# 归一化，确保概率值在0到1之间
p_crash_given_lttb = np.clip(p_crash_given_lttb, 0, 1)

# -----> 计算事故概率的置信区间 (Bootstrap 方法)
bootstrap_samples = 1000  # Bootstrap 重抽样次数
bootstrap_p_crash_given_lttb = np.zeros((bootstrap_samples, len(x_vals)))

for i in range(bootstrap_samples):
    # 在非事故和事故数据中进行抽样
    sample_non_crash = non_crash.sample(n=len(non_crash), replace=True)
    sample_crash = crash.sample(n=len(crash), replace=True)

    # 计算每个LTTB阈值下的事故概率
    for idx, x in enumerate(x_vals):
        p_lttb_sample = (sample_non_crash < x).mean()
        p_crash_given_lttb_sample = (len(sample_crash[sample_crash < x]) / len(sample_crash)) * P_crash / p_lttb_sample
        bootstrap_p_crash_given_lttb[i, idx] = p_crash_given_lttb_sample

# 计算事故概率的95%置信区间
p_crash_lower = np.percentile(bootstrap_p_crash_given_lttb, 2.5, axis=0)
p_crash_upper = np.percentile(bootstrap_p_crash_given_lttb, 97.5, axis=0)

# 可视化事故概率曲线和置信区间
plt.figure(figsize=(10, 5))
plt.plot(x_vals, p_crash_given_lttb, label="P(crash | LTTB < x)", color="red")
plt.fill_between(x_vals, p_crash_lower, p_crash_upper, color="red", alpha=0.3, label="95% Confidence Interval")

plt.xlabel("TTC threshold x (s)")
plt.ylabel("P(crash | TTC < x)")
plt.title("Estimated Crash Probability given TTC < x with 95% Confidence Interval")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # 设置y轴范围为0到1
plt.tight_layout()
plt.show()

# 输入LTTB值，计算事故发生的概率
def calculate_crash_probability(lttb_value):
    p_lttb = (non_crash < lttb_value).mean()  # 计算 P(LTTB < x)
    p_crash_given_lttb_value = (len(crash[crash < lttb_value]) / len(crash)) * P_crash / p_lttb  # 计算 P(crash | LTTB < x)
    return p_crash_given_lttb_value

# 输入一个LTTB值，计算事故发生的概率
lttb_input = 0.05  # 
probability = calculate_crash_probability(lttb_input)
print(f"Estimated crash probability for LTTB = {lttb_input} is: {probability:.4f}")
