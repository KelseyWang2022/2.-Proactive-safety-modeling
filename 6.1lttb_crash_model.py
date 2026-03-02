import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, probplot, kstest
from statsmodels.distributions.empirical_distribution import ECDF

# 固定随机种子确保可重复性
np.random.seed(42)

# 读取非事故和事故 LTTB 数据
non_crash = pd.read_csv("merged_filtered_data(april_may_june).csv")["LTTB"].dropna()
crash = pd.read_csv("TTC_results.csv")["LTTB"].dropna()

# 设置初始阈值为 1.2
threshold = 2.0
excesses = threshold - non_crash[non_crash < threshold]
excesses = excesses[excesses > 0]

# GPD 拟合
c, loc, scale = genpareto.fit(excesses)
print(f"GPD parameters: c={c:.4f}, loc={loc:.4f}, scale={scale:.4f}")

# KS检验（对GPD拟合的优劣进行定量检验）
D_stat, p_value = kstest(excesses, 'genpareto', args=(c, loc, scale))
print(f"KS test p-value: {p_value:.3f}")

# 经验分布函数 vs 理论分布函数
ecdf = ECDF(excesses)
x_vals = np.linspace(0, 1, 100)
ecdf_vals = ecdf(x_vals)
theoretical_vals = genpareto.cdf(x_vals, c, loc=loc, scale=scale)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, ecdf_vals, label='Empirical CDF', color='blue')
plt.plot(x_vals, theoretical_vals, label='Theoretical CDF (GPD)', color='red')
plt.title('Empirical CDF vs Theoretical CDF (GPD)')
plt.xlabel('Excesses')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid()
plt.show()

# 绘制QQ图 + 置信带
fig, ax = plt.subplots(figsize=(8, 5))
osm, osr = probplot(excesses, dist=genpareto(c, loc=0, scale=scale), fit=False)
sorted_osr = np.sort(osr)
theoretical = np.sort(osm)

ax.plot(theoretical, sorted_osr, 'o', label='Observed Data')
ax.plot(theoretical, theoretical, 'r-', label='Ideal Fit Line')

# 加入置信带
ci_upper = theoretical + 1.96 * np.std(sorted_osr)
ci_lower = theoretical - 1.96 * np.std(sorted_osr)
ax.fill_between(theoretical, ci_lower, ci_upper, color='gray', alpha=0.3, label="95% Confidence Band")

ax.set_title(f"QQ Plot With 95% CI for Fitted GPD\nKolmogorov-Smirnov p-value = {p_value:.3f}")
ax.set_xlabel("Theoretical Quantiles (GPD)")
ax.set_ylabel("Ordered Observed Excesses")
ax.legend()
plt.tight_layout()
plt.show()

# 估算事故概率曲线 P(crash | LTTB < x)
total_samples = len(non_crash) + len(crash)
total_crash_obs = len(crash)
P_crash = total_crash_obs / total_samples

x_vals = np.linspace(0.1, 5, 100)
p_lttb_less_x = [(non_crash < x).mean() for x in x_vals]
p_lttb_less_x_given_crash = [(crash < x).mean() for x in x_vals]
p_crash_given_lttb = (np.array(p_lttb_less_x_given_crash) * P_crash) / np.array(p_lttb_less_x)

plt.figure(figsize=(10, 5))
plt.plot(x_vals, p_crash_given_lttb, label="P(crash | LTTB < x)", color="red")
plt.xlabel("LTTB threshold x (s)")
plt.ylabel("P(crash | LTTB < x)")
plt.title("Estimated Crash Probability given LTTB < x")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 输入LTTB值，计算事故发生的概率
def calculate_crash_probability(lttb_value):
    p_lttb = (non_crash < lttb_value).mean()
    p_crash_given_lttb_value = (len(crash[crash < lttb_value]) / len(crash)) * P_crash / p_lttb
    return p_crash_given_lttb_value

# 示例：输入一个LTTB值
lttb_input = 0.05
probability = calculate_crash_probability(lttb_input)
print(f"Estimated crash probability for LTTB = {lttb_input} is: {probability:.4f}")

# 绘制 GPD 拟合的 PDF 和 CDF
x_vals = np.linspace(0.01, threshold - 1e-5, 100)
excess_vals = threshold - x_vals
pdf_vals = genpareto.pdf(excess_vals, c, loc=loc, scale=scale)
cdf_vals = genpareto.cdf(excess_vals, c, loc=loc, scale=scale)

plt.figure(figsize=(10,5))
plt.plot(x_vals, pdf_vals, label="GPD PDF (for LTTB < threshold)")
plt.xlabel("LTTB Value")
plt.ylabel("Density")
plt.title("Probability Density Function (GPD fitted to excesses)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(x_vals, cdf_vals, label="GPD CDF (for LTTB < threshold)", color='red')
plt.xlabel("LTTB Value")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution Function (GPD fitted to excesses)")
plt.legend()
plt.grid()
plt.show()

# 某一点的 PDF 和 CDF （如 LTTB = 0.9）
lttb_value = 0.9
if lttb_value < threshold:
    excess_at_point = threshold - lttb_value
    pdf_at_point = genpareto.pdf(excess_at_point, c, loc=loc, scale=scale)
    cdf_at_point = genpareto.cdf(excess_at_point, c, loc=loc, scale=scale)
    print(f"At LTTB = {lttb_value}, GPD PDF = {pdf_at_point:.4f}, CDF = {cdf_at_point:.4f}")
else:
    print(f"LTTB = {lttb_value} exceeds threshold {threshold}, GPD is not defined here.")
