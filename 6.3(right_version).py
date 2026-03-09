import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, probplot, kstest
from statsmodels.distributions.empirical_distribution import ECDF

# 固定随机种子确保可重复性
np.random.seed(42)

# 读取非事故和事故 LTTB 数据
non_crash = pd.read_csv("merged_filtered_data(april_may_june).csv")["TTC"].dropna()
crash = pd.read_csv("TTC_results.csv")["TTC"].dropna()
#取倒数


# 设置初始阈值
threshold = 1.1#1.2
excesses = threshold - non_crash[non_crash < threshold]
excesses = excesses[excesses > 0]

# GPD 拟合
c, loc, scale = genpareto.fit(excesses)

# KS检验
D_stat, p_value = kstest(excesses, 'genpareto', args=(c, loc, scale))
print(f"KS test p-value: {p_value:.3f}")

# 绘制经验CDF vs GPD理论CDF
ecdf = ECDF(excesses)
x_vals = np.linspace(0, 1, 100)
ecdf_vals = ecdf(x_vals)
theoretical_vals = genpareto.cdf(x_vals, c, loc=loc, scale=scale)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, ecdf_vals, label='Empirical CDF', color='blue')
plt.plot(x_vals, theoretical_vals, label='Theoretical CDF (GPD)', color='red')

plt.title('Empirical CDF vs Theoretical CDF (GPD)')
plt.xlabel('Excesses (Threshold - TTC)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the probability density function (PDF)
plt.figure(figsize=(10, 5))
x_vals = np.linspace(0, max(excesses), 100)
pdf_vals = genpareto.pdf(x_vals, c, loc=loc, scale=scale)

plt.plot(x_vals, pdf_vals, label='Theoretical PDF (GPD)', color='red')
plt.title('Probability Density Function (PDF) of GPD Fit')
plt.xlabel('Excesses (Threshold - LTTB)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
