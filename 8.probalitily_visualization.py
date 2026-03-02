import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# 读取事故与非事故数据
crash_df = pd.read_csv("TTC_results.csv")
non_crash_df = pd.read_csv("merged_filtered_data(april_may_june).csv")

# 添加标签并合并
crash_df["crash"] = 1
non_crash_df["crash"] = 0
combined_df = pd.concat([crash_df, non_crash_df], ignore_index=True)

# EVT 拟合 LTTB 的 GPD 参数
threshold = 2.0
excesses_lttb = threshold - non_crash_df["LTTB"]
excesses_lttb = excesses_lttb[excesses_lttb > 0]
c_lttb, loc_lttb, scale_lttb = genpareto.fit(excesses_lttb)

# 定义风险概率计算函数
def compute_risk_curve(df, var_name, c_evt, loc_evt, scale_evt, thresholds, n_bootstrap=300):
    risk_means, ci_lower, ci_upper = [], [], []

    for t in thresholds:
        risks = []
        for _ in range(n_bootstrap):
            sample = df.sample(frac=1, replace=True)
            subset = sample[sample[var_name] <= t]
            if len(subset) == 0:
                risks.append(0)
            else:
                risks.append(subset["crash"].mean())
        risk_means.append(np.mean(risks))
        ci_lower.append(np.percentile(risks, 2.5))
        ci_upper.append(np.percentile(risks, 97.5))

    return np.array(risk_means), np.array(ci_lower), np.array(ci_upper)

# 设置阈值范围
thresholds = np.linspace(0.2, 2.5, 30)

# 计算 LTTB 风险概率与置信区间
risk_lttb, lttb_lower, lttb_upper = compute_risk_curve(
    combined_df, "LTTB", c_lttb, loc_lttb, scale_lttb, thresholds
)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(thresholds, risk_lttb, label="LTTB Risk", color="blue")
plt.fill_between(thresholds, lttb_lower, lttb_upper, color="blue", alpha=0.2)
plt.xlabel("Threshold (s)")
plt.ylabel("Crash Risk Probability")
plt.title("LTTB Risk Probability Curve with 95% Confidence Interval")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
