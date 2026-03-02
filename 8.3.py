import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from tqdm import tqdm

# 读取数据
crash_df = pd.read_csv("TTC_results.csv")
non_crash_df = pd.read_csv("merged_filtered_data(april_may_june).csv")

crash_df["crash"] = 1
non_crash_df["crash"] = 0

# 合并数据
combined_df = pd.concat([crash_df, non_crash_df], ignore_index=True)
combined_df = combined_df[combined_df["LTTB"].notna()]  # 删除缺失值
X = combined_df[["LTTB"]]
y = combined_df["crash"]

# ---------- EVT 模型：拟合事故数据中的低 LTTB ----------
crash_lttb = combined_df[combined_df["crash"] == 1]["LTTB"]
threshold_evt = 2.0
excesses = threshold_evt - crash_lttb[crash_lttb < threshold_evt]
c_evt, loc_evt, scale_evt = genpareto.fit(excesses)

# ---------- KDE 拟合 P(t)（无条件） ----------
kde_all = gaussian_kde(combined_df["LTTB"])

# ---------- RF 模型 ----------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# ---------- 贝叶斯融合 ----------
thresholds = np.linspace(0.5, 2.5, 60)
n_bootstrap = 200

mean_probs = []
lower_ci = []
upper_ci = []

for t in tqdm(thresholds):
    fused_probs = []
    for _ in range(n_bootstrap):
        boot_df = resample(combined_df, replace=True)
        Xb = boot_df[["LTTB"]]
        yb = boot_df["crash"]
        rf_bs = RandomForestClassifier(n_estimators=100)
        rf_bs.fit(Xb, yb)
        p_rf = rf_bs.predict_proba([[t]])[0][1]

        # EVT: P(t | crash)
        if t < threshold_evt:
            p_evt = genpareto.pdf(threshold_evt - t, c_evt, loc_evt, scale_evt)
        else:
            p_evt = 1e-6  # 防止为0

        # KDE: P(t)
        p_kde = kde_all(t)[0]

        # P(crash)
        p_crash = np.mean(yb)

        # 贝叶斯融合概率
        bayes_prob = (p_rf * p_evt * p_crash) / (p_kde + 1e-6)
        bayes_prob = min(max(bayes_prob, 0), 1)  # 限制在[0,1]
        fused_probs.append(bayes_prob)

    mean_probs.append(np.mean(fused_probs))
    lower_ci.append(np.percentile(fused_probs, 2.5))
    upper_ci.append(np.percentile(fused_probs, 97.5))

# ---------- 绘图 ----------
plt.figure(figsize=(10, 6))
plt.plot(thresholds, mean_probs, label="Bayesian Fused Crash Risk", color="darkgreen")
plt.fill_between(thresholds, lower_ci, upper_ci, color="lightgreen", alpha=0.4)
plt.xlabel("LTTB Threshold (s)")
plt.ylabel("Crash Risk Probability")
plt.title("Bayesian Fusion: EVT + RF Crash Probability Curve with 95% CI")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
