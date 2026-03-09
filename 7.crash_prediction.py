import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
ttc_data = pd.read_csv('TTC_results.csv')
lttb_data = pd.read_csv('merged_filtered_data(april_may_june).csv')

# 合并数据（假设两者列结构相同：'TTC', 'LTTB', 'crash'）
data = pd.concat([ttc_data[['TTC', 'LTTB', 'crash']], lttb_data[['TTC', 'LTTB', 'crash']]], ignore_index=True)

# 绘制 ROC 曲线和计算 AUC
def plot_roc_and_metrics(data, feature):
    y_true = data['crash']
    y_scores = data[feature]
    y_scores_neg = -y_scores  # 反转分数（LTTB/TTC越小风险越大）

    fpr, tpr, _ = roc_curve(y_true, y_scores_neg)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{feature} (AUC = {roc_auc:.2f})')
    return roc_auc

plt.figure(figsize=(8, 6))
auc_lttb = plot_roc_and_metrics(data, 'LTTB')
auc_ttc = plot_roc_and_metrics(data, 'TTC')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (LTTB vs TTC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f'\nAUC Comparison: LTTB = {auc_lttb:.3f}, TTC = {auc_ttc:.3f}')

#  在不同阈值下计算灵敏度、特异性、Youden指数，并绘制曲线
def plot_extended_metrics_and_youden(data, feature, thresholds=np.arange(0.5, 3.5, 0.5)):
    y_true = data['crash']
    y_scores = data[feature]
    sensitivity_list, specificity_list, youden_list = [], [], []

    print(f'\nExtended threshold performance for {feature}:')
    for thresh in thresholds:
        y_pred = (y_scores <= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sensitivity + specificity - 1
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        youden_list.append(youden)
        print(f'Threshold: {thresh:.1f} | Sensitivity: {sensitivity:.2f} | Specificity: {specificity:.2f} | Youden Index: {youden:.2f}')

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sensitivity_list, marker='o', label='Sensitivity')
    plt.plot(thresholds, specificity_list, marker='s', label='Specificity')
    plt.plot(thresholds, youden_list, marker='^', label="Youden's Index")
    plt.xlabel(f'{feature} Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Extended Threshold Performance for {feature}')
    plt.legend()
    plt.grid(True)
    plt.show()

# 分别绘制 LTTB 和 TTC 的性能曲线
plot_extended_metrics_and_youden(data, 'LTTB')
plot_extended_metrics_and_youden(data, 'TTC')
