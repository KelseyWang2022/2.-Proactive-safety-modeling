import os, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

NON_CRASH_PATH = "/mnt/data/merged_filtered_data.csv"
CRASH_PATHS = ["/mnt/data/case_1.csv","/mnt/data/case_2.csv","/mnt/data/case_4.csv","/mnt/data/case_5.csv"]
HORIZONS = [0.5,1.0,1.5,2.0]
RANDOM_STATE=42

def detect_column(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k.lower() in col.lower():
                return col
    raise ValueError(f"Cannot find column with keywords {keywords}. Available: {list(df.columns)}")

def build_crash_labels(df, time_col, horizon):
    T_acc = df[time_col].max()
    out = df.copy()
    out["label"] = ((out[time_col] >= T_acc - horizon) & (out[time_col] < T_acc)).astype(int)
    return out

def load_standardize_case(path, horizon):
    df = pd.read_csv(path)
    time_col = detect_column(df, ["time"])
    lttb_col = detect_column(df, ["lttb"])
    ttc_col = detect_column(df, ["ttc"])
    df = build_crash_labels(df, time_col, horizon)
    df = df[[time_col, lttb_col, ttc_col, "label"]].copy()
    df.columns = ["time","LTTB","TTC","label"]
    return df

def load_standardize_noncrash(path):
    df = pd.read_csv(path)
    time_col = detect_column(df, ["time"])
    lttb_col = detect_column(df, ["lttb"])
    ttc_col = detect_column(df, ["ttc"])
    df["label"] = 0
    df = df[[time_col, lttb_col, ttc_col, "label"]].copy()
    df.columns = ["time","LTTB","TTC","label"]
    return df

def run_loio_experiment(horizons=HORIZONS, random_state=RANDOM_STATE, n_estimators=300):
    non_crash = load_standardize_noncrash(NON_CRASH_PATH)
    results = []
    for horizon in horizons:
        crash_dfs = [load_standardize_case(p, horizon) for p in CRASH_PATHS]
        for holdout_idx, holdout_path in enumerate(CRASH_PATHS):
            test_crash = crash_dfs[holdout_idx]
            train_crash = pd.concat([crash_dfs[i] for i in range(len(crash_dfs)) if i!=holdout_idx], ignore_index=True)
            train_nc, test_nc = train_test_split(non_crash, test_size=0.3, random_state=random_state, shuffle=True)
            train_df = pd.concat([train_nc, train_crash], ignore_index=True)
            test_df = pd.concat([test_nc, test_crash], ignore_index=True)
            for feature in ["LTTB","TTC"]:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1
                )
                med = train_df[feature].median()
                X_train = train_df[[feature]].fillna(med)
                y_train = train_df["label"].astype(int)
                X_test = test_df[[feature]].fillna(med)
                y_test = test_df["label"].astype(int)

                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:,1]
                roc = roc_auc_score(y_test, probs)
                pr = average_precision_score(y_test, probs)

                results.append({
                    "horizon": horizon,
                    "feature": feature,
                    "holdout_case": os.path.basename(holdout_path),
                    "roc_auc": float(roc),
                    "pr_auc": float(pr),
                    "n_test": int(len(y_test)),
                    "pos_test": int(y_test.sum())
                })
    df_results = pd.DataFrame(results)
    summary = (df_results.groupby(["horizon","feature"])
               .agg(roc_mean=("roc_auc","mean"),
                    roc_std=("roc_auc","std"),
                    pr_mean=("pr_auc","mean"),
                    pr_std=("pr_auc","std"))
               .reset_index()
               .sort_values(["horizon","feature"]))
    return df_results, summary

df_results, summary = run_loio_experiment()

df_results_path = "/mnt/data/rf_per_fold_results_rawdata.csv"
summary_path = "/mnt/data/rf_summary_results_rawdata.csv"
df_results.to_csv(df_results_path, index=False)
summary.to_csv(summary_path, index=False)

# Plotting
summ = summary.copy()
horizons = sorted(summ["horizon"].unique())
features = ["LTTB","TTC"]

def plot_metric(metric_mean, metric_std, title, ylabel, outpath):
    x = np.arange(len(horizons))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,4.5))
    for i, feat in enumerate(features):
        sub = summ[summ["feature"]==feat].set_index("horizon").loc[horizons]
        means = sub[metric_mean].values
        stds  = sub[metric_std].values
        ax.bar(x + (i-0.5)*width, means, width, yerr=stds, capsize=4, label=feat)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0,1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

roc_fig = "/mnt/data/rf_roc_auc_by_horizon.png"
pr_fig  = "/mnt/data/rf_pr_auc_by_horizon.png"
plot_metric("roc_mean","roc_std","Random Forest performance vs prediction horizon (ROC-AUC)","ROC-AUC",roc_fig)
plot_metric("pr_mean","pr_std","Random Forest performance vs prediction horizon (PR-AUC)","PR-AUC",pr_fig)

(os.path.exists(df_results_path), os.path.exists(summary_path), os.path.exists(roc_fig), os.path.exists(pr_fig), summary)