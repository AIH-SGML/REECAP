import pandas as pd
import numpy as np
import os
from os.path import join
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(script_dir, '../data/external_val/processed')

OUT_DIR = join(BASE_DIR, "retfound_vs_reecap_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

def paired_dot_plot(x_vals, y_vals, labels, ylabel, title, out_path, ylim=None):
    plt.figure(figsize=(3.2, 4))

    jitter = 0.04
    x_positions = np.arange(len(labels))

    COLORS = {
        "REECAP": "#E69F00",
        "RETFound": "#0072B2"
    }

    for i, vals in enumerate([x_vals, y_vals]):
        x_jittered = np.random.normal(x_positions[i], jitter, size=len(vals))
        plt.scatter(x_jittered, vals, s=30, alpha=0.9, color=COLORS[labels[i]])
        mean_val = np.mean(vals)
        plt.plot(
            [x_positions[i] - 0.15, x_positions[i] + 0.15],
            [mean_val, mean_val],
            linewidth=2,
            color=COLORS[labels[i]]
        )

    plt.xticks(x_positions, labels, rotation=30)
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="0.7", alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def compare_and_plot(df_r1, df_r2, metric, label1, label2, ylim=None):
    x = df_r1[metric].values
    y = df_r2[metric].values

    t, p = ttest_rel(x, y)

    paired_dot_plot(
        x_vals=x,
        y_vals=y,
        labels=[label1, label2],
        ylabel=metric if metric != "MAE" else "MAE, years",
        title="age predictions",
        out_path=join(OUT_DIR, f"{metric}_dotplot.png"),
        ylim=ylim
    )

    return {
        "metric": metric,
        "mean_" + label1: x.mean(),
        "mean_" + label2: y.mean(),
        "std_" + label1: x.std(),
        "std_" + label2: y.std(),
        "t_stat": t,
        "p_value": p
    }


df_reecap = pd.read_csv(join(BASE_DIR, "REECAP_features/age_prediction_fold_metrics.csv"))
df_retfound = pd.read_csv(join(BASE_DIR, "RETFound_features/age_prediction_fold_metrics.csv"))

assert len(df_reecap) == len(df_retfound), "Mismatch in number of folds"

all_results = []
for metric in ["MAE", "R2"]:
    res = compare_and_plot(
        df_reecap, df_retfound,
        metric=metric,
        label1="REECAP",
        label2="RETFound",
        ylim=(4, 6) if metric == "MAE" else None
    )
    all_results.append(res)

stats_df = pd.DataFrame(all_results)
stats_path = join(OUT_DIR, "paired_ttest_results.csv")
stats_df.to_csv(stats_path, index=False)

print(f"Saved paired t-test results to:\n{stats_path}")