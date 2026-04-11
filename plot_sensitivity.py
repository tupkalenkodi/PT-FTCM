import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import config
from config import TIMING_GRID

OUTPUT_DIR = config.OUTPUT_DIR

METRICS = {
    "eta_trip": r"$\eta_{\mathrm{trip}}$",
    "eta_pt": r"$\eta_{\mathrm{pt}}$",
    "eta_proximity": r"$\eta_{\mathrm{proximity}}$ (m)",
}

CASE_LABELS = {
    1: r"Case 1" + "\n" + r"$w_1\!=\!1.0,\,w_2\!=\!1.0$",
    2: r"Case 2" + "\n" + r"$w_1\!=\!1.5,\,w_2\!=\!1.5$",
    3: r"Case 3" + "\n" + r"$w_1\!=\!2.0,\,w_2\!=\!2.0$",
    4: r"Case 4" + "\n" + r"$w_1\!=\!2.5,\,w_2\!=\!2.5$",
    5: r"Case 5" + "\n" + r"$w_1\!=\!2.1,\,w_2\!=\!2.2$",
}

CASE_COLORS = {
    case_num: plt.cm.viridis(i / (len(TIMING_GRID) - 1))
    for i, (_, _, case_num) in enumerate(TIMING_GRID)
}


def load_sensitivity_data():
    cases = {}
    for w1, w2, case_num in TIMING_GRID:
        path = OUTPUT_DIR / f"sensitivity_results_case{case_num}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping Case {case_num}.")
            continue
        cases[case_num] = pd.read_csv(path)
    return cases


def plot_sensitivity(cases: dict):
    n_metrics = len(METRICS)
    n_cases = len(cases)

    fig, axes = plt.subplots(
        n_metrics, n_cases,
        figsize=(3.0 * n_cases, 3.2 * n_metrics),
        sharey="row",
        sharex="col",
    )

    if n_cases == 1:
        axes = np.array(axes).reshape(n_metrics, 1)

    metric_keys = list(METRICS.keys())
    metric_labels = list(METRICS.values())
    sorted_cases = sorted(cases.items())

    for col, (case_num, df) in enumerate(sorted_cases):
        df_sorted = df.sort_values("delta").reset_index(drop=True)
        deltas = df_sorted["delta"].tolist()
        color = CASE_COLORS[case_num]

        for row, metric in enumerate(metric_keys):
            ax = axes[row][col]

            means_arr = np.array(df_sorted[f"{metric}_mean"].tolist())
            stds_arr = np.array(df_sorted[f"{metric}_std"].tolist())

            ax.fill_between(
                deltas, means_arr - stds_arr, means_arr + stds_arr,
                alpha=0.25, color=color, linewidth=0,
            )
            ax.plot(
                deltas, means_arr,
                marker="o", markersize=4.5, linewidth=1.8, color=color,
            )

            ax.set_xticks(deltas)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(CASE_LABELS.get(case_num, f"Case {case_num}"), fontsize=8.5, pad=6)
            if row == n_metrics - 1:
                ax.set_xlabel(r"Perturbation $\delta$", fontsize=8.5, labelpad=4)
            if col == 0:
                ax.set_ylabel(metric_labels[row], fontsize=9, labelpad=4)

    fig.suptitle(
        "Sensitivity Analysis: Metric Distributions under Demand Perturbation\n"
        r"(shaded band = $\pm 1$ std over 10 realisations)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = OUTPUT_DIR / f"sensitivity_plot.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {p}")
    plt.show()


if __name__ == "__main__":
    plot_sensitivity(load_sensitivity_data())
