import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import config
from config import TIMING_GRID

OUTPUT_DIR = config.OUTPUT_DIR

CASE_LABELS = {
    1: r"Case 1: Baseline FTCM" + "\n" + r"$(w_1=1.0,\,w_2=1.0)$",
    2: r"Case 2: De-prioritized MM" + "\n" + r"$(w_1=1.5,\,w_2=1.5)$",
    3: r"Case 3: Equal Single-Tier" + "\n" + r"$(w_1=2.0,\,w_2=2.0)$",
    4: r"Case 4: Prioritized MM" + "\n" + r"$(w_1=2.5,\,w_2=2.5)$",
    5: r"Case 5: Fully Differentiated" + "\n" + r"$(w_1=2.1,\,w_2=2.2)$",
}


def load_timing_data():
    w_cases = {}
    for w1, w2, case_num in TIMING_GRID:
        path = OUTPUT_DIR / f"timing_results_case{case_num}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping Case {case_num}.")
            continue
        w_cases[case_num] = pd.read_csv(path)
    return w_cases


def _build_xtick_labels(df):
    summary = (
        df.groupby("fraction")[["n_od_pairs", "n_stations"]]
        .first()
        .reset_index()
        .sort_values("fraction")
    )
    labels = [
        f"{row['fraction']:.1f}\n$|Q|$={int(row['n_od_pairs']):,}\n$|J|$={int(row['n_stations'])}"
        for _, row in summary.iterrows()
    ]
    return summary["fraction"].tolist(), labels


def plot_timing(w_cases: dict):
    n_cases = len(w_cases)
    fig, axes = plt.subplots(1, n_cases, figsize=(4.2 * n_cases, 5.2), sharey=False)
    if n_cases == 1:
        axes = [axes]

    all_alphas = sorted(set(a for df in w_cases.values() for a in df["alpha"].unique()))
    cmap = plt.cm.viridis
    alpha_colors = {
        a: cmap(i / max(len(all_alphas) - 1, 1))
        for i, a in enumerate(all_alphas)
    }

    for ax, (case_num, df) in zip(axes, sorted(w_cases.items())):
        fractions, xtick_labels = _build_xtick_labels(df)
        frac_to_x = {f: i for i, f in enumerate(fractions)}

        for alpha_val in sorted(df["alpha"].unique()):
            sub = df[df["alpha"] == alpha_val].sort_values("fraction")
            ax.plot(
                [frac_to_x[f] for f in sub["fraction"]],
                sub["solve_time_s"].tolist(),
                marker="o", markersize=5, linewidth=1.8,
                color=alpha_colors[alpha_val],
                label=f"$\\alpha={alpha_val}$",
            )

        ax.set_xticks(list(range(len(fractions))))
        ax.set_xticklabels(xtick_labels, fontsize=7.5)
        ax.set_xlabel("Fraction of $Q$", fontsize=9, labelpad=8)
        ax.set_title(CASE_LABELS.get(case_num, f"Case {case_num}"), fontsize=9, pad=10)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(axis="x", linestyle=":", alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Solve time (seconds)", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(all_alphas), fontsize=9,
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Time-Feasibility Analysis: Solve Time vs. OD Sample Fraction",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    png_path = OUTPUT_DIR / "timing_plot.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"  Saved -> {png_path}")
    plt.show()


if __name__ == "__main__":
    plot_timing(load_timing_data())
