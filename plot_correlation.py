import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

import config

OUTPUT_DIR = config.OUTPUT_DIR

METRICS = {
    "eta_trip": r"$\eta_{\mathrm{trip}}$",
    "eta_pt": r"$\eta_{\mathrm{pt}}$",
    "eta_proximity": r"$\eta_{\mathrm{proximity}}$ (m)",
}


def _classify(row):
    return "diagonal" if abs(row["w1"] - row["w2"]) < 1e-9 else "offdiag"


def _ordered_label(row):
    return f"({row['w1']:.1f}, {row['w2']:.1f})"


def plot_metric_trends(results_df: pd.DataFrame):
    df = results_df.copy()
    df["kind"] = df.apply(_classify, axis=1)

    diag = df[df["kind"] == "diagonal"].sort_values("w1").reset_index(drop=True)
    offdiag = df[df["kind"] == "offdiag"].sort_values(["w1", "w2"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, (col, ylabel) in zip(axes, METRICS.items()):
        ax.plot(
            range(len(diag)), diag[col],
            marker="o", markersize=6, linewidth=1.8, color="#21918c",
            label=r"Diagonal ($w_1 = w_2$)", zorder=3,
        )
        for i, (_, row) in enumerate(offdiag.iterrows()):
            ax.scatter(
                len(diag) + i, row[col],
                marker="D", s=50, color="#fde725",
                edgecolors="#555555", linewidths=0.8, zorder=4,
            )

        ax.axvline(len(diag) - 0.5, color="#aaaaaa", linestyle="--", linewidth=0.9, zorder=1)

        xticks = list(range(len(diag))) + list(range(len(diag), len(diag) + len(offdiag)))
        xlabels_diag = [f"$w_1\\!=\\!{r.w1:.1f}$" for _, r in diag.iterrows()]
        xlabels_off = [_ordered_label(r) for _, r in offdiag.iterrows()]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels_diag + xlabels_off, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("$(w_1,\\ w_2)$ configuration", fontsize=9, labelpad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    h1 = mlines.Line2D([], [], color="#21918c", marker="o", markersize=6,
                       label=r"Diagonal ($w_1 = w_2$)")
    h2 = mlines.Line2D([], [], color="none", marker="D", markersize=7,
                       markerfacecolor="#fde725", markeredgecolor="#555555",
                       label=r"Off-diagonal ($w_1 \neq w_2$, Case 5)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, fontsize=9,
               frameon=True, framealpha=0.9, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Correlation Analysis: Metric Response to Policy Weight Grid",
                 fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = OUTPUT_DIR / f"correlation_trends.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {p}")
    plt.show()


def plot_spearman_table(spearman_df: pd.DataFrame):
    df = spearman_df.copy()

    def _stars(p):
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    df["sig"] = df["p_value"].apply(_stars)
    df["r_fmt"] = df.apply(lambda r: f"{r['spearman_r']:+.4f}{r['sig']}", axis=1)
    df["p_fmt"] = df["p_value"].apply(lambda p: f"{p:.4f}")

    col_labels = ["Correlation", r"$r_s$", "$p$-value", "Direction"]
    table_data = [
        [row["correlation"], row["r_fmt"], row["p_fmt"], row["sign"]]
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, 0.55 * (len(df) + 1.5)))
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#31688e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(df) + 1):
        bg = "#f0f8ff" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        "Partial Spearman Rank Correlations\n" r"(* $p<0.05$,  ** $p<0.01$)",
        fontsize=10, fontweight="bold", pad=8,
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = OUTPUT_DIR / f"spearman_table.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {p}")
    plt.show()


def run(results_csv=None, spearman_csv=None):
    results_path = results_csv or OUTPUT_DIR / "correlation_results.csv"
    spearman_path = spearman_csv or OUTPUT_DIR / "spearman_rank.csv"

    print(f"  Loading {results_path} ...")
    results_df = pd.read_csv(results_path)

    print(f"  Loading {spearman_path} ...")
    spearman_df = pd.read_csv(spearman_path)

    print(f"  {len(results_df)} configurations loaded.")
    plot_metric_trends(results_df)
    plot_spearman_table(spearman_df)


if __name__ == "__main__":
    run()
