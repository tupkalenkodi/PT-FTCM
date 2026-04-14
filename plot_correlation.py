import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config


# Shared style, identical in plot_sensitivity.py
_RC = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "LMRoman10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
}
mpl.rcParams.update(_RC)

_DATA_LW = 1.5  # line width for w2 slices
_REF_LW = 2.0  # line width for mean / reference lines
_MARKER_SZ = 3.5 # marker size for mean w2 slices

# Sequential colormap for w2-indexed lines in the correlation plot
SEQ_CMAP = "viridis_r"

MEAN_COLOR = "black"
MEAN_STYLE = dict(color=MEAN_COLOR, linestyle="--", linewidth=_REF_LW, zorder=10)

METRIC_COL = "eta_pt"
METRIC_LABEL = r"$\eta_{\mathrm{pt}}$"

OUTPUT_DIR = config.OUTPUT_DIR


def main():
    csv_path = OUTPUT_DIR / "correlation_results.csv"
    spearman_path = OUTPUT_DIR / "spearman_rank.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[plot_correlation] Error: {csv_path} not found.  Run Stage 2 first.")
        return

    w1_vals = sorted(df["w1"].unique())
    w2_vals = sorted(df["w2"].unique())

    cmap = plt.get_cmap(SEQ_CMAP)
    norm = mpl.colors.Normalize(vmin=min(w2_vals), vmax=max(w2_vals))

    # Load partial Spearman results if available
    spearman_lookup: dict[str, tuple[float, float]] = {}
    if spearman_path.exists():
        for _, row in pd.read_csv(spearman_path).iterrows():
            spearman_lookup[str(row["correlation"])] = (
                float(row["spearman_r"]),
                float(row["p_value"]),
            )

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    # One line per w2 slice, colored by w2
    for w2 in w2_vals:
        subset = df[df["w2"] == w2].sort_values("w1")
        if subset.empty:
            continue
        ax.plot(
            subset["w1"], subset[METRIC_COL],
            marker="o", markersize=_MARKER_SZ,
            linewidth=_DATA_LW, alpha=0.7,
            color=cmap(norm(w2)),
        )

    # Mean across all w2 slices
    mean_df = df.groupby("w1")[METRIC_COL].mean().reset_index()
    ax.plot(mean_df["w1"], mean_df[METRIC_COL], label="Mean", **MEAN_STYLE)

    ax.set_xlabel(r"Policy weight $w_1$", fontsize=14, labelpad=3)
    ax.set_ylabel(METRIC_LABEL, fontsize=14, labelpad=3)
    ax.set_xticks(w1_vals)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.legend(loc="upper left", frameon=True, framealpha=0.88,
              edgecolor="#CCCCCC", fontsize=10)

    # colorbar appended to the right
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.14)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$w_2$", fontsize=12, labelpad=5)
    cbar.set_ticks(w2_vals)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.style.use("seaborn-v0_8-white")
    out_path = OUTPUT_DIR / "correlation_plot.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"[plot_correlation]  Saved → {out_path}")


if __name__ == "__main__":
    main()
