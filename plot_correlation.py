import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["LMRoman10", "serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 13,
    "pgf.rcfonts": False,
})

OUTPUT_DIR = config.OUTPUT_DIR

W_VALS = config.w_vals
W_TICK_LBLS = [f"{w:.2f}" for w in W_VALS]
N = len(W_VALS)
W_IDX = {round(w, 2): i for i, w in enumerate(W_VALS)}

METRICS = [
    ("eta_cov", r"$\eta_{\mathrm{cov}}$", "Blues"),
    ("eta_pt", r"$\eta_{\mathrm{pt}}$", "Oranges"),
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _build_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    mat = np.full((N, N), np.nan)
    for _, row in df.iterrows():
        i = W_IDX.get(round(float(row["w1"]), 2))
        j = W_IDX.get(round(float(row["w2"]), 2))
        if i is not None and j is not None:
            mat[i, j] = row[col]
    return mat


def _text_color(val: float, vmin: float, vmax: float) -> str:
    t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    return "#1a1a1a" if t < 0.6 else "#f5f5f5"


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_path = OUTPUT_DIR / "correlation_results.csv"
    spearman_path = OUTPUT_DIR / "spearman_rank.csv"
    if not corr_path.exists():
        raise FileNotFoundError(f"Not found: {corr_path}")
    df = pd.read_csv(corr_path)
    spearman_df = pd.read_csv(spearman_path) if spearman_path.exists() else None
    return df, spearman_df


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_heatmaps(df: pd.DataFrame, spearman_df: pd.DataFrame | None) -> None:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 5.0),
        constrained_layout=True,
    )

    for ax, (col, label, cmap_name) in zip(axes, METRICS):
        mat = _build_matrix(df, col)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#ececec")

        masked = np.ma.masked_invalid(mat)
        vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))

        im = ax.imshow(
            masked,
            cmap=cmap,
            aspect="equal",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Cell value annotations
        for i in range(N):
            for j in range(N):
                if not np.isnan(mat[i, j]):
                    v = mat[i, j]
                    ax.text(
                        j, i,
                        f"{v:.3f}",
                        ha="center", va="center",
                        fontsize=7.5,
                        color=_text_color(v, vmin, vmax),
                    )

        # Axes labels and ticks
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(W_TICK_LBLS, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(W_TICK_LBLS, fontsize=10)
        ax.set_xlabel(r"$w_2$", fontsize=12, labelpad=6)
        ax.set_ylabel(r"$w_1$", fontsize=12, labelpad=6)
        ax.set_title(label, fontsize=13, pad=10)

        # Thin grid lines between cells
        ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=10)
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()

        # Spearman annotation box
        if spearman_df is not None:
            rows_col = spearman_df[spearman_df["correlation"].str.contains(
                f"w1, {col}"
            )]
            if not rows_col.empty:
                r_val = rows_col.iloc[0]["spearman_r"]
                p_val = rows_col.iloc[0]["p_value"]
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")
                annot = f"$r_s(w_1,\\,{label[1:-1]}\\mid w_2)={r_val:+.3f}${sig}"
                ax.text(
                    0.03, 0.97, annot,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top", ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="#aaaaaa",
                        alpha=0.85,
                    ),
                )

    fig.suptitle(
        r"Policy weight effects on evaluation metrics"
        "\n"
        r"(lower triangle: $w_2 \geq w_1$; upper cells structurally infeasible)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    p = OUTPUT_DIR / "correlation_plot.png"
    plt.savefig(p, bbox_inches="tight", dpi=300)
    print(f"  Saved -> {p}")
    plt.show()


if __name__ == "__main__":
    df, spearman_df = load_data()
    plot_heatmaps(df, spearman_df)
