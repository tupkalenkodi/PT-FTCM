import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import config
from plot_correlation import _RC, _DATA_LW, _REF_LW, _MARKER_SZ


# Shared style
mpl.rcParams.update(_RC)
SEQ_CMAP = "viridis_r"

OUTPUT_DIR = config.OUTPUT_DIR

CASE_META = {
    (1.0, 1.0): (1, r"Case 1: Direct-Only $(w_1{=}1.0,\,w_2{=}1.0)$"),
    (2.0, 2.0): (3, r"Case 3: Equal Weighting $(w_1{=}2.0,\,w_2{=}2.0)$"),
    (2.0, 3.0): (5, r"Case 5: Full Tiered $(w_1{=}2.0,\,w_2{=}3.0)$"),
}

# Build colors from viridis_r
cmap = plt.get_cmap(SEQ_CMAP)
case_numbers = sorted([meta[0] for meta in CASE_META.values()])
n_cases = len(case_numbers)

CASE_COLORS = {
    case_num: cmap(0.2 + 0.6 * i / (n_cases - 1) if n_cases > 1 else 0.5)
    for i, case_num in enumerate(case_numbers)
}

METRICS = {
    "eta_cov": r"$\eta_{\mathrm{cov}}$",
    "eta_pt": r"$\eta_{\mathrm{pt}}$",
}


def load_sensitivity_data() -> dict[tuple, dict]:
    path = OUTPUT_DIR / "sensitivity_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"sensitivity_results.csv not found at {path}")

    df = pd.read_csv(path)

    cases: dict[tuple, dict] = {}
    for (w1, w2) in CASE_META:
        sub = df[(df["w1"] == w1) & (df["w2"] == w2)].copy()
        if sub.empty:
            print(f"  WARNING: no rows for w1={w1}, w2={w2} — skipping.")
            continue

        nominal = sub[sub["delta"] == 0.0].reset_index(drop=True)
        perturbed = sub[sub["delta"] > 0.0].sort_values("delta").reset_index(drop=True)

        if perturbed.empty:
            print(f"  WARNING: no perturbed rows for w1={w1}, w2={w2} — skipping.")
            continue

        cases[(w1, w2)] = {"nominal": nominal, "perturbed": perturbed}

    return cases


def plot_sensitivity(cases: dict[tuple, dict]) -> None:
    n_metrics = len(METRICS)
    n_cases = len(cases)

    if n_cases == 0:
        print("  No cases to plot.")
        return

    fig, axes = plt.subplots(
        n_metrics, n_cases,
        figsize=(3.6 * n_cases, 2.4 * n_metrics),
        sharey="row", sharex="col",
    )

    # Normalize axes to always be 2-D
    if n_cases == 1 and n_metrics == 1:
        axes = np.array([[axes]])
    elif n_cases == 1:
        axes = np.array(axes).reshape(n_metrics, 1)
    elif n_metrics == 1:
        axes = np.array(axes).reshape(1, n_cases)

    metric_keys = list(METRICS.keys())
    metric_labels = list(METRICS.values())
    sorted_cases = sorted(cases.items(), key=lambda kv: CASE_META[kv[0]][0])

    for col, ((w1, w2), data) in enumerate(sorted_cases):
        case_num, _ = CASE_META[(w1, w2)]
        color = CASE_COLORS[case_num]

        nominal = data["nominal"]
        perturbed = data["perturbed"]
        deltas = perturbed["delta"].tolist()

        for row, metric in enumerate(metric_keys):
            ax = axes[row][col]

            means_arr = perturbed[f"{metric}_mean"].to_numpy()
            stds_arr = perturbed[f"{metric}_std"].to_numpy()

            # Shaded +-1 std band
            ax.fill_between(
                deltas,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.25,
                color=color,
                linewidth=0,
            )

            # Mean line
            ax.plot(
                deltas,
                means_arr,
                marker="o",
                markersize=_MARKER_SZ,
                linewidth=_DATA_LW,
                color=color,
            )

            # Nominal reference
            if not nominal.empty:
                ref_val = float(nominal[f"{metric}_mean"].iloc[0])

                ax.axhline(
                    ref_val,
                    color="black",
                    linewidth=_REF_LW,
                    linestyle="--",
                    alpha=0.9,
                    label=r"$\delta\!=\!0$ (nominal)",
                )

                ax.scatter(
                    [0],
                    [ref_val],
                    color="black",
                    marker="D",
                    s=16,
                    zorder=5,
                    clip_on=False,
                )

            # Axes formatting
            ax.set_xticks([0.0] + deltas)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", labelsize=10)

            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(0.7)

            # Column title
            if row == 0:
                ax.set_title(
                    r"Case " + f"{case_num}" + r" ($w_1\!=\!" + f"{w1}" + r",\,w_2\!=\!" + f"{w2}" + r"$)",
                    fontsize=11,
                    pad=4,
                )

            # X label
            if row == n_metrics - 1:
                ax.set_xlabel(
                    r"Perturbation $\delta$",
                    fontsize=11,
                    labelpad=2,
                )

            # Y label + legend
            if col == 0:
                ax.set_ylabel(
                    metric_labels[row],
                    fontsize=11,
                    labelpad=2,
                )

                # ONLY draw the legend if it's the first row (row == 0)
                if not nominal.empty and row == 0:
                    ax.legend(
                        loc="best",
                        frameon=True,
                        framealpha=0.88,
                        edgecolor="#CCCCCC",
                        fontsize=9,
                    )

    plt.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.4)
    plt.style.use("seaborn-v0_8-white")
    out_path = OUTPUT_DIR / "sensitivity_plot.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=600)
    print(f"[plot_sensitivity]  Saved -> {out_path}")

    plt.show()


if __name__ == "__main__":
    plot_sensitivity(load_sensitivity_data())