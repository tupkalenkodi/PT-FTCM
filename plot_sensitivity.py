import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

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

CASE_META = {
    (1.0, 1.0): (1, r"Case 1: Direct-Only $(w_1{=}1.0,\,w_2{=}1.0)$"),
    (1.5, 1.5): (2, r"Case 2: Discounted PT-Hop $(w_1{=}1.5,\,w_2{=}1.5)$"),
    (2.0, 2.0): (3, r"Case 3: Equal Weighting $(w_1{=}2.0,\,w_2{=}2.0)$"),
    (2.5, 2.5): (4, r"Case 4: Prioritized PT-Hop $(w_1{=}2.5,\,w_2{=}2.5)$"),
    (2.0, 3.0): (5, r"Case 5: Full Tiered $(w_1{=}2.0,\,w_2{=}3.0)$"),
}

_cividis = plt.colormaps["cividis"] if hasattr(plt, "colormaps") else plt.get_cmap("cividis")
CASE_COLORS = {
    case_num: _cividis(i * 0.75 / (len(CASE_META) - 1))
    for i, (_, (case_num, _)) in enumerate(
        sorted(CASE_META.items(), key=lambda x: x[1][0])
    )
}

METRICS = {
    "eta_cov": r"$\eta_{\mathrm{cov}}$",
    "eta_pt": r"$\eta_{\mathrm{pt}}$",
}


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_sensitivity_data() -> dict[tuple, dict]:
    """
    Returns a dict keyed by (w1, w2).  Each value is itself a dict with:
      - "nominal": single-row DataFrame (delta=0)
      - "perturbed": DataFrame of delta>0 rows sorted by delta
    The sensitivity_results.csv produced by sensitivity.py is the only
    required input — no separate correlation_results.csv needed.
    """
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


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_sensitivity(cases: dict[tuple, dict]) -> None:
    n_metrics = len(METRICS)
    n_cases = len(cases)

    if n_cases == 0:
        print("  No cases to plot.")
        return

    fig, axes = plt.subplots(
        n_metrics, n_cases,
        figsize=(3.6 * n_cases, 4.0 * n_metrics),
        sharey="row",
        sharex="col",
    )

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

            # Shaded ±1 std band over perturbed realisations
            ax.fill_between(
                deltas,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.25, color=color, linewidth=0,
            )
            ax.plot(
                deltas, means_arr,
                marker="o", markersize=5.5, linewidth=2.0, color=color,
                label="_nolegend_",
            )

            # delta=0 nominal reference drawn from sensitivity data directly
            if not nominal.empty:
                ref_val = float(nominal[f"{metric}_mean"].iloc[0])
                ax.axhline(
                    ref_val,
                    color=color, linewidth=1.4, linestyle="--", alpha=0.85,
                    label=r"$\delta\!=\!0$ (nominal)",
                )
                ax.scatter(
                    [0], [ref_val],
                    color=color, marker="D", s=32, zorder=5,
                    clip_on=False,
                )

            # Axes formatting
            ax.set_xticks([0.0] + deltas)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
            ax.grid(axis="x", linestyle=":", alpha=0.25, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Column title (top row only)
            if row == 0:
                title_str = (
                        f"Case {case_num}\n"
                        r"$w_1\!=\!" + f"{w1}" + r",\,w_2\!=\!" + f"{w2}" + r"$"
                )
                ax.set_title(title_str, fontsize=12, pad=7)

            # x-axis label (bottom row only)
            if row == n_metrics - 1:
                ax.set_xlabel(r"Perturbation $\delta$", fontsize=12, labelpad=5)

            # y-axis label (left column only)
            if col == 0:
                ax.set_ylabel(metric_labels[row], fontsize=13, labelpad=5)

            # Legend (left column only, shown once per row)
            if col == 0 and not nominal.empty:
                ax.legend(
                    fontsize=10,
                    loc="best",
                    frameon=True,
                    framealpha=0.85,
                    edgecolor="#cccccc",
                )

    n_realisations = config.SENSITIVITY_N_REALISATIONS
    fig.suptitle(
        "Sensitivity analysis: metric distributions under demand perturbation\n"
        rf"(shaded band $=\pm 1$ std over {n_realisations} realisations;"
        r" dashed line $=$ nominal $\delta\!=\!0$)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    p = OUTPUT_DIR / "sensitivity_plot.png"
    plt.savefig(p, bbox_inches="tight", dpi=300)
    print(f"  Saved -> {p}")
    plt.show()


if __name__ == "__main__":
    plot_sensitivity(load_sensitivity_data())
