import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config


def check_correlation_assumptions(results_df, x_col, y_col, control_col):
    """
    Generates diagnostic plots to validate the assumptions behind
    the Partial Spearman Correlation analysis.
    """
    # Create a sub-dataframe to handle NaNs consistently with correlation.py
    sub = results_df[[x_col, y_col, control_col]].dropna()
    if len(sub) == 0:
        print(f"Skipping diagnostic for {x_col} vs {y_col}: No valid data.")
        return

    plt.figure(figsize=(15, 5))

    # 1. Raw Monotonicity: Check the general trend of the data
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=sub, x=x_col, y=y_col, hue=control_col, palette="viridis")
    plt.title(f"1. Monotonicity: {x_col} vs {y_col}\n(Color: {control_col})")

    # 2. Rank Linearity: Spearman assumes a linear relationship between ranks
    plt.subplot(1, 3, 2)
    plt.scatter(sub[x_col].rank(), sub[y_col].rank(), alpha=0.6)
    plt.title(f"2. Rank-Rank Relationship\nrank({x_col}) vs rank({y_col})")
    plt.xlabel(f"rank({x_col})")
    plt.ylabel(f"rank({y_col})")

    # 3. Partial Residuals: Visualizing the correlation after removing control_col
    def get_res(dep, pred):
        # Logic mirroring the _partial_spearman function in correlation.py
        ones = np.ones(len(dep))
        X = np.column_stack([ones, pred.values])
        beta, *_ = np.linalg.lstsq(X, dep.values, rcond=None)
        return dep.values - X @ beta

    ex = get_res(sub[x_col].rank(), sub[control_col].rank())
    ey = get_res(sub[y_col].rank(), sub[control_col].rank())

    plt.subplot(1, 3, 3)
    plt.scatter(ex, ey, color='red', alpha=0.5)
    # Add a regression line to see the partial correlation trend
    m, b = np.polyfit(ex, ey, 1)
    plt.plot(ex, m * ex + b, color='black', linestyle='--', linewidth=1)

    plt.title(f"3. Partial Residuals\nControlling for {control_col}")
    plt.xlabel(f"Resid: rank({x_col}) ~ rank({control_col})")
    plt.ylabel(f"Resid: rank({y_col}) ~ rank({control_col})")

    plt.tight_layout()

    # Save to the outputs directory defined in config
    save_path = config.OUTPUT_DIR / f"diagnostic_{x_col}_{y_col}_ctrl_{control_col}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Diagnostic plot saved to: {save_path}")


def generate_all_diagnostics(results_df: pd.DataFrame):
    """
    Helper to run the diagnostic suite for all primary
    policy-metric relationships.
    """
    print("\n  --- Generating Diagnostic Correlation Plots ---")

    # Matching the tests defined in correlation.py
    tests = [
        ("w1", "eta_cov", "w2"),
        ("w1", "eta_pt", "w2"),
        ("w2", "eta_cov", "w1"),
        ("w2", "eta_pt", "w1"),
    ]

    for x, y, ctrl in tests:
        check_correlation_assumptions(results_df, x, y, ctrl)


if __name__ == "__main__":
    results_path = config.OUTPUT_DIR / "correlation_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        generate_all_diagnostics(df)
