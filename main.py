import time
import config
from preprocess import build_model_data
from correlation import run_correlation_analysis
from sensitivity import run_sensitivity_analysis

# Stage toggles
RUN_CORRELATION = True
RUN_SENSITIVITY = True

# Policy parameters used for the sensitivity analysis (Tier weights)
# (1, 1) = Case 1: Baseline FTCM
SENSITIVITY_W1 = 1.0
SENSITIVITY_W2 = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separator(title=""):
    width = 65
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("\n" + "=" * width)


def _hms(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total_start = time.perf_counter()

    print(f"  Output directory : {config.OUTPUT_DIR}")
    print(f"  R_walk_BSS       : {config.R_WALK_BSS} m")
    print(f"  R_walk_PT        : {config.R_WALK_PT} m")
    print(f"  R_ride_dir       : {config.R_RIDE_DIR} m")
    print(f"  R_ride_PT        : {config.R_RIDE_PT} m")
    print(f"  Budget alpha     : {config.ALPHA}")


    # -----------------------------------------------------------------------
    # Stage 1 — Data Preprocessing
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    # P_dir_q corresponds to P_q^dir, S_pt_q corresponds to S_q^pt
    J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, od_df, trips_df = build_model_data()
    t1 = time.perf_counter()
    print(f"\n  Stage 1 complete (Preprocessing)  [{_hms(t1 - t0)}]")


    # Accumulated solve times from both stages
    all_solve_times = []

    # -----------------------------------------------------------------------
    # Stage 2 — Correlation Analysis
    # -----------------------------------------------------------------------
    if RUN_CORRELATION:
        _separator("Stage 2 — Correlation Analysis")
        t0 = time.perf_counter()

        corr_df, spearman_df, corr_solve_times = run_correlation_analysis(
            J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df
        )
        all_solve_times.extend(corr_solve_times)

        corr_path = config.OUTPUT_DIR / "correlation_results.csv"
        spearman_path = config.OUTPUT_DIR / "spearman_rank.csv"

        corr_df.to_csv(corr_path, index=False)
        spearman_df.to_csv(spearman_path, index=False)

        t1 = time.perf_counter()
        print(f"\n  Results saved -> {corr_path}")
        print(f"  Results saved -> {spearman_path}")
        print(f"  Stage 2 complete  [{_hms(t1 - t0)}]")
    else:
        print("\n  Stage 2 (correlation) skipped.")


    # -----------------------------------------------------------------------
    # Stage 3 — Sensitivity analysis
    # -----------------------------------------------------------------------
    if RUN_SENSITIVITY:
        _separator("Stage 3 — Sensitivity Analysis")
        print(f"  Fixed policy weights: w1={SENSITIVITY_W1}, w2={SENSITIVITY_W2}")
        t0 = time.perf_counter()

        sens_df, sens_solve_times = run_sensitivity_analysis(
            J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df,
            w1=SENSITIVITY_W1,
            w2=SENSITIVITY_W2,
        )
        all_solve_times.extend(sens_solve_times)

        sens_path = config.OUTPUT_DIR / "sensitivity_results.csv"
        sens_df.to_csv(sens_path, index=False)

        t1 = time.perf_counter()
        print(f"\n  Results saved → {sens_path}")
        print(f"  Stage 3 complete  [{_hms(t1 - t0)}]")
    else:
        print("\n  Stage 3 (sensitivity) skipped.")


    # -----------------------------------------------------------------------
    # Combined solve-time statistics
    # -----------------------------------------------------------------------
    if all_solve_times:
        import numpy as np
        import pandas as pd

        n   = len(all_solve_times)
        arr = np.array(all_solve_times)
        stats = {
            "n_solves":       n,
            "mean_s":         float(np.mean(arr)),
            "std_s":          float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            "var_s2":         float(np.var(arr, ddof=1)) if n > 1 else 0.0,
            "min_s":          float(arr.min()),
            "max_s":          float(arr.max()),
        }

        _separator("Solve-time Summary (correlation + sensitivity combined)")
        print(
            f"  n solves : {stats['n_solves']}\n"
            f"  mean     : {stats['mean_s']:.3f} s\n"
            f"  std      : {stats['std_s']:.3f} s\n"
            f"  var      : {stats['var_s2']:.4f} s²\n"
            f"  min      : {stats['min_s']:.3f} s\n"
            f"  max      : {stats['max_s']:.3f} s"
        )

        stats_path = config.OUTPUT_DIR / "solve_time_stats.csv"
        pd.DataFrame([stats]).to_csv(stats_path, index=False)
        print(f"\n  Results saved → {stats_path}")


    t_total_end = time.perf_counter()
    _separator()
    print(f"  Pipeline Finished. Total time: {_hms(t_total_end - t_total_start)}")
    _separator()


if __name__ == "__main__":
    main()
