import time
import config
from config import MINIMAL_GRID
from preprocess import build_model_data
from correlation import run_correlation_analysis
from sensitivity import run_sensitivity_analysis
from time_feasibility import run_timing_analysis


# ---------------------------------------------------------------------------
# Stage toggles
# ---------------------------------------------------------------------------
RUN_TIMING = False
RUN_CORRELATION = True
RUN_SENSITIVITY = False


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
    m, s   = divmod(rem, 60)
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
    _separator("Stage 1 - Preprocessing")
    t0 = time.perf_counter()

    # P_dir_q corresponds to P_q^dir, S_pt_q corresponds to S_q^pt
    J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, od_df, trips_df = build_model_data()

    t1 = time.perf_counter()
    print(f"\n  Stage 1 complete  [{_hms(t1 - t0)}]")


    # -----------------------------------------------------------------------
    # Stage 2 - Time-Feasibility Analysis
    # -----------------------------------------------------------------------
    if RUN_TIMING:
        _separator("Stage 2 - Time-Feasibility Analysis")
        t0 = time.perf_counter()
        for w1, w2, case_num in MINIMAL_GRID:
            print(f"  Fixed policy weights: w1={w1}, w2={w2}")

            timing_df = run_timing_analysis(Q, F, P_dir_q, S_pt_q, M_q, w1, w2)

            timing_path = config.OUTPUT_DIR / f"timing_results_case{case_num}.csv"
            timing_df.to_csv(timing_path, index=False)

        t1 = time.perf_counter()
        print(f"  Stage 2 complete  [{_hms(t1 - t0)}]")
    else:
        print("\n  Stage 2 (timing) skipped.")


    # -----------------------------------------------------------------------
    # Stage 3 — Correlation Analysis
    # -----------------------------------------------------------------------
    if RUN_CORRELATION:
        _separator("Stage 3 - Correlation Analysis")
        t0 = time.perf_counter()

        corr_df, spearman_df, corr_solve_times = run_correlation_analysis(
            J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df
        )

        corr_path = config.OUTPUT_DIR / "correlation_results.csv"
        spearman_path = config.OUTPUT_DIR / "spearman_rank.csv"
        corr_df.to_csv(corr_path, index=False)
        spearman_df.to_csv(spearman_path, index=False)

        t1 = time.perf_counter()
        print(f"\n  Results saved -> {corr_path}")
        print(f"  Results saved -> {spearman_path}")
        print(f"  Stage 3 complete  [{_hms(t1 - t0)}]")
    else:
        print("\n  Stage 3 (correlation) skipped.")


    # -----------------------------------------------------------------------
    # Stage 4 - Sensitivity Analysis (demand perturbation)
    # -----------------------------------------------------------------------
    if RUN_SENSITIVITY:
        _separator("Stage 4 - Sensitivity Analysis")
        t0 = time.perf_counter()

        for w1, w2, case_num in MINIMAL_GRID:
            print(f"  Fixed policy weights: w1={w1}, w2={w2}")

            sens_df, sens_solve_times = run_sensitivity_analysis(
                J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df,
                w1=w1, w2=w2,
            )

            sens_path = config.OUTPUT_DIR / f"sensitivity_results_case{case_num}.csv"
            sens_df.to_csv(sens_path, index=False)

        t1 = time.perf_counter()
        print(f"  Stage 4 complete  [{_hms(t1 - t0)}]")
    else:
        print("\n  Stage 4 (sensitivity) skipped.")


    t_total_end = time.perf_counter()
    _separator()
    print(f"  Pipeline finished.  Total time: {_hms(t_total_end - t_total_start)}")
    _separator()


if __name__ == "__main__":
    main()
