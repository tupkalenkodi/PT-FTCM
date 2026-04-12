import time

import config
from correlation import run_correlation_analysis
from feasibility import run_feasibility_analysis
from preprocess import build_model_data
from sensitivity import run_sensitivity_analysis


RUN_FEASIBILITY = True
RUN_CORRELATION = False
RUN_SENSITIVITY = False


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


def main():
    t_total_start = time.perf_counter()
    _separator("Setup")
    print(f"  Output directory : {config.OUTPUT_DIR}")
    print(f"  R_walk_BSS       : {config.R_WALK_BSS} m")
    print(f"  R_walk_PT        : {config.R_WALK_PT} m")
    print(f"  R_ride_dir       : {config.R_RIDE_DIR} m")
    print(f"  R_ride_PT        : {config.R_RIDE_PT} m")
    print(f"  T_PT_max_min     : {config.T_PT_MAX_MIN} min")
    print(f"  Budget alpha     : {config.ALPHA}")

    # Stage 1 - Data Preprocessing
    _separator("Stage 1 - Preprocessing")
    t0 = time.perf_counter()
    J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, od_df, trips_df = build_model_data()
    _separator(f"Stage 1 complete  [{_hms(time.perf_counter() - t0)}]")

    # Stage 2 - Time-Feasibility Analysis
    if RUN_FEASIBILITY:
        _separator("Stage 2 - Time-Feasibility Analysis")
        t0 = time.perf_counter()
        timing_df = run_feasibility_analysis(Q, F, P_dir_q, S_pt_q, M_q)
        timing_df.to_csv(config.OUTPUT_DIR / "timing_results.csv", index=False)
        _separator(f"Stage 2 complete  [{_hms(time.perf_counter() - t0)}]")
    else:
        _separator("Stage 2 (timing) skipped.")

    # Stage 3 - Correlation Analysis
    if RUN_CORRELATION:
        _separator("Stage 3 - Correlation Analysis")
        t0 = time.perf_counter()
        corr_df, spearman_df = run_correlation_analysis(J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df)
        corr_path = config.OUTPUT_DIR / "correlation_results.csv"
        spearman_path = config.OUTPUT_DIR / "spearman_rank.csv"
        corr_df.to_csv(corr_path, index=False)
        spearman_df.to_csv(spearman_path, index=False)
        print(f"\n  Results saved -> {corr_path},  {spearman_path}")

        _separator(f"Stage 3 complete  [{_hms(time.perf_counter() - t0)}]")
    else:
        _separator("Stage 3 (correlation) skipped.")

    # Stage 4 - Sensitivity Analysis
    if RUN_SENSITIVITY:
        _separator("Stage 4 - Sensitivity Analysis")
        t0 = time.perf_counter()
        sens_df = run_sensitivity_analysis(J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df)
        sensitivity_path = config.OUTPUT_DIR / "sensitivity_results.csv"
        sens_df.to_csv(sensitivity_path, index=False)
        print(f"\n  Results saved -> {sensitivity_path}")

        _separator(f"Stage 4 complete  [{_hms(time.perf_counter() - t0)}]")
    else:
        _separator("Stage 4 (sensitivity) skipped.")

    _separator()
    _separator()
    _separator()
    print(f"  Pipeline finished.  Total time: {_hms(time.perf_counter() - t_total_start)}")


if __name__ == "__main__":
    main()
