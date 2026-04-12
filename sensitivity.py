import numpy as np
import pandas as pd

import config
from correlation import (
    compute_eta_trip,
    compute_eta_pt,
    compute_eta_proximity,
    _build_station_pt_proximity,
    _build_station_pt_min_dist,
)
from model import build_and_solve_pt_ftcm
from preprocess import precompute_aggregation_structure


def run_sensitivity_analysis(J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df):
    print(f"  Deltas        : {config.SENSITIVITY_DELTAS}")
    print(f"  Realisations  : {config.SENSITIVITY_N_REALISATIONS}\n")

    station_near_pt = _build_station_pt_proximity(bss_df, pt_df)
    station_min_dist_pt = _build_station_pt_min_dist(bss_df, pt_df)

    # Precompute aggregation structure once - Q and coverage sets are fixed
    Q_active = [q for q in Q if P_dir_q.get(q) or S_pt_q.get(q)]
    precomputed = precompute_aggregation_structure(Q_active, P_dir_q, S_pt_q, M_q)

    records = []
    np.random.seed(1)
    for w1, w2 in config.TIMING_GRID:
        print(f"\n  {'-' * 55}")
        print(f"  Policy weights: w1={w1}, w2={w2}")
        print(f"  {'-' * 55}")

        for delta in config.SENSITIVITY_DELTAS:

            print(f"\n    delta={delta}")
            eta_trip_vals = []
            eta_pt_vals = []
            eta_prox_vals = []
            solve_time_vals = []

            for rep in range(config.SENSITIVITY_N_REALISATIONS):
                eps = np.random.uniform(1 - delta, 1 + delta, size=len(Q))
                F_perturbed = {q: F[q] * eps[i] for i, q in enumerate(Q)}

                obj_val, opened, solve_time = build_and_solve_pt_ftcm(
                    J, Q, F_perturbed, P_dir_q, S_pt_q, M_q,
                    alpha=config.ALPHA, w1=w1, w2=w2,
                    precomputed=precomputed,
                )

                if obj_val is None or not opened:
                    print(f"    delta={delta:.1f}  rep={rep + 1} -> no solution, skipped.")
                    continue

                opened_set = set(opened)
                solve_time_vals.append(solve_time)
                eta_trip_vals.append(compute_eta_trip(trips_df, opened_set))
                eta_pt_vals.append(compute_eta_pt(trips_df, opened_set, station_near_pt))
                eta_prox_vals.append(compute_eta_proximity(opened_set, station_min_dist_pt))

                print(f"    {rep} completed")

            if not eta_trip_vals:
                print(f"    delta={delta:.1f}  -> all realisations infeasible, skipped.")
                continue

            n_solved = len(eta_trip_vals)
            record = {
                "w1": w1,
                "w2": w2,
                "delta": delta,
                "eta_trip_mean": np.mean(eta_trip_vals),
                "eta_trip_std": np.std(eta_trip_vals),
                "eta_pt_mean": np.mean(eta_pt_vals),
                "eta_pt_std": np.std(eta_pt_vals),
                "eta_proximity_mean": np.mean(eta_prox_vals),
                "eta_proximity_std": np.std(eta_prox_vals),
            }
            records.append(record)

            print(
                f"\n    delta={delta:.1f}  (n={n_solved:2d})"
                f"    eta_trip={record['eta_trip_mean']:.3f} +/- {record['eta_trip_std']:.3f}"
                f"    eta_pt={record['eta_pt_mean']:.3f} +/- {record['eta_pt_std']:.3f}"
                f"    eta_proximity={record['eta_proximity_mean']:.1f} +/- {record['eta_proximity_std']:.1f} m"
            )

    return pd.DataFrame(records)
