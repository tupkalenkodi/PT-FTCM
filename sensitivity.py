import numpy as np
import pandas as pd

import config
from correlation import (
    compute_eta_cov,
    compute_eta_pt,
    _build_station_pt_proximity,
)
from model import build_and_solve_pt_ftcm
from preprocess import precompute_aggregation_structure


def run_sensitivity_analysis(J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df):
    print(f"  Deltas        : {config.SENSITIVITY_DELTAS}")
    print(f"  Realisations  : {config.SENSITIVITY_N_REALISATIONS}")
    print(f"  Cases         : {config.SENSITIVITY_GRID}\n")

    candidate_set = set(J)
    station_near_pt = _build_station_pt_proximity(bss_df, pt_df)

    # Precompute aggregation structure once - Q and coverage sets are fixed
    Q_active = [q for q in Q if P_dir_q.get(q) or S_pt_q.get(q)]
    precomputed = precompute_aggregation_structure(Q_active, P_dir_q, S_pt_q, M_q)

    total_runs = (
            len(config.SENSITIVITY_GRID) *
            len(config.SENSITIVITY_DELTAS) *
            config.SENSITIVITY_N_REALISATIONS
    )
    run_idx = 0

    records = []
    np.random.seed(1)

    for (w1, w2) in config.SENSITIVITY_GRID:
        print(f"\n  {'=' * 55}")
        print(f"  Policy weights: w1={w1}, w2={w2}")
        print(f"  {'=' * 55}")

        print(f"\n  Solving nominal instance (delta=0) ...")
        _, nominal_opened, nominal_time = build_and_solve_pt_ftcm(
            J, Q, F, P_dir_q, S_pt_q, M_q,
            alpha=config.ALPHA, w1=w1, w2=w2,
            precomputed=precomputed,
        )

        if nominal_opened:
            nominal_set = set(nominal_opened)
            eta_cov_nom = compute_eta_cov(trips_df, nominal_set, candidate_set)
            eta_pt_nom = compute_eta_pt(trips_df, nominal_set, station_near_pt)
            print(
                f"  Nominal solved in {nominal_time:.1f}s  "
                f"|S*|={len(nominal_set)}  "
                f"eta_cov={eta_cov_nom:.3f}  eta_pt={eta_pt_nom:.3f}"
            )
            records.append({
                "w1": w1,
                "w2": w2,
                "delta": 0.0,
                "eta_cov_mean": eta_cov_nom,
                "eta_cov_std": 0.0,
                "eta_pt_mean": eta_pt_nom,
                "eta_pt_std": 0.0,
            })
        else:
            print("  Nominal solve failed - skipping delta=0 record.")

        # ------------------------------------------------------------------
        # Perturbed realisations
        # ------------------------------------------------------------------
        for delta in config.SENSITIVITY_DELTAS:
            print(f"\n    delta={delta}")
            eta_cov_vals = []
            eta_pt_vals = []

            for rep in range(config.SENSITIVITY_N_REALISATIONS):
                run_idx += 1
                eps = np.random.uniform(1 - delta, 1 + delta, size=len(Q))
                F_perturbed = {q: F[q] * eps[i] for i, q in enumerate(Q)}

                obj_val, opened, solve_time = build_and_solve_pt_ftcm(
                    J, Q, F_perturbed, P_dir_q, S_pt_q, M_q,
                    alpha=config.ALPHA, w1=w1, w2=w2,
                    precomputed=precomputed,
                )

                if obj_val is None or not opened:
                    print(f"      [{run_idx}/{total_runs}] rep={rep + 1} "
                          f"-> no solution, skipped.")
                    continue

                opened_set = set(opened)
                eta_cov_vals.append(
                    compute_eta_cov(trips_df, opened_set, candidate_set)
                )
                eta_pt_vals.append(
                    compute_eta_pt(trips_df, opened_set, station_near_pt)
                )
                print(f"      [{run_idx}/{total_runs}] rep={rep + 1} "
                      f"completed  ({solve_time:.1f}s)")

            if not eta_cov_vals:
                print(f"    delta={delta:.1f}  -> all realisations infeasible, skipped.")
                continue

            n_solved = len(eta_cov_vals)
            record = {
                "w1": w1,
                "w2": w2,
                "delta": delta,
                "eta_cov_mean": np.mean(eta_cov_vals),
                "eta_cov_std": np.std(eta_cov_vals),
                "eta_pt_mean": np.mean(eta_pt_vals),
                "eta_pt_std": np.std(eta_pt_vals),
            }
            records.append(record)

            print(
                f"\n    delta={delta:.1f}  (n={n_solved:2d})"
                f"    eta_cov={record['eta_cov_mean']:.3f} +/- {record['eta_cov_std']:.3f}"
                f"    eta_pt={record['eta_pt_mean']:.3f} +/- {record['eta_pt_std']:.3f}"
            )

    return pd.DataFrame(records)
