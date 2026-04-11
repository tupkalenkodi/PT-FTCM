import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config
from model import build_and_solve_pt_ftcm
from preprocess import (
    _pairs_within_radius,
    haversine,
    precompute_aggregation_structure,
)


# ---------------------------------------------------------------------------
# STATION-LEVEL PT PROXIMITY
# ---------------------------------------------------------------------------

def _build_station_pt_proximity(bss_df, pt_df):
    bss_coords_rad = np.radians(bss_df[["lat", "lon"]].values)
    pt_coords_rad = np.radians(pt_df[["stop_lat", "stop_lon"]].values)
    neighbours = _pairs_within_radius(bss_coords_rad, pt_coords_rad, config.R_WALK_PT)
    return {
        row["station_id"]: bool(neighbours[idx])
        for idx, (_, row) in enumerate(bss_df.iterrows())
    }


def _build_station_pt_min_dist(bss_df, pt_df):
    pt_coords = pt_df[["stop_lat", "stop_lon"]].values
    return {
        row["station_id"]: min(
            haversine(row["lon"], row["lat"], pt[1], pt[0])
            for pt in pt_coords
        )
        for _, row in bss_df.iterrows()
    }


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_eta_trip(trips_df, opened_set):
    covered = trips_df[
        trips_df["start_station_id"].isin(opened_set) &
        trips_df["end_station_id"].isin(opened_set)
        ]
    return float(len(covered) / len(trips_df))


def compute_eta_pt(trips_df, opened_set, station_near_pt):
    t_star = trips_df[
        trips_df["start_station_id"].isin(opened_set) &
        trips_df["end_station_id"].isin(opened_set)
        ]
    near_pt = (
            t_star["start_station_id"].map(station_near_pt) |
            t_star["end_station_id"].map(station_near_pt)
    )
    return float(near_pt.sum() / len(t_star))


def compute_eta_proximity(opened_set, station_min_dist_pt):
    dists = [station_min_dist_pt[j] for j in opened_set]
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# PARTIAL SPEARMAN CORRELATION
# ---------------------------------------------------------------------------

def _partial_spearman(df, x_col, y_col, control_col):
    sub = df[[x_col, y_col, control_col]].dropna()
    n = len(sub)

    rx = sub[x_col].rank()
    ry = sub[y_col].rank()
    rc = sub[control_col].rank()

    def ols_residuals(dependent, predictor):
        c = predictor.values
        ones = np.ones(n)
        X = np.column_stack([ones, c])
        beta, *_ = np.linalg.lstsq(X, dependent.values, rcond=None)
        return dependent.values - X @ beta

    ex = ols_residuals(rx, rc)
    ey = ols_residuals(ry, rc)
    r, p = pearsonr(ex, ey)
    return float(r), float(p)


def _spearman_test(results_df):
    df = results_df.dropna(subset=["eta_trip", "eta_pt", "eta_proximity"])

    tests = [
        ("w1", "eta_trip", "w2"),
        ("w1", "eta_pt", "w2"),
        ("w1", "eta_proximity", "w2"),
        ("w2", "eta_trip", "w1"),
        ("w2", "eta_pt", "w1"),
        ("w2", "eta_proximity", "w1"),
    ]

    rows = []
    for x_col, y_col, control_col in tests:
        r, p = _partial_spearman(df, x_col, y_col, control_col)
        rows.append({
            "correlation": f"r_s({x_col}, {y_col} | {control_col})",
            "spearman_r": round(r, 4),
            "p_value": round(p, 4),
            "sign": ("+" if r > 0 else "-"),
        })

    summary = pd.DataFrame(rows)
    print("\n  --- Partial Spearman Correlation Test ---")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# MAIN EVALUATION LOOP
# ---------------------------------------------------------------------------

def run_correlation_analysis(J, Q, F, P_dir_q, S_pt_q, M_q, bss_df, pt_df, trips_df):
    station_near_pt = _build_station_pt_proximity(bss_df, pt_df)
    station_min_dist_pt = _build_station_pt_min_dist(bss_df, pt_df)
    print(f"  Stations within R_PT of a PT stop : {sum(station_near_pt.values())}")
    print(f"  Avg min-distance to PT (all)      : {np.mean(list(station_min_dist_pt.values())):.1f} m")
    print(f"  Trip log                          : {len(trips_df):,} records")

    # Precompute aggregation structure once - Q and coverage sets are fixed
    Q_active = [q for q in Q if P_dir_q.get(q) or S_pt_q.get(q)]
    precomputed = precompute_aggregation_structure(Q_active, P_dir_q, S_pt_q, M_q)

    records = []
    for w1, w2 in config.CORRELATION_GRID:
        print(f"\n  Policy weights: w1={w1}, w2={w2}")
        obj_val, opened, solve_time = build_and_solve_pt_ftcm(
            J, Q, F, P_dir_q, S_pt_q, M_q,
            alpha=config.ALPHA, w1=w1, w2=w2,
            precomputed=precomputed,
        )

        if obj_val is None:
            print("    INFEASIBLE - skipped.")
            continue

        opened_set = set(opened)
        eta_trip = compute_eta_trip(trips_df, opened_set)
        eta_pt = compute_eta_pt(trips_df, opened_set, station_near_pt)
        eta_dist = compute_eta_proximity(opened_set, station_min_dist_pt)

        print(
            f"    |S*|={len(opened)};  "
            f"    eta_trip={eta_trip:.3f};  "
            f"    eta_adj={eta_pt:.3f};  "
            f"    eta_proximity={eta_dist:.1f} m;  "
            f"    solve_time={solve_time:.2f}s"
        )
        records.append({
            "w1": w1,
            "w2": w2,
            "n_stations_opened": len(opened),
            "eta_trip": eta_trip,
            "eta_pt": eta_pt,
            "eta_proximity": eta_dist,
        })

    results_df = pd.DataFrame(records)
    spearman_df = _spearman_test(results_df)

    return results_df, spearman_df
