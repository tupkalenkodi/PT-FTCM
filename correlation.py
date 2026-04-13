import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config
from model import build_and_solve_pt_ftcm
from preprocess import (
    _pairs_within_radius,
    precompute_aggregation_structure,
)


# ---------------------------------------------------------------------------
# STATION-LEVEL PT PROXIMITY
# ---------------------------------------------------------------------------

def _build_station_pt_proximity(bss_df: pd.DataFrame, pt_df: pd.DataFrame) -> dict[str, bool]:
    bss_coords_rad = np.radians(bss_df[["lat", "lon"]].values)
    pt_coords_rad = np.radians(pt_df[["stop_lat", "stop_lon"]].values)
    neighbours = _pairs_within_radius(bss_coords_rad, pt_coords_rad, config.R_WALK_PT)
    return {
        row["station_id"]: bool(neighbours[idx])
        for idx, (_, row) in enumerate(bss_df.iterrows())
    }


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_eta_cov(trips_df: pd.DataFrame, opened_set: set[str], candidate_set: set[str]) -> float:
    eligible = trips_df[
        trips_df["start_station_id"].isin(candidate_set) &
        trips_df["end_station_id"].isin(candidate_set)
        ]
    if len(eligible) == 0:
        return 0.0
    covered = eligible[
        eligible["start_station_id"].isin(opened_set) &
        eligible["end_station_id"].isin(opened_set)
        ]
    return float(len(covered) / len(eligible))


def compute_eta_pt(trips_df: pd.DataFrame, opened_set: set[str], station_near_pt: dict[str, bool]) -> float:
    t_star = trips_df[
        trips_df["start_station_id"].isin(opened_set) &
        trips_df["end_station_id"].isin(opened_set)
        ]
    if len(t_star) == 0:
        return 0.0
    near_pt = (
            t_star["start_station_id"].map(station_near_pt) |
            t_star["end_station_id"].map(station_near_pt)
    )
    return float(near_pt.sum() / len(t_star))


# ---------------------------------------------------------------------------
# PARTIAL SPEARMAN CORRELATION
# ---------------------------------------------------------------------------

def _partial_spearman(df: pd.DataFrame, x_col: str, y_col: str, control_col: str) -> tuple[float, float]:
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


def _spearman_test(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.dropna(subset=["eta_cov", "eta_pt"])

    tests = [
        ("w1", "eta_cov", "w2"),
        ("w1", "eta_pt", "w2"),
        ("w2", "eta_cov", "w1"),
        ("w2", "eta_pt", "w1"),
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

def run_correlation_analysis(
    J: list[str],
    Q: list[tuple[str, str]],
    F: dict[tuple[str, str], float],
    P_dir_q: dict[tuple[str, str], list[tuple[str, str]]],
    S_pt_q: dict[tuple[str, str], list[str]],
    M_q: dict[tuple[tuple[str, str], str], list[str]],
    bss_df: pd.DataFrame,
    pt_df: pd.DataFrame,
    trips_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_set = set(J)

    station_near_pt = _build_station_pt_proximity(bss_df, pt_df)

    eligible = trips_df[
        trips_df["start_station_id"].isin(candidate_set) &
        trips_df["end_station_id"].isin(candidate_set)
        ]
    print(f"  Stations within R_PT of a PT stop  : {sum(station_near_pt.values())}")
    print(f"  Trip log (all records)             : {len(trips_df):,}")
    print(f"  Candidate-eligible trips           : {len(eligible):,}  "
          f"({100 * len(eligible) / len(trips_df):.1f}% of total)")

    # Precompute aggregation structure once - Q and coverage sets are fixed
    Q_active = [q for q in Q if P_dir_q.get(q) or S_pt_q.get(q)]
    precomputed = precompute_aggregation_structure(Q_active, P_dir_q, S_pt_q, M_q)

    records = []
    for run_idx, (w1, w2) in enumerate(config.CORRELATION_GRID):
        print(f"\n  [{run_idx + 1}/{len(config.CORRELATION_GRID)}] "
              f"Policy weights: w1={w1}, w2={w2}")
        obj_val, opened, solve_time = build_and_solve_pt_ftcm(
            J, Q, F, P_dir_q, S_pt_q, M_q,
            alpha=config.ALPHA, w1=w1, w2=w2,
            precomputed=precomputed,
        )

        if obj_val is None:
            print("    INFEASIBLE - skipped.")
            continue

        opened_set = set(opened)
        eta_cov = compute_eta_cov(trips_df, opened_set, candidate_set)
        eta_pt = compute_eta_pt(trips_df, opened_set, station_near_pt)

        print(
            f"    |S*|={len(opened)};  "
            f"eta_cov={eta_cov:.3f};  "
            f"eta_pt={eta_pt:.3f};  "
            f"solve_time={solve_time:.2f}s"
        )
        records.append({
            "w1": w1,
            "w2": w2,
            "|S*|": len(opened),
            "eta_cov": eta_cov,
            "eta_pt": eta_pt,
            "solve_time": solve_time,
        })

    results_df = pd.DataFrame(records)
    spearman_df = _spearman_test(results_df)

    return results_df, spearman_df
