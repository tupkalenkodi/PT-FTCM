import random
import pandas as pd

import config
from model import build_and_solve_pt_ftcm


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

# Draw a random subset of OD pairs at the given fraction and derive all
# model inputs that depend on Q from that subset.
def _filter_by_od_sample(Q_full, F_full, P_dir_q, S_pt_q, M_q, fraction, seed=1):
    rng = random.Random(seed)
    n_sample = max(1, int(round(len(Q_full) * fraction)))
    Q_sample = rng.sample(Q_full, n_sample)
    Q_sample_set = set(Q_sample)

    F_sample = {q: F_full[q]          for q in Q_sample}
    P_dir_sample = {q: P_dir_q.get(q, []) for q in Q_sample}
    S_pt_sample = {q: S_pt_q.get(q, [])  for q in Q_sample}
    M_sample = {
        (q, j): stations
        for (q, j), stations in M_q.items()
        if q in Q_sample_set
    }

    # Recompute J: only stations that are reachable under the sampled demand
    reachable = set()
    for q in Q_sample:
        for j, m in P_dir_sample[q]:
            reachable.add(j)
            reachable.add(m)
        for j in S_pt_sample[q]:
            reachable.add(j)
    for (q, j), stations in M_sample.items():
        reachable.add(j)
        reachable.update(stations)

    J_sample = list(reachable)
    return J_sample, Q_sample, F_sample, P_dir_sample, S_pt_sample, M_sample


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

# For each OD-sample fraction in TIMING_FRACTIONS and each budget level in
# TIMING_ALPHAS, solve the PT-FTCM with fixed policy weights (w1=w2=1,
# i.e. baseline FTCM) and record the Gurobi wall-clock solve time.
def run_timing_analysis(Q, F, P_dir_q, S_pt_q, M_q, w1, w2):
    print(f"  OD fractions : {config.TIMING_FRACTIONS}")
    print(f"  Alpha levels : {config.TIMING_ALPHAS}")

    records = []

    for frac in config.TIMING_FRACTIONS:
        J_s, Q_s, F_s, P_dir_s, S_pt_s, M_s = _filter_by_od_sample(
            Q, F, P_dir_q, S_pt_q, M_q, fraction=frac
        )

        print(f"\n  Fraction {frac:.1f}: |Q|={len(Q_s):,}  |J|={len(J_s)}")

        for alpha in config.TIMING_ALPHAS:
            obj_val, _, solve_time = build_and_solve_pt_ftcm(
                J_s, Q_s, F_s, P_dir_s, S_pt_s, M_s,
                alpha=alpha, w1=w1, w2=w2,
            )

            feasible = obj_val is not None
            print(
                f"    alpha={alpha:.2f}  "
                f"feasible={feasible}  "
                f"solve_time={solve_time:.2f}s"
            )

            records.append({
                "fraction":      frac,
                "n_od_pairs":    len(Q_s),
                "n_stations":    len(J_s),
                "alpha":         alpha,
                "feasible":      feasible,
                "solve_time_s":  solve_time,
            })

    return pd.DataFrame(records)
