import random
import pandas as pd

import config
from model import build_and_solve_pt_ftcm
from preprocess import precompute_aggregation_structure


def _filter_by_od_sample(Q_full, F_full, P_dir_q, S_pt_q, M_q, fraction, seed=1):
    rng = random.Random(seed)
    n_sample = max(1, int(round(len(Q_full) * fraction)))
    Q_sample = rng.sample(Q_full, n_sample)
    Q_sample_set = set(Q_sample)

    F_sample = {q: F_full[q] for q in Q_sample}
    P_dir_sample = {q: P_dir_q.get(q, []) for q in Q_sample}
    S_pt_sample = {q: S_pt_q.get(q, []) for q in Q_sample}
    M_sample = {
        (q, j): stations
        for (q, j), stations in M_q.items()
        if q in Q_sample_set
    }

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


def run_feasibility_analysis(Q, F, P_dir_q, S_pt_q, M_q):
    print(f"  OD fractions : {config.TIMING_FRACTIONS}")
    print(f"  Alpha level  : {config.ALPHA}")

    records = []

    for frac in config.TIMING_FRACTIONS:
        J_s, Q_s, F_s, P_dir_s, S_pt_s, M_s = _filter_by_od_sample(
            Q, F, P_dir_q, S_pt_q, M_q, fraction=frac
        )
        print(f"\n  {'-' * 55}")
        print(f"  Fraction {frac:.1f}  |  |Q|={len(Q_s):,}  |J|={len(J_s)}")
        print(f"  {'-' * 55}")

        # Precompute aggregation structure once per fraction — shared across all (w1, w2)
        Q_active = [q for q in Q_s if P_dir_s.get(q) or S_pt_s.get(q)]
        precomputed = precompute_aggregation_structure(Q_active, P_dir_s, S_pt_s, M_s)

        for (w1, w2, _) in config.TIMING_GRID:
            print(f"\n    Policy weights: w1={w1}, w2={w2}")
            obj_val, _, solve_time = build_and_solve_pt_ftcm(
                J_s, Q_s, F_s, P_dir_s, S_pt_s, M_s,
                alpha=config.ALPHA, w1=w1, w2=w2,
                precomputed=precomputed,
            )
            print(f"    solve_time={solve_time:.2f}s")

            records.append({
                "w1": w1,
                "w2": w2,
                "fraction": frac,
                "n_od_pairs": len(Q_s),
                "n_stations": len(J_s),
                "alpha": config.ALPHA,
                "solve_time_s": solve_time,
            })

    return pd.DataFrame(records)
