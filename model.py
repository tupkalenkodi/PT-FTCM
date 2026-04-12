import math
import gurobipy as gp
from gurobipy import GRB


def build_and_solve_pt_ftcm(J, Q, F, P_dir_q, S_pt_q, M_q, alpha, w1, w2,
                            precomputed=None, skip=False):
    assert w1 >= 1.0, "w_1 must be >= 1"
    assert w2 >= w1, "w_2 must be >= w_1"

    # Prune OD pairs that can never be covered
    Q_active = [q for q in Q if P_dir_q.get(q) or S_pt_q.get(q)]
    n_pruned = len(Q) - len(Q_active)

    # Aggregate coverage-equivalent OD pairs
    if precomputed is None:
        from preprocess import precompute_aggregation_structure, aggregate_flows
        struct = precompute_aggregation_structure(Q_active, P_dir_q, S_pt_q, M_q)
        Q_agg, P_agg, S_agg, M_agg, sig_to_rep, q_to_sig, n_merged = struct
        F_agg = aggregate_flows(F, Q_active, sig_to_rep, q_to_sig)
    else:
        from preprocess import aggregate_flows
        _, Q_agg, P_agg, S_agg, M_agg, sig_to_rep, q_to_sig, n_merged = precomputed
        F_agg = aggregate_flows(F, Q_active, sig_to_rep, q_to_sig)

    print(
        f"    [model] |Q|={len(Q)} -> pruned {n_pruned} -> merged {n_merged} "
        f"-> |Q_agg|={len(Q_agg)}"
    )

    # Collect unique (j, m) pairs for shared Tier-1 auxiliaries
    unique_t1_pairs = set()
    for q in Q_agg:
        for jm in P_agg.get(q, []):
            unique_t1_pairs.add(jm)

    # Collect unique (j, frozenset(M)) signatures for shared Tier-2 auxiliaries
    t2_sig_to_mlist: dict[tuple, list] = {}
    qj_to_t2sig: dict[tuple, tuple] = {}

    for q in Q_agg:
        for j in S_agg.get(q, []):
            m_list = M_agg.get((q, j), [])
            m_sig = frozenset(m_list)
            key = (j, m_sig)
            if key not in t2_sig_to_mlist:
                t2_sig_to_mlist[key] = m_list
            qj_to_t2sig[(q, j)] = key

    # Build model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    mdl = gp.Model("PT_FTCM", env=env)
    mdl.setParam("OutputFlag", 1)
    mdl.setParam("MIPGap", 0.01)
    mdl.setParam("MIPFocus", 1)
    mdl.setParam("ImproveStartGap", 0.05)
    mdl.setParam("SubMIPNodes", 500)
    mdl.setParam("Heuristics", 0.5)
    mdl.setParam("RINS", 10)
    mdl.setParam("Cuts", 1)
    mdl.setParam("MIRCuts", 1)
    mdl.setParam("RLTCuts", 2)
    mdl.setParam("Presolve", 2)
    mdl.setParam("Aggregate", 2)
    mdl.setParam("Method", 2)
    mdl.setParam("Crossover", 1)
    mdl.setParam("Threads", 10)

    x = mdl.addVars(J, vtype=GRB.BINARY, name="x")

    # Set branch priorities by flow weight
    station_flow_weight = {j: 0.0 for j in J}
    for q in Q_agg:
        fq = F_agg[q]
        for (jj, mm) in P_agg.get(q, []):
            station_flow_weight[jj] = station_flow_weight.get(jj, 0.0) + fq
            station_flow_weight[mm] = station_flow_weight.get(mm, 0.0) + fq
        for jj in S_agg.get(q, []):
            station_flow_weight[jj] = station_flow_weight.get(jj, 0.0) + fq

    max_w = max(station_flow_weight.values()) if station_flow_weight else 1.0
    for j in J:
        x[j].BranchPriority = int(10 * station_flow_weight.get(j, 0.0) / max_w)

    y1 = mdl.addVars(Q_agg, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y1")
    y2 = mdl.addVars(Q_agg, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y2")
    y3 = mdl.addVars(Q_agg, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y3")

    # Tier-1 auxiliary variables: b[j,m] = x[j] AND x[m]
    b = mdl.addVars(list(unique_t1_pairs), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="b")
    for (j, m) in unique_t1_pairs:
        mdl.addConstr(b[j, m] <= x[j])
        mdl.addConstr(b[j, m] <= x[m])
        mdl.addConstr(b[j, m] >= x[j] + x[m] - 1)

    # Tier-2 auxiliary variables: c[j, m_sig] = x[j] AND (any x[i] in M)
    c = mdl.addVars(list(t2_sig_to_mlist.keys()), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="c")
    for (j, m_sig), m_list in t2_sig_to_mlist.items():
        mdl.addConstr(c[j, m_sig] <= x[j])
        if m_list:
            mdl.addConstr(c[j, m_sig] <= gp.quicksum(x[i] for i in m_list))
            for i in m_list:
                mdl.addConstr(c[j, m_sig] >= x[j] + x[i] - 1)
        else:
            mdl.addConstr(c[j, m_sig] <= 0)

    mdl.setObjective(
        gp.quicksum(
            F_agg[q] * (y1[q] + (w1 - 1) * y2[q] + (w2 - w1) * y3[q])
            for q in Q_agg
        ),
        GRB.MAXIMIZE,
    )

    mdl.addConstr(
        gp.quicksum(x[j] for j in J) <= math.floor(alpha * len(J)),
        name="Budget",
    )

    for q in Q_agg:
        p_dir = P_agg.get(q, [])
        s_pt = S_agg.get(q, [])

        # Tier 1
        if p_dir:
            mdl.addConstr(y1[q] <= gp.quicksum(b[jm] for jm in p_dir))
            orig_stations = set(j for (j, _) in p_dir)
            dest_stations = set(m for (_, m) in p_dir)
            mdl.addConstr(y1[q] <= gp.quicksum(x[j] for j in orig_stations))
            mdl.addConstr(y1[q] <= gp.quicksum(x[m] for m in dest_stations))
        else:
            mdl.addConstr(y1[q] <= 0)

        # Tier 2
        if s_pt:
            mdl.addConstr(y2[q] <= gp.quicksum(c[qj_to_t2sig[(q, j)]] for j in s_pt))
            mdl.addConstr(y2[q] <= gp.quicksum(x[j] for j in s_pt))
            all_M_stations = set()
            for j in s_pt:
                all_M_stations.update(M_agg.get((q, j), []))
            if all_M_stations:
                mdl.addConstr(y2[q] <= gp.quicksum(x[i] for i in all_M_stations))
        else:
            mdl.addConstr(y2[q] <= 0)

        # Tier 3
        mdl.addConstr(y3[q] <= y1[q])
        mdl.addConstr(y3[q] <= y2[q])
        mdl.addConstr(y3[q] >= y1[q] + y2[q] - 1)

    mdl.optimize()
    solve_time = mdl.Runtime

    if mdl.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and mdl.SolCount > 0:
        opened = [j for j in J if x[j].X > 0.5]
        return mdl.ObjVal, opened, solve_time
    else:
        return None, [], solve_time
