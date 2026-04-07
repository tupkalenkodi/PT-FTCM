import gurobipy as gp
from gurobipy import GRB


# Parameters
# ----------
# J        : list[str]                    candidate BSS station IDs
# Q        : list[tuple]                  OD pairs (origin_id, dest_id)
# F        : dict[(str,str)->float]       demand weights F_q
# P_dir_q  : dict[tuple->list[tuple]]     valid (j,m) BSS pairs per OD pair (Tier 1)
# S_pt_q   : dict[tuple->list[str]]       origin stations enabling a PT-hop for q (Tier 2)
# M_q      : dict[(tuple,str)->list[str]] M_{q,j} - intermediate BSS stations for (q, j)
# alpha    : float                        budget proportion in [0,1]
# w1       : float ≥ 1                    tier weight w_1
# w2       : float ≥ w1                   tier weight w_2
#
# Returns
# -------
# obj_val         : float | None
# opened_stations : list[str]


def build_and_solve_pt_ftcm(J, Q, F, P_dir_q, S_pt_q, M_q, alpha, w1, w2):
    assert w1 >= 1.0, "w_1 must be geq 1"
    assert w2 >= w1, "w_2 must be geq w_1"

    mdl = gp.Model("PT_FTCM")
    mdl.setParam("OutputFlag", 0)
    mdl.setParam("MIPGap", 0.01)


    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------

    # x_j in {0,1} - station j open
    x = mdl.addVars(J, vtype=GRB.BINARY, name="x")
    #
    # # y1_q in [0,1] - Tier 1
    # y1 = mdl.addVars(Q, vtype=GRB.BINARY, name="y1")
    #
    # # y2_q in [0,1] - Tier 2
    # y2 = mdl.addVars(Q, vtype=GRB.BINARY, name="y2")
    #
    # # y3_q in [0,1] - Tier 3
    # y3 = mdl.addVars(Q, vtype=GRB.BINARY, name="y3")

    #
    y1 = mdl.addVars(Q, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y1")
    y2 = mdl.addVars(Q, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y2")
    y3 = mdl.addVars(Q, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y3")


    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------

    mdl.setObjective(
        gp.quicksum(
            F[q] * (y1[q] + (w1 - 1) * y2[q] + (w2 - w1) * y3[q]) for q in Q
        ),
        GRB.MAXIMIZE,
    )



    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------

    # Budget (using proportion alpha)
    mdl.addConstr(
        gp.quicksum(x[j] for j in J) <= int(alpha * len(J)), name="Budget"
    )

    for q in Q:
        orig, dest = q

        p_dir = P_dir_q.get(q, [])
        s_pt = S_pt_q.get(q, [])

        # Tier 1
        if p_dir:
            pair_vars = []
            for j, m in p_dir:
                a1 = mdl.addVar(
                    vtype=GRB.BINARY,
                    name=f"a1_{orig}_{dest}_{j}_{m}",
                )
                mdl.addConstr(
                    a1 <= x[j], name=f"a1_ub_j_{orig}_{dest}_{j}_{m}"
                )
                mdl.addConstr(
                    a1 <= x[m], name=f"a1_ub_m_{orig}_{dest}_{j}_{m}"
                )
                mdl.addConstr(
                    a1 >= x[j] + x[m] - 1, name=f"a1_lb_{orig}_{dest}_{j}_{m}"
                )
                pair_vars.append(a1)

            mdl.addConstr(
                y1[q] <= gp.quicksum(pair_vars), name=f"T1_{orig}_{dest}"
            )
        else:
            mdl.addConstr(y1[q] <= 0, name=f"T1_none_{orig}_{dest}")

        # Tier 2
        if s_pt:
            a2_vars = []
            for j in s_pt:
                a2 = mdl.addVar(
                    vtype=GRB.BINARY,
                    name=f"a2_{orig}_{dest}_{j}",
                )
                mdl.addConstr(a2 <= x[j], name=f"a2_ub_j_{orig}_{dest}_{j}")

                m_qj = M_q.get((q, j), [])
                if m_qj:
                    mdl.addConstr(
                        a2 <= gp.quicksum(x[i] for i in m_qj),
                        name=f"a2_ub_i_{orig}_{dest}_{j}",
                    )
                else:
                    mdl.addConstr(a2 <= 0, name=f"a2_empty_{orig}_{dest}_{j}")

                a2_vars.append(a2)

            mdl.addConstr(
                y2[q] <= gp.quicksum(a2_vars), name=f"T2_{orig}_{dest}"
            )
        else:
            mdl.addConstr(y2[q] <= 0, name=f"T2_none_{orig}_{dest}")

        # Tier 3
        mdl.addConstr(y3[q] <= y1[q], name=f"T3_zy_{orig}_{dest}")
        mdl.addConstr(y3[q] <= y2[q], name=f"T3_zv_{orig}_{dest}")
        mdl.addConstr(
            y3[q] >= y1[q] + y2[q] - 1, name=f"T3_lb_{orig}_{dest}"
        )


    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    mdl.optimize()

    solve_time = mdl.Runtime  # wall-clock seconds recorded by Gurobi

    if mdl.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and mdl.SolCount > 0:
        opened = [j for j in J if x[j].X > 0.5]
        return mdl.ObjVal, opened, solve_time
    else:
        return None, [], solve_time
