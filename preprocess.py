from collections import defaultdict
from math import radians, cos, sin, asin, sqrt
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_bss_candidate_stations():
    trips = pd.read_csv(
        config.BSS_TRIP_LOG_PATH,
        dtype={"start_station_id": str, "end_station_id": str},
        usecols=[
            "start_station_id", "start_lat", "start_lng",
            "end_station_id", "end_lat", "end_lng",
        ],
    )
    starts = trips[["start_station_id", "start_lat", "start_lng"]].rename(
        columns={"start_station_id": "station_id", "start_lat": "lat", "start_lng": "lon"}
    )
    ends = trips[["end_station_id", "end_lat", "end_lng"]].rename(
        columns={"end_station_id": "station_id", "end_lat": "lat", "end_lng": "lon"}
    )
    df = (
        pd.concat([starts, ends], ignore_index=True)
        .dropna(subset=["station_id", "lat", "lon"])
        .drop_duplicates(subset=["station_id"], keep="first")
        .reset_index(drop=True)
    )
    print(f"  Loaded {len(df)} BSS candidate stations.")
    return df


def load_trip_log():
    trips = pd.read_csv(
        config.BSS_TRIP_LOG_PATH,
        dtype={"start_station_id": str, "end_station_id": str},
        usecols=[
            "start_station_id", "start_lat", "start_lng",
            "end_station_id", "end_lat", "end_lng",
        ],
    )
    trips = trips.dropna(subset=[
        "start_station_id", "end_station_id",
        "start_lat", "start_lng", "end_lat", "end_lng",
    ]).reset_index(drop=True)
    print(f"  Loaded {len(trips):,} trip records from the BSS trip log.")
    return trips


def load_pt_stops():
    train_stops = pd.read_csv(config.TRAIN_STOPS_PATH)
    if "location_type" in train_stops.columns:
        train_stops = train_stops[train_stops["location_type"] == 1]
    train_stops = train_stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    train_stops = train_stops.dropna(subset=["stop_lat", "stop_lon"])
    train_stops["type"] = "train"

    bus_stops = pd.read_csv(config.BUS_STOPS_PATH)
    bus_stops = bus_stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    bus_stops = bus_stops.dropna(subset=["stop_lat", "stop_lon"])
    bus_stops["type"] = "bus"

    stops = pd.concat([train_stops, bus_stops], ignore_index=True).reset_index(drop=True)
    print(f"  Loaded {len(train_stops)} PT stops (Metro rail) and {len(bus_stops)} PT stops (Metrobus).")
    return stops


def load_od_demand():
    od = pd.read_csv(config.OD_DATA_PATH, dtype={"h_geocode": str, "w_geocode": str})
    od = od[["h_geocode", "w_geocode", "S000"]].dropna()
    od = od[od["S000"] > 0]
    od.columns = ["origin_id", "dest_id", "flow"]

    xwalk = pd.read_csv(
        config.XWALK_PATH,
        dtype={"tabblk2020": str},
        usecols=["tabblk2020", "blklatdd", "blklondd"],
    ).dropna().drop_duplicates(subset=["tabblk2020"])

    od = od.merge(
        xwalk.rename(columns={"tabblk2020": "origin_id", "blklatdd": "origin_lat", "blklondd": "origin_lon"}),
        on="origin_id", how="inner",
    ).merge(
        xwalk.rename(columns={"tabblk2020": "dest_id", "blklatdd": "dest_lat", "blklondd": "dest_lon"}),
        on="dest_id", how="inner",
    )
    print(f"  Loaded {len(od):,} OD pairs with positive demand.")
    return od.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SPATIAL HELPERS
# ---------------------------------------------------------------------------

# Great-circle distance in metres between two (lon, lat) points
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lon2 - lon1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lat2 - lat1) / 2) ** 2
    return 2 * asin(sqrt(a)) * 6_371_000


def _latlon_to_xyz(lat_rad, lon_rad):
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


# Returns, for each query point, all ref indices within radius_m metres
def _pairs_within_radius(query_coords_rad, ref_coords_rad, radius_m):
    R = 6_371_000.0
    chord = 2.0 * np.sin(radius_m / (2.0 * R))
    query_xyz = _latlon_to_xyz(query_coords_rad[:, 0], query_coords_rad[:, 1])
    ref_xyz = _latlon_to_xyz(ref_coords_rad[:, 0], ref_coords_rad[:, 1])
    tree = cKDTree(ref_xyz)
    return tree.query_ball_point(query_xyz, r=chord)


# ---------------------------------------------------------------------------
# SET CONSTRUCTION
# ---------------------------------------------------------------------------

def build_pt_reachability(pt_df):
    print("  Building PT reachability matrix C_{kk'} ...")
    valid_pt_ids = set(pt_df["stop_id"].astype(str))
    C = {}

    def _to_minutes(t_series):
        parts = t_series.str.split(":", expand=True).astype(int)
        return parts[0] * 60 + parts[1] + parts[2] / 60

    modes = [
        {"name": "train", "path": config.TRAIN_STOP_TIMES_PATH, "needs_mapping": True},
        {"name": "bus", "path": config.BUS_STOP_TIMES_PATH, "needs_mapping": False},
    ]

    for mode in modes:
        print(f"    -> Processing {mode['name']} stop times...")
        prev = len(C)
        stop_times = pd.read_csv(
            mode["path"],
            dtype={"stop_id": str, "trip_id": str},
            usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"],
        )

        if mode["needs_mapping"]:
            stops_meta = pd.read_csv(config.TRAIN_STOPS_PATH,
                                     dtype={"stop_id": str, "parent_station": str})
            id_map = dict(zip(stops_meta["stop_id"], stops_meta["parent_station"]))
            stop_times["mapped_id"] = stop_times["stop_id"].map(id_map)
        else:
            stop_times["mapped_id"] = stop_times["stop_id"]

        stop_times = stop_times[stop_times["mapped_id"].isin(valid_pt_ids)].copy()
        stop_times["t"] = _to_minutes(stop_times["departure_time"])
        stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

        for _, grp in stop_times.groupby("trip_id"):
            ids = grp["mapped_id"].tolist()
            t = grp["t"].tolist()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if ids[i] == ids[j]:
                        continue
                    travel_time = t[j] - t[i]
                    if 0 <= travel_time <= config.T_PT_MAX_MIN:
                        C[(ids[i], ids[j])] = 1

        print(f"  {mode['name']}: {len(C) - prev} directed reachable pairs added.")

    print(f"  Total PT reachability C has {len(C)} directed pairs.")
    return C


def build_coverage_sets(od_df, bss_df, pt_df, C_pt):
    bss_ids = bss_df["station_id"].tolist()
    pt_ids = pt_df["stop_id"].astype(str).tolist()

    bss_coords_rad = np.radians(bss_df[["lat", "lon"]].values)
    pt_coords_rad = np.radians(pt_df[["stop_lat", "stop_lon"]].values)
    bss_id_to_idx = {sid: idx for idx, sid in enumerate(bss_ids)}

    # N^o_q and N^d_q — BSS stations walkable from each origin/destination
    unique_origins = od_df[["origin_id", "origin_lat", "origin_lon"]].drop_duplicates("origin_id").reset_index(
        drop=True)
    unique_destinations = od_df[["dest_id", "dest_lat", "dest_lon"]].drop_duplicates("dest_id").reset_index(drop=True)

    orig_coords_rad = np.radians(unique_origins[["origin_lat", "origin_lon"]].values)
    dest_coords_rad = np.radians(unique_destinations[["dest_lat", "dest_lon"]].values)

    orig_nbrs_idx = _pairs_within_radius(orig_coords_rad, bss_coords_rad, config.R_WALK_BSS)
    dest_nbrs_idx = _pairs_within_radius(dest_coords_rad, bss_coords_rad, config.R_WALK_BSS)

    _N_o_loc = {
        unique_origins.loc[idx, "origin_id"]: [bss_ids[i] for i in orig_nbrs_idx[idx]]
        for idx in range(len(unique_origins))
    }
    _N_d_loc = {
        unique_destinations.loc[idx, "dest_id"]: [bss_ids[i] for i in dest_nbrs_idx[idx]]
        for idx in range(len(unique_destinations))
    }

    N_o_q = {}
    N_d_q = {}
    for _, row in od_df.iterrows():
        q = (row["origin_id"], row["dest_id"])
        N_o_q[q] = _N_o_loc.get(row["origin_id"], [])
        N_d_q[q] = _N_d_loc.get(row["dest_id"], [])

    print(f"  N_o_q / N_d_q built for {len(N_o_q)} OD flows.")

    # P^dir_q — direct BSS-to-BSS pairs within cycling range
    bss_within_cycle = _pairs_within_radius(bss_coords_rad, bss_coords_rad, config.R_RIDE_DIR)
    bss_reachability = {
        bss_ids[i]: {bss_ids[nb] for nb in bss_within_cycle[i]}
        for i in range(len(bss_ids))
    }

    P_dir_q = {}
    for _, row in od_df.iterrows():
        q = (row["origin_id"], row["dest_id"])
        P_dir_q[q] = [
            (j, m)
            for j in N_o_q[q]
            for m in N_d_q[q]
            if m in bss_reachability[j]
        ]

    n_nonempty_dq = sum(1 for v in P_dir_q.values() if v)
    print(f"  P_dir_q built: {n_nonempty_dq}/{len(P_dir_q)} OD pairs have >=1 valid BSS station pair.")

    # Precompute per-BSS-station reachable destination IDs via PT
    bss_pt_nbrs_idx = _pairs_within_radius(bss_coords_rad, pt_coords_rad, config.R_WALK_PT)

    dest_nbrs_from_pt = _pairs_within_radius(pt_coords_rad, dest_coords_rad, config.R_WALK_BSS)
    pt_to_reachable_dest_ids: dict[str, set] = {
        pt_ids[k]: {unique_destinations.loc[d_idx, "dest_id"] for d_idx in dest_nbrs_from_pt[k]}
        for k in range(len(pt_ids))
    }

    pt_reachable_from: dict[str, set] = {}
    for (k, k_prime) in C_pt:
        pt_reachable_from.setdefault(k, set()).add(k_prime)

    bss_reachable_dest_ids: dict[str, set] = {}
    for i_idx, i_id in enumerate(bss_ids):
        reachable: set = set()
        for k_idx in bss_pt_nbrs_idx[i_idx]:
            k = pt_ids[k_idx]
            for k_prime in pt_reachable_from.get(k, set()):
                reachable |= pt_to_reachable_dest_ids.get(k_prime, set())
        bss_reachable_dest_ids[i_id] = reachable

    # M_q and S^pt_q
    bss_cycle_of_j = _pairs_within_radius(bss_coords_rad, bss_coords_rad, config.R_RIDE_PT)

    M_q: dict[tuple, list] = {}
    S_pt_q: dict[tuple, list] = {}

    for _, row in od_df.iterrows():
        q = (row["origin_id"], row["dest_id"])
        dest_id = row["dest_id"]
        phi_stations = []

        for j in N_o_q[q]:
            j_idx = bss_id_to_idx[j]
            m_qj = [
                bss_ids[i_idx]
                for i_idx in bss_cycle_of_j[j_idx]
                if dest_id in bss_reachable_dest_ids.get(bss_ids[i_idx], set())
            ]
            if m_qj:
                M_q[(q, j)] = m_qj
                phi_stations.append(j)

        S_pt_q[q] = phi_stations

    n_nonempty_phi = sum(1 for v in S_pt_q.values() if v)
    print(f"  S^pt_q built: {n_nonempty_phi}/{len(S_pt_q)} OD pairs have >=1 PT-hop-enabling origin station.")
    print(f"  M_q built: {len(M_q)} (q, j) entries with >=1 intermediate station.")

    return N_o_q, N_d_q, P_dir_q, S_pt_q, M_q


# ---------------------------------------------------------------------------
# COVERAGE-SIGNATURE AGGREGATION
# ---------------------------------------------------------------------------

def precompute_aggregation_structure(Q_active, P_dir_q, S_pt_q, M_q):
    sig_to_rep = {}
    q_to_sig = {}
    rep_sets = {}

    for q in Q_active:
        p = frozenset(P_dir_q.get(q, []))
        s = frozenset(S_pt_q.get(q, []))
        m_inner = tuple(
            sorted(
                (j, frozenset(M_q.get((q, j), [])))
                for j in S_pt_q.get(q, [])
            )
        )
        sig = (p, s, m_inner)
        q_to_sig[q] = sig

        if sig not in sig_to_rep:
            sig_to_rep[sig] = q
            rep_sets[q] = (
                P_dir_q.get(q, []),
                S_pt_q.get(q, []),
                {j: M_q.get((q, j), []) for j in S_pt_q.get(q, [])},
            )

    Q_agg = list(sig_to_rep.values())
    P_agg = {q: rep_sets[q][0] for q in Q_agg}
    S_agg = {q: rep_sets[q][1] for q in Q_agg}
    M_agg = {
        (q, j): m_list
        for q in Q_agg
        for j, m_list in rep_sets[q][2].items()
    }
    n_merged = len(Q_active) - len(Q_agg)

    return Q_active, Q_agg, P_agg, S_agg, M_agg, sig_to_rep, q_to_sig, n_merged


def aggregate_flows(F, Q_active, sig_to_rep, q_to_sig):
    sig_to_flow = defaultdict(float)
    for q in Q_active:
        sig_to_flow[q_to_sig[q]] += F[q]
    return {sig_to_rep[sig]: flow for sig, flow in sig_to_flow.items()}


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def build_model_data():
    print("[1/6] Loading BSS candidate stations ...")
    bss_df = load_bss_candidate_stations()

    print("[2/6] Loading PT stops ...")
    pt_df = load_pt_stops()

    print("[3/6] Loading OD demand ...")
    od_df = load_od_demand()

    print("[4/6] Building PT reachability matrix C_{kk'} ...")
    C_pt = build_pt_reachability(pt_df)

    print("[5/6] Building spatial coverage sets ...")
    N_o_q, N_d_q, P_dir_q, S_pt_q, M_q = build_coverage_sets(od_df, bss_df, pt_df, C_pt)

    reachable_stations = set()
    for pairs in P_dir_q.values():
        for j, m in pairs:
            reachable_stations.add(j)
            reachable_stations.add(m)
    for stations in S_pt_q.values():
        reachable_stations.update(stations)
    for m_list in M_q.values():
        reachable_stations.update(m_list)

    bss_df_filtered = bss_df[bss_df["station_id"].isin(reachable_stations)].reset_index(drop=True)
    J = bss_df_filtered["station_id"].tolist()
    print(f"  Reduced |J|: {len(bss_df)} -> {len(J)} stations after filtering.")

    Q = list(zip(od_df["origin_id"], od_df["dest_id"]))
    F = {
        (row["origin_id"], row["dest_id"]): float(row["flow"])
        for _, row in od_df.iterrows()
    }

    print("[6/6] Loading BSS trip log ...")
    trips_df = load_trip_log()

    print(f"\nModel data ready: |J|={len(J)}, |Q|={len(Q):,}")
    return J, Q, F, P_dir_q, S_pt_q, M_q, bss_df_filtered, pt_df, od_df, trips_df
