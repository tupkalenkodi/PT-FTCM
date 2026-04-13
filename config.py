from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "cleaned_data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OD_DATA_PATH = DATA_DIR / "dc_od_main_JT00_2023.csv"
XWALK_PATH = DATA_DIR / "dc_xwalk.csv"
TRAIN_STOPS_PATH = DATA_DIR / "train_stops.txt"
TRAIN_STOP_TIMES_PATH = DATA_DIR / "train_stop_times.txt"
BUS_STOPS_PATH = DATA_DIR / "bus_stops.txt"
BUS_STOP_TIMES_PATH = DATA_DIR / "bus_stop_times.txt"
BSS_TRIP_LOG_PATH = DATA_DIR / "202306-capitalbikeshare-tripdata.csv"

# Max walking distance (m) from trip origin/destination to a BSS station
R_WALK_BSS = 300

# Max walking distance (m) from BSS station to PT station
R_WALK_PT = 50

# Max cycling distance (m) for a direct BSS-to-BSS trip
R_RIDE_DIR = 5000

# Max cycling distance (m) from BSS station to a BSS station near a PT stop
R_RIDE_PT = 2000

# Max PT travel time (min) for a connection between two PT stops
T_PT_MAX_MIN = 20

# Proportion of candidate BSS stations to open
ALPHA = 0.3

# (w1, w2) grid used in correlation analysis
w_vals = [round(1.0 + 0.25 * i, 2) for i in range(9)]
CORRELATION_GRID = [
    (w1, w2)
    for w1 in w_vals
    for w2 in w_vals
    if w2 >= w1
]  # -> 45 points

# (w1, w2) grid used in timing and sensitivity analysis
SENSITIVITY_GRID = [
    (1.0, 1.0),
    (2.0, 2.0),
    (2.0, 3.0),
]

# Perturbation levels for demand uncertainty analysis
SENSITIVITY_DELTAS = [0.1, 0.2, 0.3]

# Number of noise realizations per delta level
SENSITIVITY_N_REALISATIONS = 10
