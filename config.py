from pathlib import Path

# --- DIRECTORIES ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- FILE PATHS ---
OD_DATA_PATH = DATA_DIR / "dc_od_main_JT00_2023.csv"
XWALK_PATH = DATA_DIR / "dc_xwalk.csv"
TRAIN_STOPS_PATH = DATA_DIR / "train_stops.txt"
TRAIN_STOP_TIMES_PATH = DATA_DIR / "train_stop_times.txt"
BUS_STOPS_PATH = DATA_DIR / "bus_stops.txt"
BUS_STOP_TIMES_PATH = DATA_DIR / "bus_stop_times.txt"
BSS_TRIP_LOG_PATH = DATA_DIR / "202306-capitalbikeshare-tripdata.csv"


# --- MODEL PARAMETERS ---
# R_WALK_BSS: Max walking distance (meters) from trip origin/destination to a BSS station
R_WALK_BSS = 300

# R_WALK_PT: Max walking distance (meters) from BSS station to PT station
R_WALK_PT = 50

# R_RIDE_DIR: Max cycling distance (meters) for a direct BSS-to-BSS trip
R_RIDE_DIR = 5000

# R_RIDE_PT: Max cycling distance (meters) from BSS station to a BSS station near a PT stop
R_RIDE_PT = 1000

# T_PT_MAX_MINUTES: Max PT travel time (minutes) for a connection between two PT stops
T_PT_MAX_MINUTES = 20

# ALPHA: Proportion of candidate BSS stations to open (0.0 to 1.0)
ALPHA = 0.25


# --- EVALUATION GRID ---
# Grid G of (w_1, w_2) policy parameter pairs to evaluate
# Constraints: w_1 >= 1.0, w_2 >= w_1
W_GRID = [
    (1.0, 1.0),  # Case 1
    (1.1, 1.1),  # Case 2
    (1.1, 1.2),
    (1.2, 1.3),
    (1.3, 1.4),
    (1.4, 1.5),
    (1.5, 1.6),
    (1.6, 1.7),
    (1.7, 1.8),
    (1.8, 1.9),
    (1.9, 2.0),
    (2.0, 2.0),  # Case 3
    (2.1, 2.1),  # Case 4
    (2.2, 2.2),
    (2.3, 2.3),
    (2.4, 2.4),
    (2.5, 2.5),
    (2.6, 2.6),
    (2.7, 2.7),
    (2.8, 2.8),
    (2.9, 2.9),
    (3.0, 3.0),
    (2.1, 2.2),  # Case 5
    (2.1, 2.3),
    (2.1, 2.4),
    (2.1, 2.5),
    (2.1, 2.6),
    (2.1, 2.7),
    (2.1, 2.8),
    (2.1, 2.9),
    (2.1, 3.0),
    (2.2, 2.3),
    (2.2, 2.4),
    (2.2, 2.5),
    (2.2, 2.6),
    (2.2, 2.7),
    (2.2, 2.8),
    (2.2, 2.9),
    (2.2, 3.0),
]


# --- SENSITIVITY ANALYSIS ---
# Perturbation levels delta for demand uncertainty analysis
SENSITIVITY_DELTAS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Number of noise realizations per delta level
SENSITIVITY_N_REALISATIONS = 10
