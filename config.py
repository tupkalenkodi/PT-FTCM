from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "cleaned_data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OD_DATA_PATH = str(DATA_DIR / "dc_od_main_JT00_2023.csv")
XWALK_PATH = str(DATA_DIR / "dc_xwalk.csv")
TRAIN_STOPS_PATH = str(DATA_DIR / "train_stops.txt")
TRAIN_STOP_TIMES_PATH = str(DATA_DIR / "train_stop_times.txt")
BUS_STOPS_PATH = str(DATA_DIR / "bus_stops.txt")
BUS_STOP_TIMES_PATH = str(DATA_DIR / "bus_stop_times.txt")
BSS_TRIP_LOG_PATH = str(DATA_DIR / "202306-capitalbikeshare-tripdata.csv")

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

# Fractions of the full OD demand set to sample
TIMING_FRACTIONS = [round(i / 10, 1) for i in range(10, 0, -1)]

# (w1, w2, case_num) grid used in timing and sensitivity analysis
TIMING_GRID = [
    (1.0, 1.0, 1),
    (1.5, 1.5, 2),
    (2.0, 2.0, 3),
    (2.5, 2.5, 4),
    (2.0, 3.0, 5),
]

# (w1, w2) grid used in correlation analysis
CORRELATION_GRID = [
    (1.0, 1.0), # Case 1
    (1.1, 1.1), (1.2, 1.2), (1.3, 1.3), (1.4, 1.4), (1.5, 1.5), (1.6, 1.6), (1.7, 1.7), (1.8, 1.8), (1.9, 1.9), # 2
    (2.0, 2.0), # 3
    (2.1, 2.1), (2.2, 2.2), (2.3, 2.3), (2.4, 2.4), (2.5, 2.5), (2.6, 2.6), (2.7, 2.7), (2.8, 2.8), (2.9, 2.9), # 4
    (2.1, 2.2), (2.1, 2.3), (2.1, 2.4), (2.1, 2.5),(2.1, 2.6), (2.1, 2.7), (2.1, 2.8), (2.1, 2.9), (2.1, 3.0), # 5
    (2.2, 2.3), (2.2, 2.4), (2.2, 2.5), (2.2, 2.6), (2.2, 2.7), (2.2, 2.8), (2.2, 2.9), (2.2, 3.0),
]

# Perturbation levels for demand uncertainty analysis
SENSITIVITY_DELTAS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Number of noise realizations per delta level
SENSITIVITY_N_REALISATIONS = 10
