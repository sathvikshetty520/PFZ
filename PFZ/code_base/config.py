"""
PFZ Prediction System - Configuration & Parameters
Project: AI-Based Dynamic Fishing Zone Prediction System
College: Sahyadri College of Engineering & Management, Mangaluru
"""

import os
from pathlib import Path

# --- DIRECTORY PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CODE_DIR = BASE_DIR / "code_base"
MODELS_DIR = CODE_DIR / "saved_models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# --- ENVIRONMENT & API KEYS ---
def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

# Load .env from workspace root or PFZ folder
_load_env_file(BASE_DIR.parent / ".env")
_load_env_file(BASE_DIR / ".env")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# Internal oceanographic raw and aligned dataset paths (inside PFZ/data)
TABULAR_PARQUET_DIR = DATA_DIR / "tabular_parquet"
NETCDF_DIR = DATA_DIR / "netcdf"
MEM4_DIR = DATA_DIR / "mem4_integrated_dataset"

SOURCE_DAILY_COINCIDENT = TABULAR_PARQUET_DIR / "aligned_coincident_daily.parquet"
SOURCE_DAILY_BASIN = TABULAR_PARQUET_DIR / "aligned_dataset_daily.parquet"
SOURCE_NETCDF_BASIN = NETCDF_DIR / "aligned_daily_basin.nc"

# Target dataset paths inside PFZ/data
OCEAN_DATA_FILE = DATA_DIR / "ocean_features_daily.parquet"
CATCH_DATA_FILE = DATA_DIR / "historical_catch_landing_data.csv"
PORTS_FILE = DATA_DIR / "coastal_ports.json"

# Create necessary directories
for d in [DATA_DIR, CODE_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- GEOGRAPHIC DOMAINS ---
# Primary Focus: Southwest Coast of India (Karnataka, Kerala, Goa)
# Bounds: Lat 7.0°N to 16.5°N, Lon 71.5°E to 77.5°E
SW_COAST_BOUNDS = {
    "lat_min": 7.0,
    "lat_max": 16.5,
    "lon_min": 71.5,
    "lon_max": 77.5
}

# --- KEY FISHING PORTS & LANDING CENTERS ---
KEY_PORTS = {
    "Mangaluru Old Port (Bunder)": {"lat": 12.8584, "lon": 74.8360, "state": "Karnataka"},
    "Malpe Harbor (Udupi)": {"lat": 13.3512, "lon": 74.7018, "state": "Karnataka"},
    "Karwar Fishing Harbor": {"lat": 14.8144, "lon": 74.1297, "state": "Karnataka"},
    "Kochi Harbor (Thoppumpady)": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "Beypore Port (Kozhikode)": {"lat": 11.1643, "lon": 75.8070, "state": "Kerala"},
    "Panaji (Mandovi Fishery Wharf)": {"lat": 15.4989, "lon": 73.8278, "state": "Goa"},
    "Ratnagiri Harbor (Mirkarwada)": {"lat": 16.9944, "lon": 73.2825, "state": "Maharashtra"}
}

# --- FEATURE DEFINITIONS ---
ENVIRONMENTAL_FEATURES = [
    "sst",               # Sea Surface Temperature (°C)
    "uo",                # Eastward ocean current velocity (m/s)
    "vo",                # Northward ocean current velocity (m/s)
    "current_speed",     # Surface current speed magnitude (m/s)
    "u10",               # 10m eastward wind (m/s)
    "v10",               # 10m northward wind (m/s)
    "wind_speed",        # 10m wind speed magnitude (m/s)
    "chlor_a",           # Chlorophyll-a concentration (mg/m³)
    "elevation",         # Bathymetry depth (m, negative=depth)
    "dist_to_coast_km",  # Shortest distance to shoreline (km)
    "adt",               # Absolute Dynamic Topography / SSH (m)
    "sla",               # Sea Level Anomaly (m)
    "ugos",              # Geostrophic current velocity u (m/s)
    "vgos",              # Geostrophic current velocity v (m/s)
    "month_sin",         # Cyclical seasonal month sine
    "month_cos"          # Cyclical seasonal month cosine
]

# Engineered ocean front features
FRONT_FEATURES = [
    "sst_gradient",      # Thermal front gradient magnitude (°C / km)
    "sst_persistence",   # Multi-day SST front persistence
    "chlor_a_front",     # Chlorophyll bloom front gradient
    "front_overlap"      # Co-occurrence of thermal front & chlorophyll bloom
]

ALL_MODEL_FEATURES = ENVIRONMENTAL_FEATURES + FRONT_FEATURES

# --- INCOIS-BENCHMARKED PFZ CRITERIA ---
PFZ_CRITERIA = {
    "sst_min": 24.0,           # Lower physiological limit for pelagic shoals
    "sst_max": 31.0,           # Upper physiological limit
    "sst_grad_threshold": 0.25, # Thermal gradient (°C / grid unit) indicating front
    "chlor_min": 0.30,         # Minimum chlorophyll for active primary productivity
    "chlor_max": 15.0,         # Cap above which eutrophic / turbid waters occur
    "shelf_break_min_depth": -1000.0, # Continental slope / shelf break upper limit
    "shelf_break_max_depth": -30.0,   # Minimum artisanal depth
    "max_distance_km": 150.0   # Operational range for small-scale/mechanized boats
}

# --- MODEL TRAINING PARAMETERS ---
RANDOM_SEED = 42
TRAIN_YEAR = 2020
TEST_YEAR = 2021
TEMPORAL_LOOKBACK_DAYS = 5     # For LSTM / front persistence window
PROBABILITY_THRESHOLD = 0.55   # PFZ classification boundary
