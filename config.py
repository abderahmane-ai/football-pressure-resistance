import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_TRACES_DIR = OUTPUT_DIR / "model_traces"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Competitions (Verified available 360 data)
COMPETITIONS = {
    "Bundesliga_2024": {"comp_id": 9, "season_id": 281}, # Germany, 2023/2024, 34 matches
    "World_Cup_2022": {"comp_id": 43, "season_id": 106}, # International, 2022, 64 matches
    "Euro_2024": {"comp_id": 55, "season_id": 282},     # Europe, 2024, 51 matches
    "Euro_2020": {"comp_id": 55, "season_id": 43},      # Europe, 2020, 51 matches
}

# Cross-validation holdout (Dynamically set via environment variable, defaults to Euro 2020)
CROSS_VALIDATION_HOLDOUT = os.environ.get("PRS_HOLDOUT", "Euro_2020")

# Model Settings
MODEL_SETTINGS = {
    "draws": 2000,
    "tune": 2000,
    "chains": 4,
    "random_seed": 42,
    "target_accept": 0.95,
    "nuts_sampler": "numpyro",
}
