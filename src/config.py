from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Dataset path
RAW_DATA_FILE = RAW_DATA_DIR / "german_credit_data.csv"


TARGET_COLUMN = "Risk"
NUMERIC_COLUMNS = ["Age", "Credit amount", "Duration"]
CATEGORICAL_COLUMNS = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
SPECIAL_COLUMNS = ["Job"]

ORDINAL_COLUMNS = ["Saving accounts", "Checking account"]
NOMINAL_COLUMNS = ["Sex", "Housing", 'Purpose']