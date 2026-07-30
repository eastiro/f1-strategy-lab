from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESEARCH_DIR = REPORTS_DIR / "research"
RESULTS_DIR = REPORTS_DIR / "results"
PRESENTATION_DIR = REPORTS_DIR / "presentation"
FIGURES_DIR = REPORTS_DIR / "figures"

OPENF1_BASE_URL = "https://api.openf1.org/v1"

DEFAULT_YEAR = 2024
DEFAULT_COUNTRY = "United Kingdom"
DEFAULT_SESSION = "Race"
