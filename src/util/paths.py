"""Canonical path constants for the project (CWD-independent)."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DOLMA_DIR = DATA_DIR / "dolma"
DOLMA_DOCBINS_DIR = DOLMA_DIR / "docbins"
DOLMA_FREQ_DIR = DOLMA_DIR / "frequency_tables"
FIG_DIR = PROJECT_ROOT / "fig"
RESULTS_DIR = PROJECT_ROOT / "results"
