"""Canonical path constants for the project (CWD-independent)."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Reference corpus (Dolma v1.7) — built once by src/reference_corpus/
DOLMA_DIR = DATA_DIR / "dolma"
DOLMA_DOCBINS_DIR = DOLMA_DIR / "docbins"
DOLMA_FREQ_DIR = DOLMA_DIR / "frequency_tables"

# Target corpora
ELLIPSE_DIR = DATA_DIR / "ellipse"            # studies 1, 2, 3
ELLIPSE_DOCBINS_DIR = ELLIPSE_DIR / "docbins"
TOEFL_DIR = DATA_DIR / "toefl11"              # studies 1, 3
TOEFL_DOCBINS_DIR = TOEFL_DIR / "docbins"

# Archived pilot-study artifacts (SlimPajama lists, delta-vector pilot)
PILOT_DIR = DATA_DIR / "pilot"

FIG_DIR = PROJECT_ROOT / "fig"
RESULTS_DIR = PROJECT_ROOT / "results"
