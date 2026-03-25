"""Central config — paths, constants, model names."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# Paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOP100_PATH = DATA_DIR / "top100_firms.json"
CHROMA_DIR = ROOT / "chroma_db"

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WRDS_USERNAME = os.getenv("WRDS_USERNAME")

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Analysis
EVENT_WINDOW_PRE = 5    # days before earnings call
EVENT_WINDOW_POST = 30  # days after earnings call
BASELINE_WINDOW = 90    # days for baseline download rate

# Study period
START_YEAR = 2003
END_YEAR = 2015
CRISIS_START = "2007-07-01"
CRISIS_END = "2009-06-30"
