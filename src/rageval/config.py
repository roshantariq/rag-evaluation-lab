"""Project-wide paths and environment settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CORPUS_MANIFEST = EVAL_DIR / "corpus_manifest.jsonl"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBED_CACHE = PROJECT_ROOT / ".cache" / "embeddings.sqlite"

RESPONSE_CACHE = PROJECT_ROOT / ".cache" / "responses.sqlite"

def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, EVAL_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)