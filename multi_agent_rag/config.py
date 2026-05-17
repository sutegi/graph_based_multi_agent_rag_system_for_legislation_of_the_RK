from __future__ import annotations
"""Environment configuration and shared constants for the RAG pipeline."""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rag_pipeline")

MAX_CONTEXT_TOKENS: int = 8_000
MAX_ITERATIONS:     int = 5
TOP_K_SEARCH:       int = 10
TOP_K_RELATED:      int = 5

DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")

NEO4J_URI      = os.getenv("NEO4J_URI", "")
NEO4J_USER     = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "")

GRAPH_RELATION_TYPES: list[str] = [
    "BASED_ON", "CAUSED_BY", "REFERENCES",
    "CORRELATED_WITH", "EXTENDS", "MANIFESTED_IN",
]
