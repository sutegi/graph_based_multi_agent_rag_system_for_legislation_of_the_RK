"""
create_definition_graph.py
──────────────────────────
Ingests keyword definitions from data/keyword_definitions/*.json into Neo4j,
creating Definition nodes and (Keyword)-[:HAS_DEFINITION]->(Definition) edges.

The JSON files were produced by the keyword-definition extraction pipeline and
have this schema:
    {
      "keyword":       "<Russian keyword text>",
      "keyword_kz":    "<Kazakh keyword text>",
      "relation":      "has_definition",
      "definition_ru": "<Russian definition>",
      "definition_kz": "<Kazakh definition>",
      "language":      "bilingual"
    }

Neo4j target schema (must be consistent with multi_agent_rag/database.py):
    (:Keyword {text_ru, text_kz, ...})
        -[:HAS_DEFINITION]->
    (:Definition {keyword_ru, keyword_kz, definition_ru, definition_kz, source_file})

Run:
    python create_definition_graph.py                  # full ingest
    python create_definition_graph.py --dry-run        # parse only, no writes
    python create_definition_graph.py --limit 500      # first N files only
    python create_definition_graph.py --clear          # wipe Definition nodes first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import islice
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("definition_graph")

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # ingestion/
_CODE_DIR   = _SCRIPT_DIR.parent                       # diploma_code/
_DATA_DIR   = _CODE_DIR / "data" / "keyword_definitions"
_ENV_FILE   = _CODE_DIR / ".env"

BATCH_SIZE  = 500   # Neo4j write batch size


# =============================================================================
# Neo4j helpers
# =============================================================================

def _neo4j_driver(uri: str, user: str, password: str) -> Driver:
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        connection_timeout=15,
        max_connection_pool_size=10,
    )


def _ping(driver: Driver, database: str) -> bool:
    try:
        with driver.session(database=database) as s:
            s.run("RETURN 1").consume()
        return True
    except Exception as exc:
        log.error("Neo4j ping failed: %s", exc)
        return False


def _ensure_indexes(driver: Driver, database: str) -> None:
    """Create indexes/constraints needed for fast MERGE operations."""
    statements = [
        # Uniqueness constraint on Definition — prevents duplicates on re-run
        "CREATE CONSTRAINT definition_unique IF NOT EXISTS "
        "FOR (d:Definition) REQUIRE (d.keyword_ru, d.definition_ru) IS NODE KEY",

        # Keyword text_ru index (may already exist from legal_graph_builder)
        "CREATE INDEX keyword_text_ru IF NOT EXISTS "
        "FOR (k:Keyword) ON (k.text_ru)",

        # Keyword text_kz index
        "CREATE INDEX keyword_text_kz IF NOT EXISTS "
        "FOR (k:Keyword) ON (k.text_kz)",
    ]
    with driver.session(database=database) as s:
        for stmt in statements:
            try:
                s.run(stmt).consume()
            except Exception as exc:
                # Constraint/index might already exist under a different name — OK
                log.debug("Index/constraint stmt skipped (%s): %s", exc, stmt[:60])
    log.info("Indexes ensured.")


def _clear_definitions(driver: Driver, database: str) -> None:
    """Remove all existing Definition nodes and HAS_DEFINITION relationships."""
    with driver.session(database=database) as s:
        result = s.run(
            "MATCH (d:Definition) "
            "DETACH DELETE d "
            "RETURN count(d) AS deleted"
        )
        rec = result.single()
        deleted = rec["deleted"] if rec else 0
    log.info("Cleared %d existing Definition nodes.", deleted)


# =============================================================================
# File loading
# =============================================================================

def _iter_definition_files(data_dir: Path, limit: int | None) -> Iterator[Path]:
    files = sorted(data_dir.glob("*.json"))
    if limit:
        files = list(islice(files, limit))
    return iter(files)


def _load_entry(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Skipping %s — JSON parse error: %s", path.name, exc)
        return None

    # Aggregate files (e.g. _all_definitions.json) are lists — skip them here;
    # they are covered by the individual per-keyword files.
    if isinstance(raw, list):
        log.debug("Skipping %s — aggregate list file, handled separately", path.name)
        return None

    keyword_ru = (raw.get("keyword") or "").strip()
    definition_ru = (raw.get("definition_ru") or "").strip()

    if not keyword_ru or not definition_ru:
        log.debug("Skipping %s — missing keyword or definition_ru", path.name)
        return None

    return {
        "keyword_ru":    keyword_ru,
        "keyword_kz":    (raw.get("keyword_kz") or "").strip(),
        "definition_ru": definition_ru,
        "definition_kz": (raw.get("definition_kz") or "").strip(),
        "source_file":   path.name,
    }


def _batch(iterable, n: int):
    """Yield successive n-sized chunks from an iterable."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# =============================================================================
# Neo4j write
# =============================================================================

_UPSERT_CYPHER = """
UNWIND $rows AS row

// ── 1. Ensure the Keyword node exists (MERGE on text_ru) ──────────────────
MERGE (k:Keyword {text_ru: row.keyword_ru})
ON CREATE SET
    k.text_kz  = row.keyword_kz,
    k.created  = timestamp()
ON MATCH SET
    k.text_kz  = CASE WHEN row.keyword_kz <> '' THEN row.keyword_kz
                      ELSE coalesce(k.text_kz, '') END

// ── 2. Ensure the Definition node exists (idempotent re-runs) ─────────────
MERGE (d:Definition {keyword_ru: row.keyword_ru, definition_ru: row.definition_ru})
ON CREATE SET
    d.keyword_kz    = row.keyword_kz,
    d.definition_kz = row.definition_kz,
    d.source_file   = row.source_file,
    d.created       = timestamp()

// ── 3. Link Keyword → Definition ──────────────────────────────────────────
MERGE (k)-[:HAS_DEFINITION]->(d)

RETURN count(d) AS written
"""


def _write_batch(session, rows: list[dict]) -> int:
    result = session.run(_UPSERT_CYPHER, rows=rows)
    rec = result.single()
    return rec["written"] if rec else len(rows)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest keyword definitions into Neo4j as Definition nodes."
    )
    p.add_argument(
        "--data-dir", type=Path, default=_DATA_DIR,
        help=f"Directory of keyword definition JSONs (default: {_DATA_DIR})",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N files (useful for testing).",
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Neo4j write batch size (default: {BATCH_SIZE}).",
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Delete all existing Definition nodes before ingesting.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Parse files and count entries but do NOT write to Neo4j.",
    )
    p.add_argument(
        "--env", type=Path, default=_ENV_FILE,
        help=f"Path to .env file (default: {_ENV_FILE}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load .env ─────────────────────────────────────────────────────────────
    if args.env.exists():
        load_dotenv(args.env)
        log.info("Loaded .env from %s", args.env)
    else:
        load_dotenv()
        log.warning(".env not found at %s — relying on environment variables.", args.env)

    neo4j_uri  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER",      "neo4j")
    neo4j_pass = os.environ.get("NEO4J_PASSWORD",  "")
    neo4j_db   = os.environ.get("NEO4J_DATABASE",  "neo4j")

    # ── Validate data directory ────────────────────────────────────────────────
    data_dir: Path = args.data_dir
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    total_files = sum(1 for _ in data_dir.glob("*.json"))
    log.info("Found %d JSON files in %s", total_files, data_dir)

    # ── Dry-run mode ──────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("[DRY-RUN] Parsing files without writing to Neo4j …")
        ok = skipped = 0
        limit = args.limit or total_files
        for path in islice(sorted(data_dir.glob("*.json")), limit):
            entry = _load_entry(path)
            if entry:
                ok += 1
            else:
                skipped += 1
        log.info("[DRY-RUN] Valid: %d | Skipped: %d | Total processed: %d",
                 ok, skipped, ok + skipped)
        return

    # ── Connect to Neo4j ──────────────────────────────────────────────────────
    log.info("Connecting to Neo4j at %s (db=%s) …", neo4j_uri, neo4j_db)
    driver = _neo4j_driver(neo4j_uri, neo4j_user, neo4j_pass)

    if not _ping(driver, neo4j_db):
        log.error("Cannot reach Neo4j. Check that it is running and .env is correct.")
        driver.close()
        sys.exit(1)
    log.info("Neo4j connection OK.")

    # ── Optionally clear existing definitions ─────────────────────────────────
    if args.clear:
        log.info("--clear flag set: removing existing Definition nodes …")
        _clear_definitions(driver, neo4j_db)

    # ── Ensure indexes ────────────────────────────────────────────────────────
    _ensure_indexes(driver, neo4j_db)

    # ── Ingest ────────────────────────────────────────────────────────────────
    log.info("Starting ingestion (batch_size=%d, limit=%s) …",
             args.batch_size, args.limit or "all")

    total_written = 0
    total_skipped = 0
    batch_num     = 0

    files_iter = _iter_definition_files(data_dir, args.limit)

    def _entry_stream():
        nonlocal total_skipped
        for path in files_iter:
            entry = _load_entry(path)
            if entry:
                yield entry
            else:
                total_skipped += 1

    with driver.session(database=neo4j_db) as session:
        for rows in _batch(_entry_stream(), args.batch_size):
            batch_num += 1
            written = _write_batch(session, rows)
            total_written += written
            if batch_num % 10 == 0 or batch_num == 1:
                log.info(
                    "Batch %d — written so far: %d definitions (skipped: %d)",
                    batch_num, total_written, total_skipped,
                )

    driver.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Ingestion complete.")
    log.info("  Definition nodes written : %d", total_written)
    log.info("  Files skipped (invalid)  : %d", total_skipped)
    log.info("  Total files processed    : %d", total_written + total_skipped)
    log.info("=" * 60)
    log.info("Ingestion complete.")
    log.info("  Definition nodes written : %d", total_written)
    log.info("  Files skipped (invalid)  : %d", total_skipped)
    log.info("  Total files processed    : %d", total_written + total_skipped)
    log.info("=" * 60)
    log.info(
        "Verify in Neo4j Browser:\n"
        "  MATCH (d:Definition) RETURN count(d);\n"
        "  MATCH (k:Keyword)-[:HAS_DEFINITION]->(d:Definition) "
        "RETURN k.text_ru, d.definition_ru LIMIT 10;"
    )


if __name__ == "__main__":
    main()
