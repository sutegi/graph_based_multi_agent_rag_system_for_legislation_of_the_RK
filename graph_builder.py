#!/usr/bin/env python3
"""
graph_builder.py  —  Neo4j Legal Knowledge Graph Builder
=========================================================
Three-stage pipeline for Kazakhstani legislation (RU + KZ, bilingual).

  Stage 1  Parse & Merge
    Reads  data/merged/<codex>_merged.json
           data/llm_results/<codex>/keywords.json
           data/llm_results/<codex>/relations.json
    Writes data/hierarchical_graph_base/final_hierarchy_and_keywords.json
           data/hierarchical_graph_base/final_horizontal_relations.json

  Stage 2  Build Hierarchy in Neo4j
    (:Code)-[:CONTAINS]->(:Section)-[:CONTAINS]->(:Chapter)
           -[:CONTAINS]->(:Paragraph)-[:CONTAINS]->(:Article)
    Missing levels are skipped; chain adapts automatically per article.

  Stage 3  Keywords & Horizontal Relations
    (:Article)-[:HAS_KEYWORD]->(:Keyword)
    (:Article)-[:BASED_ON|CAUSED_BY|CORRELATED_WITH|
                 EXTENDS|MANIFESTED_IN|REFERENCES]->(:Article)

Requirements:
    pip install neo4j python-dotenv

.env:
    NEO4J_URI      = bolt://localhost:7687
    NEO4J_USER     = neo4j
    NEO4J_PASSWORD = your_password
    NEO4J_DATABASE = neo4j          # optional, default neo4j

Usage:
    python graph_builder.py               # all stages
    python graph_builder.py --stage 1     # stage 1 only
    python graph_builder.py --stage 2 3   # stages 2 and 3
    python graph_builder.py --data-root ./data --env .env
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("graph_builder")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 500

RELATION_TYPES: list[str] = [
    "BASED_ON",
    "CAUSED_BY",
    "CORRELATED_WITH",
    "EXTENDS",
    "MANIFESTED_IN",
    "REFERENCES",
]

# Display names (ru, kz) for each codex prefix
CODEX_NAMES: dict[str, tuple[str, str]] = {
    "admin_offenses":  (
        "Кодекс об административных правонарушениях",
        "Әкімшілік құқық бұзушылықтар туралы кодекс",
    ),
    "admin_proc": (
        "Административный процедурно-процессуальный кодекс",
        "Әкімшілік рәсімдік-процестік кодекс",
    ),
    "budget": (
        "Бюджетный кодекс",
        "Бюджет кодексі",
    ),
    "civil_general": (
        "Гражданский кодекс (Общая часть)",
        "Азаматтық кодекс (Жалпы бөлім)",
    ),
    "civil_proc": (
        "Гражданский процессуальный кодекс",
        "Азаматтық процестік кодекс",
    ),
    "civil_special": (
        "Гражданский кодекс (Особенная часть)",
        "Азаматтық кодекс (Ерекше бөлім)",
    ),
    "construction": (
        "Градостроительный кодекс",
        "Қала құрылысы кодексі",
    ),
    "criminal": (
        "Уголовный кодекс",
        "Қылмыстық кодекс",
    ),
    "criminal_exec": (
        "Уголовно-исполнительный кодекс",
        "Қылмыстық-атқару кодексі",
    ),
    "criminal_proc": (
        "Уголовно-процессуальный кодекс",
        "Қылмыстық-процестік кодекс",
    ),
    "customs": (
        "Кодекс о таможенном регулировании",
        "Кедендік реттеу туралы кодекс",
    ),
    "digital": (
        "Цифровой кодекс",
        "Цифрлық кодекс",
    ),
    "entrepreneurial": (
        "Предпринимательский кодекс",
        "Кәсіпкерлік кодекс",
    ),
    "environmental": (
        "Экологический кодекс",
        "Экологиялық кодекс",
    ),
    "family": (
        "Кодекс о браке (супружестве) и семье",
        "Неке (ерлі-зайыптылық) және отбасы туралы кодекс",
    ),
    "forest": (
        "Лесной кодекс",
        "Орман кодексі",
    ),
    "health": (
        "Кодекс о здоровье народа и системе здравоохранения",
        "Халық денсаулығы және денсаулық сақтау жүйесі туралы кодекс",
    ),
    "labor": (
        "Трудовой кодекс",
        "Еңбек кодексі",
    ),
    "land": (
        "Земельный кодекс",
        "Жер кодексі",
    ),
    "social": (
        "Социальный кодекс",
        "Әлеуметтік кодекс",
    ),
    "subsoil": (
        "Кодекс о недрах и недропользовании",
        "Жер қойнауы және жер қойнауын пайдалану туралы кодекс",
    ),
    "tax_code": (
        "Налоговый кодекс",
        "Салық кодексі",
    ),
    "tax_payments": (
        "Кодекс о налогах и других обязательных платежах в бюджет",
        "Салықтар және бюджетке төленетін басқа да міндетті төлемдер туралы кодекс",
    ),
    "water": (
        "Водный кодекс",
        "Су кодексі",
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _make_id(*parts: str) -> str:
    """Deterministic 16-char ID from arbitrary string parts."""
    key = "||".join(p or "" for p in parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _batched(iterable, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Any, path: Path, label: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    count = len(obj) if isinstance(obj, (list, dict)) else "?"
    logger.info("  Saved %-52s  (%s entries)", path.name, count)


def _neo4j_driver(cfg: dict) -> Driver:
    return GraphDatabase.driver(
        cfg["uri"],
        auth=(cfg["user"], cfg["password"]),
    )


# ---------------------------------------------------------------------------
# STAGE 1 — Parse hierarchy and merge all data
# ---------------------------------------------------------------------------

def stage1(merged_dir: Path, llm_dir: Path, graph_dir: Path) -> None:
    logger.info("=" * 60)
    logger.info("STAGE 1  Parse & Merge")
    logger.info("=" * 60)

    # Discover matched codex pairs
    merged_prefixes = {
        p.stem.removesuffix("_merged")
        for p in merged_dir.glob("*_merged.json")
    }
    llm_prefixes = {d.name for d in llm_dir.iterdir() if d.is_dir()}
    codexes = sorted(merged_prefixes & llm_prefixes)
    only_merged = merged_prefixes - llm_prefixes
    only_llm    = llm_prefixes - merged_prefixes
    if only_merged:
        logger.warning("Merged-only (no LLM results): %s", sorted(only_merged))
    if only_llm:
        logger.warning("LLM-only (no merged file): %s", sorted(only_llm))
    logger.info("Processing %d codex pair(s): %s", len(codexes), ", ".join(codexes))

    # ---- Accumulators ----
    # seen_struct: node_id → structural node record (Code/Section/Chapter/Paragraph)
    seen_struct: dict[str, dict] = {}
    all_articles: list[dict]     = []
    all_relations: list[dict]    = []
    seen_rels: set[tuple]        = set()

    stats = dict(codes=0, sections=0, chapters=0, paragraphs=0,
                 articles=0, keywords=0, relations=0, rel_duped=0)

    for prefix in codexes:
        name_ru, name_kz = CODEX_NAMES.get(prefix, (prefix, prefix))
        code_id = _make_id("code", prefix)

        # ---- Code node ----
        if code_id not in seen_struct:
            seen_struct[code_id] = dict(
                id=code_id, label="Code",
                prefix=prefix,
                name_ru=name_ru, name_kz=name_kz,
                parent_id=None, parent_label=None,
            )
            stats["codes"] += 1

        # ---- Load keywords ----
        # Supported formats:
        #   dict: {article_id: {ru:[...], kz:[...]}, ...}   <- actual format
        #   list: [[article_id, {ru:[...], kz:[...]}], ...]  <- fallback
        kw_map: dict[str, dict] = {}
        kw_path = llm_dir / prefix / "keywords.json"
        if kw_path.exists():
            raw_kw = _load_json(kw_path)
            if isinstance(raw_kw, dict):
                kw_map = raw_kw
            elif isinstance(raw_kw, list):
                for entry in raw_kw:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        kw_map[entry[0]] = entry[1]
        else:
            logger.warning("  [%s] keywords.json not found", prefix)

        # ---- Load relations ----
        rel_path = llm_dir / prefix / "relations.json"
        if rel_path.exists():
            for rel in _load_json(rel_path):
                key = (rel.get("source_id"), rel.get("target_id"), rel.get("type"))
                if None in key:
                    continue
                if key in seen_rels:
                    stats["rel_duped"] += 1
                    continue
                seen_rels.add(key)
                all_relations.append(rel)
                stats["relations"] += 1
        else:
            logger.warning("  [%s] relations.json not found", prefix)

        # ---- Load merged articles ----
        articles = _load_json(merged_dir / f"{prefix}_merged.json")

        for art in articles:
            art_id = art["article_id"]

            # Bilingual field helpers
            def _bi(field: str, lang: str) -> str | None:
                return (art.get(field) or {}).get(lang)

            section_ru = _bi("section",   "ru")
            section_kz = _bi("section",   "kz")
            chapter_ru = _bi("chapter",   "ru")
            chapter_kz = _bi("chapter",   "kz")
            para_ru    = _bi("paragraph", "ru")
            para_kz    = _bi("paragraph", "kz")
            title_ru   = _bi("title",     "ru")
            title_kz   = _bi("title",     "kz")
            body_ru    = _bi("context",   "ru")
            body_kz    = _bi("context",   "kz")

            # ---- Section node ----
            section_id: str | None = None
            if section_ru:
                section_id = _make_id("section", prefix, section_ru)
                if section_id not in seen_struct:
                    seen_struct[section_id] = dict(
                        id=section_id, label="Section",
                        name_ru=section_ru, name_kz=section_kz or section_ru,
                        parent_id=code_id, parent_label="Code",
                    )
                    stats["sections"] += 1

            # ---- Chapter node ----
            chapter_id: str | None = None
            if chapter_ru:
                chapter_id = _make_id("chapter", prefix, chapter_ru)
                if chapter_id not in seen_struct:
                    ch_parent_id    = section_id or code_id
                    ch_parent_label = "Section" if section_id else "Code"
                    seen_struct[chapter_id] = dict(
                        id=chapter_id, label="Chapter",
                        name_ru=chapter_ru, name_kz=chapter_kz or chapter_ru,
                        parent_id=ch_parent_id, parent_label=ch_parent_label,
                    )
                    stats["chapters"] += 1

            # ---- Paragraph node ----
            para_id: str | None = None
            if para_ru:
                para_id = _make_id("paragraph", prefix, para_ru)
                if para_id not in seen_struct:
                    p_parent_id = chapter_id or section_id or code_id
                    p_parent_label = (
                        "Chapter"  if chapter_id else
                        "Section"  if section_id else
                        "Code"
                    )
                    seen_struct[para_id] = dict(
                        id=para_id, label="Paragraph",
                        name_ru=para_ru, name_kz=para_kz or para_ru,
                        parent_id=p_parent_id, parent_label=p_parent_label,
                    )
                    stats["paragraphs"] += 1

            # ---- Article parent (closest non-null ancestor) ----
            art_parent_id = para_id or chapter_id or section_id or code_id
            art_parent_label = (
                "Paragraph" if para_id    else
                "Chapter"   if chapter_id else
                "Section"   if section_id else
                "Code"
            )

            # ---- Keywords — pair ru/kz by index ----
            kw_entry = kw_map.get(art_id, {})
            kw_ru: list[str] = kw_entry.get("ru") or []
            kw_kz: list[str] = kw_entry.get("kz") or []
            keywords: list[dict] = []
            for i, ru_text in enumerate(kw_ru):
                keywords.append({
                    "text_ru": ru_text,
                    "text_kz": kw_kz[i] if i < len(kw_kz) else None,
                })
            # kz surplus (shouldn't happen, but guard)
            for j in range(len(kw_ru), len(kw_kz)):
                keywords.append({"text_ru": None, "text_kz": kw_kz[j]})
            stats["keywords"] += len(keywords)

            all_articles.append(dict(
                id=art_id,
                codex_prefix=prefix,
                number=art.get("number"),
                source=art.get("source"),
                name_ru=title_ru,
                name_kz=title_kz,
                body_ru=body_ru,
                body_kz=body_kz,
                parent_id=art_parent_id,
                parent_label=art_parent_label,
                keywords=keywords,
            ))
            stats["articles"] += 1

    # ---- Save outputs ----
    _save_json(
        {"structural_nodes": list(seen_struct.values()), "articles": all_articles},
        graph_dir / "final_hierarchy_and_keywords.json",
    )
    _save_json(all_relations, graph_dir / "final_horizontal_relations.json")

    logger.info(
        "Stage 1 done: %d Codes | %d Sections | %d Chapters | %d Paragraphs | "
        "%d Articles | %d Keywords | %d Relations (%d duplicates removed)",
        stats["codes"], stats["sections"], stats["chapters"], stats["paragraphs"],
        stats["articles"], stats["keywords"],
        stats["relations"], stats["rel_duped"],
    )


# ---------------------------------------------------------------------------
# STAGE 2 — Build hierarchical graph in Neo4j
# ---------------------------------------------------------------------------

def _create_constraints(session, labels: list[str]) -> None:
    """Create unique constraints for each node label (Neo4j 4.4+ syntax)."""
    for label in labels:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        )
    logger.info("  Constraints ensured for: %s", ", ".join(labels))


def _merge_nodes(session, label: str, rows: list[dict], extra_set: str = "") -> None:
    """MERGE nodes of a given label in one UNWIND batch."""
    set_clause = "n.name_ru = row.name_ru, n.name_kz = row.name_kz"
    if extra_set:
        set_clause += f", {extra_set}"
    session.run(
        f"UNWIND $rows AS row "
        f"MERGE (n:{label} {{id: row.id}}) "
        f"SET {set_clause}",
        rows=rows,
    )


def _merge_contains(session, parent_label: str, child_label: str, rows: list[dict]) -> None:
    """MERGE [:CONTAINS] edges between existing parent/child nodes."""
    session.run(
        f"UNWIND $rows AS row "
        f"MATCH (p:{parent_label} {{id: row.parent_id}}) "
        f"MATCH (c:{child_label}  {{id: row.id}}) "
        f"MERGE (p)-[:CONTAINS]->(c)",
        rows=rows,
    )


def stage2(graph_dir: Path, neo4j_cfg: dict) -> None:
    logger.info("=" * 60)
    logger.info("STAGE 2  Build Hierarchy in Neo4j")
    logger.info("=" * 60)

    data_path = graph_dir / "final_hierarchy_and_keywords.json"
    if not data_path.exists():
        logger.error("Stage 1 output not found: %s — run Stage 1 first.", data_path)
        sys.exit(1)

    raw = _load_json(data_path)
    structural_nodes: list[dict] = raw["structural_nodes"]
    articles: list[dict]         = raw["articles"]

    # Bucket structural nodes by label
    by_label: dict[str, list[dict]] = {
        lbl: [] for lbl in ("Code", "Section", "Chapter", "Paragraph")
    }
    for node in structural_nodes:
        lbl = node.get("label", "")
        if lbl in by_label:
            by_label[lbl].append(node)

    driver = _neo4j_driver(neo4j_cfg)
    db     = neo4j_cfg.get("database", "neo4j")
    stats  = {lbl: 0 for lbl in ("Code", "Section", "Chapter", "Paragraph", "Article")}

    try:
        with driver.session(database=db) as session:

            # ---- Constraints ----
            _create_constraints(
                session,
                ["Code", "Section", "Chapter", "Paragraph", "Article"],
            )

            # ---- Pass 1: MERGE all structural nodes ----
            for label in ("Code", "Section", "Chapter", "Paragraph"):
                nodes = by_label[label]
                if not nodes:
                    continue
                logger.info("  Merging %d %s nodes...", len(nodes), label)
                extra = "n.prefix = row.prefix" if label == "Code" else ""
                for batch in _batched(nodes, BATCH_SIZE):
                    _merge_nodes(session, label, batch, extra)
                stats[label] = len(nodes)

            # ---- Pass 2: MERGE [:CONTAINS] for structural nodes ----
            for label in ("Section", "Chapter", "Paragraph"):
                nodes = by_label[label]
                if not nodes:
                    continue
                logger.info("  Linking %d %s → parent...", len(nodes), label)
                # Group by parent_label because Cypher label must be static
                by_parent: dict[str, list[dict]] = {}
                for n in nodes:
                    by_parent.setdefault(n["parent_label"], []).append(n)
                for parent_label, rows in by_parent.items():
                    for batch in _batched(rows, BATCH_SIZE):
                        _merge_contains(session, parent_label, label, batch)

            # ---- Pass 3: MERGE Article nodes ----
            logger.info("  Merging %d Article nodes...", len(articles))
            for batch in _batched(articles, BATCH_SIZE):
                session.run(
                    "UNWIND $rows AS row "
                    "MERGE (n:Article {id: row.id}) "
                    "SET n.number       = row.number, "
                    "    n.name_ru      = row.name_ru, "
                    "    n.name_kz      = row.name_kz, "
                    "    n.body_ru      = row.body_ru, "
                    "    n.body_kz      = row.body_kz, "
                    "    n.source       = row.source, "
                    "    n.codex_prefix = row.codex_prefix",
                    rows=batch,
                )
            stats["Article"] = len(articles)

            # ---- Pass 4: MERGE [:CONTAINS] for Articles ----
            logger.info("  Linking %d Articles → parent...", len(articles))
            by_parent_art: dict[str, list[dict]] = {}
            for art in articles:
                by_parent_art.setdefault(art["parent_label"], []).append(art)
            for parent_label, rows in by_parent_art.items():
                for batch in _batched(rows, BATCH_SIZE):
                    _merge_contains(session, parent_label, "Article", batch)

    finally:
        driver.close()

    logger.info(
        "Stage 2 done: %d Codes | %d Sections | %d Chapters | "
        "%d Paragraphs | %d Articles",
        stats["Code"], stats["Section"], stats["Chapter"],
        stats["Paragraph"], stats["Article"],
    )


# ---------------------------------------------------------------------------
# STAGE 3 — Keywords & Horizontal Relations
# ---------------------------------------------------------------------------

def stage3(graph_dir: Path, neo4j_cfg: dict) -> None:
    logger.info("=" * 60)
    logger.info("STAGE 3  Keywords & Horizontal Relations")
    logger.info("=" * 60)

    data_path = graph_dir / "final_hierarchy_and_keywords.json"
    rel_path  = graph_dir / "final_horizontal_relations.json"
    for p in (data_path, rel_path):
        if not p.exists():
            logger.error("Stage 1 output not found: %s — run Stage 1 first.", p)
            sys.exit(1)

    articles  = _load_json(data_path)["articles"]
    relations = _load_json(rel_path)

    driver = _neo4j_driver(neo4j_cfg)
    db     = neo4j_cfg.get("database", "neo4j")

    kw_total  = 0
    rel_total = 0
    rel_skip  = 0

    try:
        with driver.session(database=db) as session:

            # ---- Index on Keyword.text_ru for fast lookups ----
            session.run(
                "CREATE INDEX keyword_text_ru IF NOT EXISTS "
                "FOR (n:Keyword) ON (n.text_ru)"
            )

            # ---- Keywords ----
            # Each keyword becomes its own node (unique context per article).
            kw_rows: list[dict] = []
            for art in articles:
                art_id = art["id"]
                for kw in art.get("keywords") or []:
                    kw_rows.append({
                        "article_id": art_id,
                        "text_ru":    kw.get("text_ru"),
                        "text_kz":    kw.get("text_kz"),
                    })

            logger.info("  Importing %d keyword nodes...", len(kw_rows))
            for batch in _batched(kw_rows, BATCH_SIZE):
                session.run(
                    "UNWIND $rows AS row "
                    "MATCH (a:Article {id: row.article_id}) "
                    "CREATE (k:Keyword {text_ru: row.text_ru, text_kz: row.text_kz}) "
                    "CREATE (a)-[:HAS_KEYWORD]->(k)",
                    rows=batch,
                )
                kw_total += len(batch)
            logger.info("  Keywords imported: %d", kw_total)

            # ---- Horizontal relations ----
            # Fetch known article IDs to skip dangling refs
            logger.info("  Fetching known Article IDs from graph...")
            known_ids: set[str] = {
                rec["id"]
                for rec in session.run("MATCH (a:Article) RETURN a.id AS id")
            }
            logger.info("  Known articles in graph: %d", len(known_ids))

            # Group by relation type (static label in Cypher)
            by_type: dict[str, list[dict]] = {t: [] for t in RELATION_TYPES}
            for rel in relations:
                rtype = rel.get("type")
                src   = rel.get("source_id")
                tgt   = rel.get("target_id")

                if rtype not in by_type:
                    logger.warning("  Unknown relation type %r — skipped", rtype)
                    rel_skip += 1
                    continue
                if src not in known_ids or tgt not in known_ids:
                    logger.debug("  Dangling ref: src=%s tgt=%s — skipped", src, tgt)
                    rel_skip += 1
                    continue

                meta = rel.get("metadata") or {}
                by_type[rtype].append({
                    "source_id":    src,
                    "target_id":    tgt,
                    "evidence":     rel.get("evidence") or "",
                    "source_kz":    meta.get("source_kz"),
                    "target_kz":    meta.get("target_kz"),
                    "ref_type":     meta.get("ref_type"),      # internal / external
                    "target_label": meta.get("target_label"),
                })

            for rtype, rows in by_type.items():
                if not rows:
                    continue
                logger.info("  Importing %d %s relations...", len(rows), rtype)
                for batch in _batched(rows, BATCH_SIZE):
                    session.run(
                        f"UNWIND $rows AS row "
                        f"MATCH (src:Article {{id: row.source_id}}) "
                        f"MATCH (tgt:Article {{id: row.target_id}}) "
                        f"MERGE (src)-[r:{rtype}]->(tgt) "
                        f"SET r.evidence     = row.evidence, "
                        f"    r.source_kz    = row.source_kz, "
                        f"    r.target_kz    = row.target_kz, "
                        f"    r.target_label = row.target_label, "
                        f"    r.ref_type     = CASE "
                        f"        WHEN row.ref_type IS NOT NULL "
                        f"        THEN row.ref_type ELSE r.ref_type END",
                        rows=batch,
                    )
                    rel_total += len(batch)

    finally:
        driver.close()

    logger.info(
        "Stage 3 done: %d Keywords | %d Relations imported | %d skipped",
        kw_total, rel_total, rel_skip,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Neo4j legal knowledge graph from merged legislation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stage", "-s",
        nargs="+",
        choices=["1", "2", "3", "all"],
        default=["all"],
        metavar="STAGE",
        help="Stage(s) to run: 1=parse, 2=hierarchy, 3=keywords+relations, all=1+2+3.",
    )
    p.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory (must contain merged/ and llm_results/).",
    )
    p.add_argument(
        "--env", type=Path, default=Path(".env"),
        help="Path to .env file with NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env)

    data_root  = args.data_root
    merged_dir = data_root / "merged"
    llm_dir    = data_root / "llm_results"
    graph_dir  = data_root / "hierarchical_graph_base"

    neo4j_cfg = {
        "uri":      os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
        "user":     os.getenv("NEO4J_USER",     "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", ""),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }

    stages = set(args.stage)
    run_all = "all" in stages

    if run_all or "1" in stages:
        stage1(merged_dir, llm_dir, graph_dir)

    if run_all or "2" in stages:
        stage2(graph_dir, neo4j_cfg)

    if run_all or "3" in stages:
        stage3(graph_dir, neo4j_cfg)

    logger.info("All requested stages complete.")


if __name__ == "__main__":
    main()
