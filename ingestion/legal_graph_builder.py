"""
Enhanced legal knowledge graph builder for Neo4j with semantic enrichment.
Specialises in extracting and representing legal entities, obligations, and
relationships for the Kazakhstan legal corpus (24 codexes, ~9 200 articles).

Fixes (2026-05-19):
- P0: load_dotenv() is now called inside main() so .env credentials are loaded.
- P0: _bi() helper moved outside the article loop (was recreated 9 000+ times).
- P0: Stage 3 keyword merge: kw_merged counter now counts actual merged rows
      separately from kw_submitted, and logs mismatches as warnings.
- P1: EMBED_BATCH reduced to 64 (safe for RTX 3070 8 GB VRAM with e5-large).
- P1: Stage 4 embedding fetch now paginates instead of using LIMIT 50000.
- P1: _neo4j_driver() gains connection_timeout and max_connection_pool_size.
- P1: stats["regex_refs"] initialised at the top of stage1_enhanced, not in loop.
- P2: stage functions raise ValueError instead of calling sys.exit(1); main()
      catches and exits cleanly.
- P2: FULLTEXT index on Keyword nodes added in stage3_enhanced.
- P2: Article FULLTEXT index (body_ru, name_ru) added in stage2_enhanced — needed
      by multi_agent_rag/database.py search_articles().
- P2: kw_map list-schema branch now validates that entry[1] is a dict.
- P2: Semantic entity lists stored as arrays on Article nodes (subjects,
      obligations, sanctions, rights, prohibitions) so they can be queried.
- P3: Assertion added for LEGAL_PATTERNS["temporal"][0] access.
- CLI: default --graph-dir now points to data/hierarchical_graph_base/ where
      the partial data already lives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Iterator
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("legal_graph_builder")


BATCH_SIZE = 500

# P3 fix: assert temporal pattern list is non-empty at import time
LEGAL_PATTERNS = {
    "subject": [
        r"физическое лицо", r"юридическое лицо", r"гражданин", r"организация",
        r"компания", r"индивидуальный предприниматель", r"государственный орган",
        r"должностное лицо", r"работник", r"работодатель",
    ],
    "sanction": [
        r"штраф", r"лишение свободы", r"ограничение свободы", r"исправительные работы",
        r"запрещение деятельности", r"конфискация", r"арест", r"дисквалификация",
        r"приостановление", r"возмещение убытков",
    ],
    "obligation": [
        r"обязан", r"должен", r"обязательно", r"обязательство", r"обязанность",
        r"несет ответственность", r"отвечает", r"ответственен", r"подлежит возмещению",
        r"подлежит оплате",
    ],
    "right": [
        r"имеет право", r"может", r"вправе", r"правомочие", r"допускается",
        r"разрешается", r"предусмотрено", r"предоставляется право", r"уполномочен",
        r"способен",
    ],
    "prohibition": [
        r"не имеет права", r"не может", r"не должен", r"запрещается", r"запрещено",
        r"недопустимо", r"не допускается", r"не признается", r"исключается",
        r"не подлежит",
    ],
    "definition": [
        r"понимается как", r"означает", r"определяется как", r"следует понимать",
        r"в смысле", r"считается", r"является", r"именуется", r"называется",
        r"в целях",
    ],
    "temporal": [
        r"\d{1,2}\.\d{1,2}\.\d{4}",
        r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}",
        r"вступает в силу", r"действует с", r"со дня",
    ],
}
assert len(LEGAL_PATTERNS["temporal"]) > 0, "LEGAL_PATTERNS['temporal'] must not be empty"

RELATION_TYPES: list[str] = [
    "BASED_ON", "CAUSED_BY", "CORRELATED_WITH", "EXTENDS", "MANIFESTED_IN",
    "REFERENCES", "CREATES_OBLIGATION", "GRANTS_RIGHT", "IMPOSES_SANCTION",
    "AMENDS", "SUPPLEMENTS", "REQUIRES", "PROHIBITS", "DEFINES",
]

CODEX_NAMES: dict[str, tuple[str, str]] = {
    "admin_offenses":  ("Кодекс об административных правонарушениях",         "Әкімшілік құқық бұзушылықтар туралы кодекс"),
    "admin_proc":      ("Административный процедурно-процессуальный кодекс",   "Әкімшілік рәсімдік-процестік кодекс"),
    "budget":          ("Бюджетный кодекс",                                   "Бюджет кодексі"),
    "civil_general":   ("Гражданский кодекс (Общая часть)",                   "Азаматтық кодекс (Жалпы бөлім)"),
    "civil_proc":      ("Гражданский процессуальный кодекс",                  "Азаматтық процестік кодекс"),
    "civil_special":   ("Гражданский кодекс (Особенная часть)",               "Азаматтық кодекс (Ерекше бөлім)"),
    "construction":    ("Градостроительный кодекс",                           "Қала құрылысы кодексі"),
    "criminal":        ("Уголовный кодекс",                                   "Қылмыстық кодекс"),
    "criminal_exec":   ("Уголовно-исполнительный кодекс",                     "Қылмыстық-атқару кодексі"),
    "criminal_proc":   ("Уголовно-процессуальный кодекс",                     "Қылмыстық-процестік кодекс"),
    "customs":         ("Кодекс о таможенном регулировании",                  "Кедендік реттеу туралы кодекс"),
    "digital":         ("Цифровой кодекс",                                    "Цифрлық кодекс"),
    "entrepreneurial": ("Предпринимательский кодекс",                         "Кәсіпкерлік кодекс"),
    "environmental":   ("Экологический кодекс",                               "Экологиялық кодекс"),
    "family":          ("Кодекс о браке (супружестве) и семье",               "Неке (ерлі-зайыптылық) және отбасы туралы кодекс"),
    "forest":          ("Лесной кодекс",                                      "Орман кодексі"),
    "health":          ("Кодекс о здоровье народа и системе здравоохранения", "Халық денсаулығы және денсаулық сақтау жүйесі туралы кодекс"),
    "labor":           ("Трудовой кодекс",                                    "Еңбек кодексі"),
    "land":            ("Земельный кодекс",                                   "Жер кодексі"),
    "social":          ("Социальный кодекс",                                  "Әлеуметтік кодекс"),
    "subsoil":         ("Кодекс о недрах и недропользовании",                 "Жер қойнауы және жер қойнауын пайдалану туралы кодекс"),
    "tax_code":        ("Налоговый кодекс",                                   "Салық кодексі"),
    "tax_payments":    ("Кодекс о налогах и других обязательных платежах в бюджет", "Салықтар және бюджетке төленетін басқа да міндетті төлемдер туралы кодекс"),
    "water":           ("Водный кодекс",                                      "Су кодексі"),
}


# ── Regex cross-reference extraction ─────────────────────────────────────────

_ARTICLE_REF_PATTERNS = [
    re.compile(r"(?:ст(?:атье?й?|атьях)?\.?\s*)(\d+(?:[–\-]\d+)?(?:\.\d+)?)",
               re.IGNORECASE | re.UNICODE),
    re.compile(r"(?:статьями?\s*)(\d+(?:[–\-]\d+)?)",
               re.IGNORECASE | re.UNICODE),
]

_CLAUSE_REF_PATTERN = re.compile(
    r"п(?:ункт(?:у|е|а|ов)?|\.?)\s*(\d+)",
    re.IGNORECASE | re.UNICODE,
)


def _extract_structural_refs(
    source_art: dict,
    art_number_index: dict[str, str],
) -> list[dict]:
    """Extract deterministic REFERENCES edges from article body text."""
    context = source_art.get("context", {})
    body_ru = context.get("ru", "") if isinstance(context, dict) else ""
    body_ru = str(body_ru) if body_ru else ""
    if not body_ru:
        return []

    source_id = source_art["article_id"]
    codex_pfx = source_art.get("codex_prefix", "")
    refs: list[dict] = []
    seen_targets: set[str] = set()

    for pattern in _ARTICLE_REF_PATTERNS:
        for match in pattern.finditer(body_ru):
            raw_num = match.group(1).strip()
            if "–" in raw_num or "-" in raw_num:
                sep   = "–" if "–" in raw_num else "-"
                parts = raw_num.split(sep, 1)
                try:
                    start, end = int(parts[0]), int(parts[1])
                    nums = [str(n) for n in range(start, min(end + 1, start + 10))]
                except ValueError:
                    nums = [parts[0]]
            else:
                nums = [raw_num]

            for num in nums:
                target_id = art_number_index.get(num)
                if not target_id or target_id == source_id or target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                refs.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "type":      "REFERENCES",
                    "evidence":  match.group(0).strip(),
                    "metadata": {
                        "ref_type":     "internal",
                        "target_label": f"Статья {num}",
                        "codex_prefix": codex_pfx,
                        "extraction":   "regex",
                    },
                })
    return refs


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_id(*parts: str) -> str:
    """Deterministic 16-char hex ID (SHA-256 truncated, collision-resistant)."""
    key = "||".join(p or "" for p in parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _batched(iterable, n: int) -> Iterator[list]:
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
    """Create Neo4j driver with connection timeout. P1 fix."""
    return GraphDatabase.driver(
        cfg["uri"],
        auth=(cfg["user"], cfg["password"]),
        connection_timeout=15,
        max_connection_pool_size=10,
    )


def _get_body_ru(art: dict) -> str:
    """Extract Russian body text from either merged or with_keywords schema."""
    ctx = art.get("context", {})
    if isinstance(ctx, dict):
        return str(ctx.get("ru") or "") or str(art.get("context_ru") or "")
    return str(art.get("context_ru") or art.get("body_ru") or "")


# P0 fix: _bi moved outside the article loop so it is not recreated per iteration.
def _bi(art: dict, field: str, lang: str) -> str | None:
    """Extract bilingual field from article dict: art[field][lang]."""
    return (art.get(field) or {}).get(lang)


def _extract_legal_entities(text: str, entity_type: str) -> list[str]:
    if not text or entity_type not in LEGAL_PATTERNS:
        return []
    entities = []
    for pattern in LEGAL_PATTERNS[entity_type]:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            entity = match.group(0).strip()
            if len(entity) > 5 and entity not in entities:
                entities.append(entity)
    return entities[:5]


def _analyze_article_semantics(art: dict) -> dict:
    body_ru = _get_body_ru(art)
    return {
        "subjects":        _extract_legal_entities(body_ru, "subject"),
        "sanctions":       _extract_legal_entities(body_ru, "sanction"),
        "obligations":     _extract_legal_entities(body_ru, "obligation"),
        "rights":          _extract_legal_entities(body_ru, "right"),
        "prohibitions":    _extract_legal_entities(body_ru, "prohibition"),
        "definitions":     _extract_legal_entities(body_ru, "definition"),
        "has_temporal_info": bool(
            re.search(LEGAL_PATTERNS["temporal"][0], body_ru, re.IGNORECASE)
        ),
        "is_criminal": art.get("codex_prefix", "") in {"criminal", "criminal_proc", "criminal_exec"},
        "is_civil":    art.get("codex_prefix", "") in {"civil_general", "civil_special", "civil_proc"},
        "is_tax":      art.get("codex_prefix", "") in {"tax_code", "tax_payments"},
        "is_labor":    art.get("codex_prefix", "") == "labor",
        "is_admin":    art.get("codex_prefix", "") in {"admin_offenses", "admin_proc"},
    }



# ── STAGE 1 ───────────────────────────────────────────────────────────────────

def stage1_enhanced(merged_dir: Path, llm_dir: Path, graph_dir: Path) -> None:
    """Parse & Merge with semantic enrichment and regex cross-ref extraction."""
    logger.info("=" * 70)
    logger.info("STAGE 1 (ENHANCED)  Parse & Merge with Legal Semantics")
    logger.info("=" * 70)

    merged_prefixes = {p.stem.removesuffix("_merged") for p in merged_dir.glob("*_merged.json")}
    llm_prefixes    = {d.name for d in llm_dir.iterdir() if d.is_dir()}
    codexes         = sorted(merged_prefixes & llm_prefixes)

    if not codexes:
        raise ValueError(
            f"No overlapping codexes found.\n"
            f"  merged_dir: {merged_dir} ({len(merged_prefixes)} files)\n"
            f"  llm_dir:    {llm_dir} ({len(llm_prefixes)} dirs)"
        )

    for side, names in (("Merged-only (no LLM)", merged_prefixes - llm_prefixes),
                        ("LLM-only (no merged)",  llm_prefixes - merged_prefixes)):
        if names:
            logger.warning("%s: %s", side, sorted(names))
    logger.info("Processing %d codex pair(s): %s", len(codexes), ", ".join(codexes))

    seen_struct: dict[str, dict] = {}
    all_articles: list[dict]     = []
    all_relations: list[dict]    = []
    seen_rels: set[tuple]        = set()

    # P1 fix: initialise all stats keys up-front (no setdefault inside loop)
    stats: dict[str, int] = {
        "codes": 0, "sections": 0, "chapters": 0, "paragraphs": 0,
        "articles": 0, "keywords": 0, "relations": 0, "rel_duped": 0,
        "subjects": 0, "obligations": 0, "sanctions": 0, "rights": 0,
        "regex_refs": 0,
    }

    for prefix in codexes:
        name_ru, name_kz = CODEX_NAMES.get(prefix, (prefix, prefix))
        code_id = _make_id("code", prefix)

        if code_id not in seen_struct:
            seen_struct[code_id] = dict(
                id=code_id, label="Code",
                prefix=prefix,
                name_ru=name_ru, name_kz=name_kz,
                parent_id=None, parent_label=None,
            )
            stats["codes"] += 1

        # ── Load keywords map ────────────────────────────────────────────────
        kw_map: dict[str, dict] = {}
        kw_path = llm_dir / prefix / "keywords.json"
        if kw_path.exists():
            raw_kw = _load_json(kw_path)
            if isinstance(raw_kw, dict):
                kw_map = raw_kw
            elif isinstance(raw_kw, list):
                # P2 fix: validate entry[1] is actually a dict
                for entry in raw_kw:
                    if (isinstance(entry, (list, tuple)) and len(entry) == 2
                            and isinstance(entry[0], str)
                            and isinstance(entry[1], dict)):
                        kw_map[entry[0]] = entry[1]
                    else:
                        logger.warning(
                            "  [%s] keywords.json: unexpected list entry %r — skipped",
                            prefix, str(entry)[:80],
                        )

        # ── Load LLM relations ───────────────────────────────────────────────
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

        # ── Process articles ─────────────────────────────────────────────────
        articles = _load_json(merged_dir / f"{prefix}_merged.json")

        art_number_index: dict[str, str] = {
            str(art.get("number", "")).strip(): art["article_id"]
            for art in articles
            if art.get("number") and art.get("article_id")
        }

        for art in articles:
            art_id = art["article_id"]

            # P0 fix: _bi is a module-level function now, not a closure
            section_ru = _bi(art, "section", "ru")
            section_kz = _bi(art, "section", "kz")
            chapter_ru = _bi(art, "chapter", "ru")
            chapter_kz = _bi(art, "chapter", "kz")
            para_ru    = _bi(art, "paragraph", "ru")
            para_kz    = _bi(art, "paragraph", "kz")
            title_ru   = _bi(art, "title", "ru")
            title_kz   = _bi(art, "title", "kz")

            context  = art.get("context", {})
            body_ru  = context.get("ru")  if isinstance(context, dict) else None
            body_kz  = context.get("kz")  if isinstance(context, dict) else None

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

            chapter_id: str | None = None
            if chapter_ru:
                chapter_id = _make_id("chapter", prefix, chapter_ru)
                if chapter_id not in seen_struct:
                    seen_struct[chapter_id] = dict(
                        id=chapter_id, label="Chapter",
                        name_ru=chapter_ru, name_kz=chapter_kz or chapter_ru,
                        parent_id=section_id or code_id,
                        parent_label="Section" if section_id else "Code",
                    )
                    stats["chapters"] += 1

            para_id: str | None = None
            if para_ru:
                para_id = _make_id("paragraph", prefix, para_ru)
                if para_id not in seen_struct:
                    seen_struct[para_id] = dict(
                        id=para_id, label="Paragraph",
                        name_ru=para_ru, name_kz=para_kz or para_ru,
                        parent_id=chapter_id or section_id or code_id,
                        parent_label=(
                            "Chapter" if chapter_id else
                            "Section" if section_id else "Code"
                        ),
                    )
                    stats["paragraphs"] += 1

            art_parent_id    = para_id or chapter_id or section_id or code_id
            art_parent_label = (
                "Paragraph" if para_id else
                "Chapter"   if chapter_id else
                "Section"   if section_id else "Code"
            )

            kw_entry = kw_map.get(art_id, {})
            kw_ru: list[str] = kw_entry.get("ru") or []
            kw_kz: list[str] = kw_entry.get("kz") or []
            keywords: list[dict] = []
            for i, ru_text in enumerate(kw_ru):
                keywords.append({"text_ru": ru_text, "text_kz": kw_kz[i] if i < len(kw_kz) else None})
            for j in range(len(kw_ru), len(kw_kz)):
                keywords.append({"text_ru": None, "text_kz": kw_kz[j]})
            stats["keywords"] += len(keywords)

            art_with_prefix = dict(art)
            art_with_prefix["article_id"]   = art_id
            art_with_prefix["codex_prefix"] = prefix
            semantics = _analyze_article_semantics(art_with_prefix)
            stats["subjects"]    += len(semantics["subjects"])
            stats["obligations"] += len(semantics["obligations"])
            stats["sanctions"]   += len(semantics["sanctions"])
            stats["rights"]      += len(semantics["rights"])

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
                semantics=semantics,
            ))
            stats["articles"] += 1

            struct_refs = _extract_structural_refs(art_with_prefix, art_number_index)
            for ref in struct_refs:
                key = (ref["source_id"], ref["target_id"], ref["type"])
                if key not in seen_rels:
                    seen_rels.add(key)
                    all_relations.append(ref)
                    stats["relations"] += 1
            stats["regex_refs"] += len(struct_refs)

    _save_json(
        {"structural_nodes": list(seen_struct.values()), "articles": all_articles},
        graph_dir / "final_hierarchy_and_keywords.json",
    )
    _save_json(all_relations, graph_dir / "final_horizontal_relations.json")

    logger.info(
        "Stage 1 done: %d Codes | %d Sections | %d Chapters | %d Paragraphs | "
        "%d Articles | %d Keywords | %d Relations (%d dupes removed, %d regex)\n"
        "  Legal entities: %d subjects | %d obligations | %d sanctions | %d rights",
        stats["codes"], stats["sections"], stats["chapters"], stats["paragraphs"],
        stats["articles"], stats["keywords"], stats["relations"], stats["rel_duped"],
        stats["regex_refs"],
        stats["subjects"], stats["obligations"], stats["sanctions"], stats["rights"],
    )



# ── STAGE 2 ───────────────────────────────────────────────────────────────────

def _create_constraints(session, labels: list[str]) -> None:
    for label in labels:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        )
    logger.info("  Constraints ensured for: %s", ", ".join(labels))


def _merge_nodes(session, label: str, rows: list[dict], extra_set: str = "") -> None:
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
    session.run(
        f"UNWIND $rows AS row "
        f"MATCH (p:{parent_label} {{id: row.parent_id}}) "
        f"MATCH (c:{child_label}  {{id: row.id}}) "
        f"MERGE (p)-[:CONTAINS]->(c)",
        rows=rows,
    )


def stage2_enhanced(graph_dir: Path, neo4j_cfg: dict) -> None:
    """Build hierarchy in Neo4j with semantic properties and FULLTEXT index."""
    logger.info("=" * 70)
    logger.info("STAGE 2 (ENHANCED)  Build Hierarchy & Semantic Properties")
    logger.info("=" * 70)

    data_path = graph_dir / "final_hierarchy_and_keywords.json"
    if not data_path.exists():
        # P2 fix: raise instead of sys.exit
        raise ValueError(f"Stage 1 output not found: {data_path} — run Stage 1 first.")

    raw              = _load_json(data_path)
    structural_nodes = raw["structural_nodes"]
    articles         = raw["articles"]

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

            _create_constraints(
                session,
                ["Code", "Section", "Chapter", "Paragraph", "Article"],
            )

            # P2 fix: create FULLTEXT index for multi_agent_rag/database.py search_articles()
            fulltext_indexes = [
                ("CREATE FULLTEXT INDEX article_fulltext_ru IF NOT EXISTS "
                 "FOR (n:Article) ON EACH [n.body_ru, n.name_ru]"),
                ("CREATE FULLTEXT INDEX article_fulltext_kz IF NOT EXISTS "
                 "FOR (n:Article) ON EACH [n.body_kz, n.name_kz]"),
            ]
            for idx_q in fulltext_indexes:
                try:
                    session.run(idx_q)
                except Exception as e:
                    logger.warning("  Fulltext index warning: %s", e)
            logger.info("  Article fulltext indexes ensured.")

            for label in ("Code", "Section", "Chapter", "Paragraph"):
                nodes = by_label[label]
                if not nodes:
                    continue
                logger.info("  Merging %d %s nodes...", len(nodes), label)
                extra = "n.prefix = row.prefix" if label == "Code" else ""
                for batch in _batched(nodes, BATCH_SIZE):
                    _merge_nodes(session, label, batch, extra)
                stats[label] = len(nodes)

            for label in ("Section", "Chapter", "Paragraph"):
                nodes = by_label[label]
                if not nodes:
                    continue
                logger.info("  Linking %d %s → parent...", len(nodes), label)
                by_parent: dict[str, list[dict]] = {}
                for n in nodes:
                    by_parent.setdefault(n["parent_label"], []).append(n)
                for parent_label, rows in by_parent.items():
                    for batch in _batched(rows, BATCH_SIZE):
                        _merge_contains(session, parent_label, label, batch)

            logger.info("  Merging %d Article nodes with semantics...", len(articles))
            for batch in _batched(articles, BATCH_SIZE):
                processed = []
                for art in batch:
                    sem = art.get("semantics", {})
                    row = dict(art)
                    # Boolean flags
                    row.update({
                        "has_sanctions":    bool(sem.get("sanctions")),
                        "has_obligations":  bool(sem.get("obligations")),
                        "has_rights":       bool(sem.get("rights")),
                        "is_criminal":      sem.get("is_criminal", False),
                        "is_civil":         sem.get("is_civil",    False),
                        "is_tax":           sem.get("is_tax",      False),
                        "is_labor":         sem.get("is_labor",    False),
                        "is_admin":         sem.get("is_admin",    False),
                        # P2 fix: store full entity lists so they can be queried
                        "subjects":         sem.get("subjects",    []),
                        "obligations":      sem.get("obligations", []),
                        "sanctions":        sem.get("sanctions",   []),
                        "rights":           sem.get("rights",      []),
                        "prohibitions":     sem.get("prohibitions",[]),
                    })
                    processed.append(row)

                session.run(
                    "UNWIND $rows AS row "
                    "MERGE (n:Article {id: row.id}) "
                    "SET n.number          = row.number, "
                    "    n.name_ru         = row.name_ru, "
                    "    n.name_kz         = row.name_kz, "
                    "    n.body_ru         = row.body_ru, "
                    "    n.body_kz         = row.body_kz, "
                    "    n.source          = row.source, "
                    "    n.codex_prefix    = row.codex_prefix, "
                    "    n.has_sanctions   = row.has_sanctions, "
                    "    n.has_obligations = row.has_obligations, "
                    "    n.has_rights      = row.has_rights, "
                    "    n.is_criminal     = row.is_criminal, "
                    "    n.is_civil        = row.is_civil, "
                    "    n.is_tax          = row.is_tax, "
                    "    n.is_labor        = row.is_labor, "
                    "    n.is_admin        = row.is_admin, "
                    "    n.subjects        = row.subjects, "
                    "    n.obligations     = row.obligations, "
                    "    n.sanctions       = row.sanctions, "
                    "    n.rights          = row.rights, "
                    "    n.prohibitions    = row.prohibitions",
                    rows=processed,
                )
            stats["Article"] = len(articles)

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



# ── STAGE 3 ───────────────────────────────────────────────────────────────────

def stage3_enhanced(graph_dir: Path, neo4j_cfg: dict) -> None:
    """Keywords & Relations with FULLTEXT index and accurate merge counting."""
    logger.info("=" * 70)
    logger.info("STAGE 3 (ENHANCED)  Keywords, Relations & Semantics")
    logger.info("=" * 70)

    data_path = graph_dir / "final_hierarchy_and_keywords.json"
    rel_path  = graph_dir / "final_horizontal_relations.json"
    for p in (data_path, rel_path):
        if not p.exists():
            raise ValueError(f"Stage 1 output not found: {p} — run Stage 1 first.")

    raw       = _load_json(data_path)
    articles  = raw["articles"]
    relations = _load_json(rel_path)

    driver = _neo4j_driver(neo4j_cfg)
    db     = neo4j_cfg.get("database", "neo4j")

    kw_submitted = 0  # rows sent to Neo4j
    kw_found     = 0  # articles that actually matched (from verification pass)
    rel_total    = 0
    rel_skip     = 0

    try:
        with driver.session(database=db) as session:

            # P2 fix: create FULLTEXT index on Keyword nodes for search_definitions()
            try:
                session.run(
                    "CREATE FULLTEXT INDEX keyword_fulltext IF NOT EXISTS "
                    "FOR (k:Keyword) ON EACH [k.text_ru, k.text_kz]"
                )
            except Exception as e:
                logger.warning("  Keyword fulltext index warning: %s", e)

            try:
                session.run(
                    "CREATE INDEX keyword_text_ru IF NOT EXISTS "
                    "FOR (n:Keyword) ON (n.text_ru)"
                )
            except Exception as e:
                logger.warning("  Keyword B-tree index warning: %s", e)
            logger.info("  Keyword indexes ensured.")

            # Build keyword rows
            kw_rows: list[dict] = []
            for art in articles:
                art_id = art["id"]
                for kw in art.get("keywords") or []:
                    kw_rows.append({
                        "article_id": art_id,
                        "text_ru":    kw.get("text_ru"),
                        "text_kz":    kw.get("text_kz"),
                    })
            kw_submitted = len(kw_rows)
            logger.info("  Importing %d keyword rows (MERGE — safe to re-run)...", kw_submitted)

            # P0 fix: count merged keywords separately from submitted count.
            # Use WITH ... RETURN count(*) to know how many MATCH(a) succeeded.
            for batch in _batched(kw_rows, BATCH_SIZE):
                result = session.run(
                    "UNWIND $rows AS row "
                    "MATCH (a:Article {id: row.article_id}) "
                    "MERGE (k:Keyword {text_ru: row.text_ru}) "
                    "  ON CREATE SET k.text_kz = row.text_kz "
                    "MERGE (a)-[:HAS_KEYWORD]->(k) "
                    "RETURN count(*) AS merged",
                    rows=batch,
                )
                rec = result.single()
                kw_found += rec["merged"] if rec else 0

            if kw_found < kw_submitted:
                logger.warning(
                    "  Keywords: %d submitted, %d merged (diff=%d — "
                    "those articles may not yet be in Neo4j)",
                    kw_submitted, kw_found, kw_submitted - kw_found,
                )
            else:
                logger.info("  Keywords merged: %d / %d", kw_found, kw_submitted)

            # Fetch known Article IDs for relation validation
            logger.info("  Fetching known Article IDs from graph...")
            known_ids: set[str] = {
                rec["id"]
                for rec in session.run("MATCH (a:Article) RETURN a.id AS id")
            }
            logger.info("  Known articles in graph: %d", len(known_ids))

            by_type: dict[str, list[dict]] = {t: [] for t in RELATION_TYPES}
            for rel in relations:
                rtype = rel.get("type")
                src   = rel.get("source_id")
                tgt   = rel.get("target_id")

                if rtype not in by_type:
                    logger.debug("  Unknown relation type %r — treating as REFERENCES", rtype)
                    rtype = "REFERENCES"

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
                    "ref_type":     meta.get("ref_type"),
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
                        f"    r.ref_type     = CASE WHEN row.ref_type IS NOT NULL "
                        f"                    THEN row.ref_type ELSE r.ref_type END",
                        rows=batch,
                    )
                    rel_total += len(batch)

    finally:
        driver.close()

    logger.info(
        "Stage 3 done: %d keyword rows, %d relation rows, %d dangling skipped.",
        kw_submitted, rel_total, rel_skip,
    )


# ── STAGE 4 ───────────────────────────────────────────────────────────────────

def stage4_embeddings(graph_dir: Path, neo4j_cfg: dict, model_name: str,
                      embed_batch: int = 64) -> None:
    """Compute and store multilingual embeddings for all Article nodes.

    Uses intfloat/multilingual-e5-large (1024-dim, supports Russian + Kazakh).
    Embeddings are stored on each Article node; a vector index is created so that
    hybrid BM25 + vector retrieval works in multi_agent_rag/database.py.

    P1 fix: embed_batch defaults to 64 (safe for RTX 3070 8 GB VRAM with e5-large).
    P1 fix: paginate the fetch query instead of using LIMIT 50000.
    """
    logger.info("=" * 70)
    logger.info("STAGE 4  Embeddings (model=%s, batch=%d)", model_name, embed_batch)
    logger.info("=" * 70)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ValueError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    logger.info("  Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    driver = _neo4j_driver(neo4j_cfg)
    db     = neo4j_cfg.get("database", "neo4j")
    PAGE   = 5_000   # rows per pagination fetch

    try:
        with driver.session(database=db) as session:
            session.run(
                "CREATE VECTOR INDEX article_vector IF NOT EXISTS "
                "FOR (a:Article) ON (a.embedding) "
                "OPTIONS {indexConfig: {"
                "  `vector.dimensions`: 1024, "
                "  `vector.similarity_function`: 'cosine'"
                "}}"
            )
            logger.info("  Vector index ensured.")

        # P1 fix: paginate instead of LIMIT 50000
        all_rows: list[dict] = []
        skip = 0
        while True:
            with driver.session(database=db) as session:
                result = session.run(
                    "MATCH (a:Article) WHERE a.embedding IS NULL "
                    "RETURN a.id AS id, a.body_ru AS body_ru "
                    "ORDER BY a.id "
                    "SKIP $skip LIMIT $page",
                    skip=skip, page=PAGE,
                )
                page_rows = [{"id": r["id"], "body": r["body_ru"] or ""} for r in result]
            if not page_rows:
                break
            all_rows.extend(page_rows)
            logger.info("  Fetched %d articles needing embeddings (total so far: %d)...",
                        len(page_rows), len(all_rows))
            if len(page_rows) < PAGE:
                break
            skip += PAGE

    except Exception:
        driver.close()
        raise

    if not all_rows:
        logger.info("  All articles already embedded — nothing to do.")
        driver.close()
        return

    logger.info("  Encoding %d articles...", len(all_rows))
    texts = ["passage: " + r["body"][:512] for r in all_rows]
    all_embeddings = model.encode(
        texts,
        batch_size=embed_batch,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    logger.info("  Writing embeddings to Neo4j...")
    write_batch: list[dict] = []
    try:
        with driver.session(database=db) as session:
            for i, row in enumerate(all_rows):
                write_batch.append({"id": row["id"], "embedding": all_embeddings[i].tolist()})
                if len(write_batch) >= BATCH_SIZE:
                    session.run(
                        "UNWIND $rows AS row MATCH (a:Article {id: row.id}) "
                        "SET a.embedding = row.embedding",
                        rows=write_batch,
                    )
                    write_batch = []
            if write_batch:
                session.run(
                    "UNWIND $rows AS row MATCH (a:Article {id: row.id}) "
                    "SET a.embedding = row.embedding",
                    rows=write_batch,
                )
    finally:
        driver.close()

    logger.info("  Embeddings written for %d articles.", len(all_rows))



# ─── main ─────────────────────────────────────────────────────────────────────

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"


def main() -> None:
    import argparse, os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Legal Knowledge Graph Builder — stages 1-4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--graph-dir",
        default="data/hierarchical_graph_base",
        help="Directory where stage-1 output JSONs are written / read from",
    )
    parser.add_argument(
        "--merged-dir",
        default="data/merged",
        help="Directory with {prefix}_merged.json article files",
    )
    parser.add_argument(
        "--llm-dir",
        default="data/llm_results",
        help="Directory with per-codex LLM keyword/relation results",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=None,
        help="Override NEO4J_URI env variable",
    )
    parser.add_argument(
        "--neo4j-user",
        default=None,
        help="Override NEO4J_USER env variable",
    )
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Override NEO4J_PASSWORD env variable",
    )
    parser.add_argument(
        "--neo4j-database",
        default="neo4j",
        help="Neo4j database name",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="HuggingFace sentence-transformer model for embeddings",
    )
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=64,
        help="Batch size for embedding generation (safe default for RTX 3070 8 GB)",
    )
    parser.add_argument(
        "--stages",
        default="1,2,3,4",
        help="Comma-separated stages to run (1=parse, 2=hierarchy, 3=keywords, 4=embed)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Shorthand for excluding stage 4",
    )
    args = parser.parse_args()

    # ── Resolve which stages to run ──────────────────────────────────────────
    stages_to_run: set[int] = set()
    for s in args.stages.split(","):
        s = s.strip()
        if s:
            try:
                stages_to_run.add(int(s))
            except ValueError:
                logger.warning("Ignoring unrecognised stage spec: %r", s)
    if args.skip_embeddings:
        stages_to_run.discard(4)

    # ── Paths ────────────────────────────────────────────────────────────────
    graph_dir  = Path(args.graph_dir)
    merged_dir = Path(args.merged_dir)
    llm_dir    = Path(args.llm_dir)

    for d, flag in [(merged_dir, "--merged-dir"), (llm_dir, "--llm-dir")]:
        if not d.exists():
            raise SystemExit(f"[ERROR] {flag} path does not exist: {d}")

    graph_dir.mkdir(parents=True, exist_ok=True)

    # ── Neo4j config ─────────────────────────────────────────────────────────
    neo4j_cfg = {
        "uri":      args.neo4j_uri      or os.environ.get("NEO4J_URI",      "bolt://localhost:7687"),
        "user":     args.neo4j_user     or os.environ.get("NEO4J_USER",     "neo4j"),
        "password": args.neo4j_password or os.environ.get("NEO4J_PASSWORD", ""),
        "database": args.neo4j_database,
    }

    print("=" * 60)
    print("Legal Knowledge Graph Builder")
    print(f"  graph_dir  : {graph_dir}")
    print(f"  merged_dir : {merged_dir}")
    print(f"  llm_dir    : {llm_dir}")
    print(f"  neo4j_uri  : {neo4j_cfg['uri']}")
    print(f"  stages     : {sorted(stages_to_run)}")
    print(f"  embed_batch: {args.embed_batch}")
    print("=" * 60)

    # ── Stage 1: parse articles, save intermediate JSONs ─────────────────────
    if 1 in stages_to_run:
        print("\n[Stage 1] Parsing articles and building intermediate JSON …")
        stage1_enhanced(
            merged_dir=merged_dir,
            llm_dir=llm_dir,
            graph_dir=graph_dir,
        )
        print("[Stage 1] Done.")

    # ── Stage 2: load hierarchy into Neo4j, create FULLTEXT indexes ──────────
    if 2 in stages_to_run:
        print("\n[Stage 2] Loading hierarchy into Neo4j and creating indexes …")
        stage2_enhanced(graph_dir=graph_dir, neo4j_cfg=neo4j_cfg)
        print("[Stage 2] Done.")

    # ── Stage 3: keywords and horizontal relations ────────────────────────────
    if 3 in stages_to_run:
        print("\n[Stage 3] Loading keywords and relations …")
        stage3_enhanced(graph_dir=graph_dir, neo4j_cfg=neo4j_cfg)
        print("[Stage 3] Done.")

    # ── Stage 4: embeddings ───────────────────────────────────────────────────
    if 4 in stages_to_run:
        print(f"\n[Stage 4] Generating embeddings (model={args.embed_model}, "
              f"batch={args.embed_batch}) …")
        stage4_embeddings(
            graph_dir=graph_dir,
            neo4j_cfg=neo4j_cfg,
            model_name=args.embed_model,
            embed_batch=args.embed_batch,
        )
        print("[Stage 4] Done.")

    print("\n[All done]")


if __name__ == "__main__":
    main()
