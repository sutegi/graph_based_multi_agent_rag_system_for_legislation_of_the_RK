"""Retrieval layer — weighted BM25 search, graph enrichment, and context building."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import (
    CONTEXT_MAX_CHARS,
    DEFINITION_KW_COUNT,
    DEFINITION_TOP_K,
    GRAPH_DECAY,
    GRAPH_LIMIT,
    GRAPH_SEED,
    LEGAL_CODEXES,
    TOP_K_BASE,
    logger,
)
from .database import bm25_search, fetch_definitions, graph_neighbours

if TYPE_CHECKING:
    from .llm import IntentResult


@dataclass
class ArticleScore:
    """Retrieved article with its combined relevance score."""

    id: str
    codex_prefix: str
    number: str
    name_ru: str
    name_kz: str
    body_ru: str
    combined_score: float
    source: str = "bm25"

    @property
    def codex_name(self) -> str:
        """Full Russian name of the codex this article belongs to."""
        names = LEGAL_CODEXES.get(self.codex_prefix or "", ("", ""))
        return names[0] or self.codex_prefix or ""

    def format_for_context(self) -> str:
        """Format the article as a labelled text block for the LLM context."""
        header = f"[{self.codex_name} | Статья {self.number}] {self.name_ru or ''}".rstrip()
        body = (self.body_ru or "").strip()
        return f"{header}\n{body}" if body else header


@dataclass
class RetrievalResult:
    """All outputs from one retrieval run."""

    articles: list[ArticleScore] = field(default_factory=list)
    definitions: dict[str, list[dict]] = field(default_factory=dict)
    context_text: str = ""
    stats: dict = field(default_factory=dict)


async def _search_keyword(
    text: str,
    weight: float,
    codex_slugs: list[str],
) -> list[tuple[dict, float]]:
    """BM25 search for one keyword; falls back to unfiltered if filtered returns nothing."""
    limit = max(5, round(TOP_K_BASE * (0.5 + weight)))
    rows: list[dict] = []
    if codex_slugs:
        rows = await bm25_search(text, limit=limit, codex_slugs=codex_slugs)
    if not rows:
        rows = await bm25_search(text, limit=limit)
    return [(row, (row.get("score") or 0.0) * weight) for row in rows]


async def retrieve(intent: "IntentResult") -> RetrievalResult:
    """Run the full retrieval pipeline for the given intent and return ranked articles with context."""
    keywords    = intent.keywords
    codex_slugs = intent.codex_slugs

    kw_tasks = [_search_keyword(kw.text, kw.weight, codex_slugs) for kw in keywords]
    results_per_kw: list[list[tuple[dict, float]]] = await asyncio.gather(*kw_tasks)

    score_map: dict[str, ArticleScore] = {}
    for pairs in results_per_kw:
        for row, score in pairs:
            aid = (row.get("id") or "").strip()
            if not aid:
                continue
            if aid not in score_map:
                score_map[aid] = ArticleScore(
                    id=aid,
                    codex_prefix=row.get("codex_prefix") or "",
                    number=str(row.get("number") or ""),
                    name_ru=row.get("name_ru") or "",
                    name_kz=row.get("name_kz") or "",
                    body_ru=row.get("body_ru") or "",
                    combined_score=score,
                    source="bm25",
                )
            else:
                # Articles matching multiple keywords get a stronger bonus (was 0.25)
                score_map[aid].combined_score += score * 0.5

    bm25_articles = sorted(score_map.values(), key=lambda a: a.combined_score, reverse=True)

    seeds       = bm25_articles[:GRAPH_SEED]
    seed_ids    = [a.id for a in seeds]
    seed_scores = {a.id: a.combined_score for a in seeds}

    graph_rows = await graph_neighbours(seed_ids, limit=GRAPH_SEED * GRAPH_LIMIT)

    for row in graph_rows:
        aid = (row.get("id") or "").strip()
        if not aid or aid in score_map:
            continue
        parents: list[str] = row.get("parents") or []
        parent_vals = [seed_scores[p] for p in parents if p in seed_scores]
        if not parent_vals:
            continue
        graph_score = (sum(parent_vals) / len(parent_vals)) * GRAPH_DECAY
        score_map[aid] = ArticleScore(
            id=aid,
            codex_prefix=row.get("codex_prefix") or "",
            number=str(row.get("number") or ""),
            name_ru=row.get("name_ru") or "",
            name_kz=row.get("name_kz") or "",
            body_ru=row.get("body_ru") or "",
            combined_score=graph_score,
            source="graph",
        )

    all_articles = sorted(score_map.values(), key=lambda a: a.combined_score, reverse=True)

    def_kws = keywords[:DEFINITION_KW_COUNT]
    def_results: list[list[dict]] = await asyncio.gather(
        *[fetch_definitions(kw.text, top_k=DEFINITION_TOP_K) for kw in def_kws]
    )
    definitions: dict[str, list[dict]] = {
        kw.text: defs for kw, defs in zip(def_kws, def_results) if defs
    }

    context_parts: list[str] = []
    total_chars = 0
    included    = 0

    for art in all_articles:
        chunk = art.format_for_context()
        if not chunk:
            continue
        chunk_len = len(chunk) + 2
        if total_chars + chunk_len > CONTEXT_MAX_CHARS:
            break
        context_parts.append(chunk)
        total_chars += chunk_len
        included    += 1

    if definitions:
        def_lines = ["\n--- ПРАВОВЫЕ ОПРЕДЕЛЕНИЯ ---"]
        for kw_text, defs in definitions.items():
            for d in defs:
                ru_def = (d.get("definition_ru") or "").strip()
                if ru_def:
                    def_lines.append(f"• {kw_text}: {ru_def}")
        def_block = "\n".join(def_lines)
        if total_chars + len(def_block) <= CONTEXT_MAX_CHARS:
            context_parts.append(def_block)

    context_text = "\n\n".join(context_parts)

    stats = {
        "bm25_hits":           len(bm25_articles),
        "graph_hits":          len(graph_rows),
        "total_candidates":    len(score_map),
        "included_in_context": included,
        "definitions_found":   sum(len(v) for v in definitions.values()),
    }
    logger.info("Retrieval stats: %s", stats)

    return RetrievalResult(
        articles=all_articles[:included],
        definitions=definitions,
        context_text=context_text,
        stats=stats,
    )
