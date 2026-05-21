"""Async Neo4j client — all raw Cypher queries live here."""
from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from .config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, logger

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    """Return the shared async driver, creating it on first call."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_pool_size=20,
        )
        logger.info("Neo4j driver initialised: %s", NEO4J_URI)
    return _driver


async def close_driver() -> None:
    """Gracefully close the shared driver."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("Neo4j driver closed.")


async def ping() -> bool:
    """Return True if Neo4j is reachable."""
    try:
        driver = await get_driver()
        async with driver.session(database=NEO4J_DATABASE) as s:
            await s.run("RETURN 1")
        return True
    except Exception as exc:
        logger.warning("Neo4j ping failed: %s", exc)
        return False


async def bm25_search(
    query: str,
    limit: int = 10,
    codex_slugs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """BM25 fulltext search over Articles; filters by codex at DB level when slugs are given."""
    driver = await get_driver()

    if codex_slugs:
        cypher = (
            "CALL db.index.fulltext.queryNodes('article_fulltext_ru', $query) "
            "YIELD node, score "
            "WHERE node.codex_prefix IN $slugs "
            "RETURN node.id            AS id, "
            "       node.codex_prefix  AS codex_prefix, "
            "       node.number        AS number, "
            "       node.name_ru       AS name_ru, "
            "       node.name_kz       AS name_kz, "
            "       node.body_ru       AS body_ru, "
            "       score "
            "ORDER BY score DESC "
            "LIMIT  $limit"
        )
        params: dict[str, Any] = {"query": query, "slugs": codex_slugs, "limit": limit}
    else:
        cypher = (
            "CALL db.index.fulltext.queryNodes('article_fulltext_ru', $query) "
            "YIELD node, score "
            "RETURN node.id            AS id, "
            "       node.codex_prefix  AS codex_prefix, "
            "       node.number        AS number, "
            "       node.name_ru       AS name_ru, "
            "       node.name_kz       AS name_kz, "
            "       node.body_ru       AS body_ru, "
            "       score "
            "ORDER BY score DESC "
            "LIMIT  $limit"
        )
        params = {"query": query, "limit": limit}

    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(cypher, params)
            records = await result.data()
        logger.debug("BM25 '%s' (slugs=%s) → %d hits", query[:40], codex_slugs, len(records))
        return records
    except Exception as exc:
        logger.error("BM25 search failed for '%s': %s", query[:40], exc)
        return []


async def graph_neighbours(
    seed_ids: list[str],
    limit: int = 40,
) -> list[dict[str, Any]]:
    """One-hop undirected traversal from seed articles across all relation types."""
    if not seed_ids:
        return []

    driver = await get_driver()
    cypher = (
        "MATCH (a:Article) "
        "WHERE a.id IN $seed_ids "
        "MATCH (a)-[:CORRELATED_WITH|REFERENCES|EXTENDS|MANIFESTED_IN|BASED_ON|CAUSED_BY]"
        "-(nb:Article) "
        "WHERE NOT nb.id IN $seed_ids "
        "WITH nb, "
        "     collect(DISTINCT a.id) AS parents, "
        "     count(*)               AS edge_count "
        "RETURN nb.id            AS id, "
        "       nb.codex_prefix  AS codex_prefix, "
        "       nb.number        AS number, "
        "       nb.name_ru       AS name_ru, "
        "       nb.name_kz       AS name_kz, "
        "       nb.body_ru       AS body_ru, "
        "       parents, "
        "       edge_count "
        "ORDER BY edge_count DESC "
        "LIMIT  $limit"
    )
    params: dict[str, Any] = {"seed_ids": seed_ids, "limit": limit}

    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(cypher, params)
            records = await result.data()
        logger.debug("Graph neighbours: %d seeds → %d results", len(seed_ids), len(records))
        return records
    except Exception as exc:
        logger.error("Graph traversal failed: %s", exc)
        return []


async def fetch_definitions(
    keyword: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Look up legal-term definitions via Keyword → HAS_DEFINITION → Definition."""
    driver = await get_driver()
    cypher = (
        "MATCH (k:Keyword) "
        "WHERE toLower(k.text_ru) CONTAINS toLower($kw) "
        "   OR toLower(coalesce(k.text_kz, '')) CONTAINS toLower($kw) "
        "WITH k LIMIT 20 "
        "OPTIONAL MATCH (k)-[:HAS_DEFINITION]->(d:Definition) "
        "RETURN DISTINCT "
        "  coalesce(d.keyword_ru, k.text_ru, '')  AS keyword, "
        "  coalesce(d.keyword_kz, k.text_kz, '')  AS keyword_kz, "
        "  coalesce(d.definition_ru, '')           AS definition_ru, "
        "  coalesce(d.definition_kz, '')           AS definition_kz "
        "LIMIT $top_k"
    )
    params: dict[str, Any] = {"kw": keyword, "top_k": top_k}

    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(cypher, params)
            records = await result.data()
        return records
    except Exception as exc:
        logger.error("Definition lookup failed for '%s': %s", keyword, exc)
        return []
