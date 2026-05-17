from __future__ import annotations
"""Neo4j async connector and DeepSeek LLM client for the RAG pipeline."""

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    TOP_K_SEARCH, TOP_K_RELATED,
    DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY,
    logger,
)
from .models import Node


class Neo4jConnector:
    """Async wrapper around the Neo4j driver."""

    def __init__(self) -> None:
        """Function __init__."""
        from neo4j import AsyncGraphDatabase
        self._driver   = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self._database = NEO4J_DATABASE
        logger.info("Neo4jConnector initialized: %s / db=%s", NEO4J_URI, NEO4J_DATABASE)

    async def close(self) -> None:
        """Function close."""
        if self._driver:
            await self._driver.close()

    async def ensure_indexes(self) -> None:
        """Create required Neo4j indexes on first run."""
        index_queries = [
            (
                "CREATE FULLTEXT INDEX article_fulltext_ru IF NOT EXISTS "
                "FOR (n:Article) ON EACH [n.body_ru, n.name_ru]"
            ),
            (
                "CREATE FULLTEXT INDEX article_fulltext_kz IF NOT EXISTS "
                "FOR (n:Article) ON EACH [n.body_kz, n.name_kz]"
            ),
            (
                "CREATE INDEX keyword_text_ru IF NOT EXISTS "
                "FOR (n:Keyword) ON (n.text_ru)"
            ),
        ]
        async with self._driver.session(database=self._database) as session:
            for q in index_queries:
                await session.run(q)
        logger.info("Neo4j indexes ensured.")

    async def _run(self, cypher: str, **params) -> list[dict]:
        """Execute a Cypher query and return all rows as dicts."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, parameters=params)
            return await result.data()

    @staticmethod
    def _row_to_node(row: dict, relevance_score: float = 0.8) -> Node:
        """Convert a Neo4j result row into a Node."""
        return Node(
            id=row["id"],
            type="Article",
            content_ru=row.get("body_ru") or "",
            content_kz=row.get("body_kz") or "",
            metadata={
                "codex_prefix": row.get("codex") or "",
                "number":        row.get("number") or "",
                "name_ru":       row.get("name_ru") or "",
                "relation":      row.get("rel_type") or "",
                "evidence":      row.get("evidence") or "",
            },
            relevance_score=relevance_score,
        )

    async def find_definitions(self, terms: list[str]) -> list[Node]:
        """Find articles connected to Keyword nodes matching the given terms."""
        rows = await self._run(
            """MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE k.text_ru IN $terms
            WITH a, count(k) AS match_count
            RETURN a.id            AS id,
                   a.name_ru       AS name_ru,
                   a.body_ru       AS body_ru,
                   a.body_kz       AS body_kz,
                   a.codex_prefix  AS codex,
                   a.number        AS number,
                   toFloat(match_count) AS score
            ORDER BY score DESC
            LIMIT 5.""",
            terms=terms,
        )
        logger.debug("[Neo4j] find_definitions: %d terms → %d rows", len(terms), len(rows))
        return [
            self._row_to_node(row, relevance_score=min(row.get("score", 1.0) / 5.0, 1.0))
            for row in rows
        ]

    async def hybrid_search(self, query: str, top_k: int = TOP_K_SEARCH) -> list[Node]:
        """BM25 full-text search using Neo4j's built-in Lucene index."""
        rows = await self._run(
            """CALL db.index.fulltext.queryNodes('article_fulltext_ru', $query)
            YIELD node, score
            RETURN node.id           AS id,
                   node.name_ru      AS name_ru,
                   node.body_ru      AS body_ru,
                   node.body_kz      AS body_kz,
                   node.codex_prefix AS codex,
                   node.number       AS number,
                   score
            ORDER BY score DESC
            LIMIT $top_k.""",
            query=query,
            top_k=top_k,
        )
        logger.debug("[Neo4j] hybrid_search: %r → %d rows", query[:60], len(rows))
        max_score = rows[0]["score"] if rows else 1.0
        return [
            self._row_to_node(
                row,
                relevance_score=round(row["score"] / max(max_score, 1e-9), 3),
            )
            for row in rows
        ]

    async def get_related_nodes(
        self,
        source_ids: list[str],
        relation_types: list[str],
        top_k: int = TOP_K_RELATED,
    ) -> list[Node]:
        """One-hop graph traversal following outgoing edges from source articles."""
        rows = await self._run(
            """UNWIND $source_ids AS src_id
            MATCH (src:Article {id: src_id})-[r]->(tgt:Article)
            WHERE type(r) IN $relation_types
            WITH DISTINCT tgt,
                 type(r)    AS rel_type,
                 r.evidence AS evidence
            RETURN tgt.id            AS id,
                   tgt.name_ru       AS name_ru,
                   tgt.body_ru       AS body_ru,
                   tgt.body_kz       AS body_kz,
                   tgt.codex_prefix  AS codex,
                   tgt.number        AS number,
                   rel_type,
                   evidence
            ORDER BY tgt.codex_prefix, tgt.number
            LIMIT $top_k.""",
            source_ids=source_ids,
            relation_types=relation_types,
            top_k=top_k,
        )
        logger.debug(
            "[Neo4j] get_related_nodes: %d sources → %d related",
            len(source_ids), len(rows),
        )
        return [self._row_to_node(row, relevance_score=0.75) for row in rows]


def _make_deepseek_client() -> AsyncOpenAI:
    """Function _make_deepseek_client."""
    if not DEEPSEEK_API_KEY:
        logger.error(
            "DEEPSEEK_API_KEY is not set. "
            "Create .env in diploma_code/ with: DEEPSEEK_API_KEY=sk-..."
        )
    else:
        masked = DEEPSEEK_API_KEY[:6] + "..." + DEEPSEEK_API_KEY[-4:]
        logger.info("DeepSeek client → %s  key=%s", DEEPSEEK_BASE_URL, masked)
    return AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _schema_description(model: type[BaseModel]) -> str:
    """Generate a human-readable field list to embed in system prompts."""
    schema = model.model_json_schema()
    required_fields = set(schema.get("required", []))
    props = schema.get("properties", {})

    lines = ["Respond ONLY with a valid JSON object (no markdown, no extra text)."]
    lines.append("Required fields:")
    for name, info in props.items():
        if "$ref" in info:
            type_str = info["$ref"].split("/")[-1]
        else:
            raw_type = info.get("type") or (
                " | ".join(x.get("type", "any") for x in info.get("anyOf", [{}]))
            )
            type_str = raw_type
        req  = "REQUIRED" if name in required_fields else "optional"
        desc = info.get("description", "")
        lines.append(f"  - \"{name}\" ({type_str}, {req}): {desc}")

    for name, info in props.items():
        enum_vals = info.get("enum")
        if enum_vals:
            lines.append(f"  * \"{name}\" must be one of: {enum_vals}")

    return "\n".join(lines)


_JSON_OBJECT_FORMAT: dict = {"type": "json_object"}

_db:  Neo4jConnector = Neo4jConnector()
_llm: AsyncOpenAI    = _make_deepseek_client()
