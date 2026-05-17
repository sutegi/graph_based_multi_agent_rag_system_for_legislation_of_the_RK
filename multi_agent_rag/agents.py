from __future__ import annotations
"""All six LangGraph agent nodes: definition, search, supervisor, router, recursive retrieval, and answering."""

from .config import MAX_CONTEXT_TOKENS, MAX_ITERATIONS, TOP_K_SEARCH, TOP_K_RELATED, DEEPSEEK_MODEL, logger
from .models import (
    Node, RouterDecision, FinalAnswer, AgentState,
    _total_tokens, _append_log, _build_context, _extract_legal_terms,
)
from .database import _db, _llm, _schema_description, _JSON_OBJECT_FORMAT


async def definition_agent(state: AgentState) -> dict:
    """Pre-fetch definition articles for legal terms detected in the query."""
    terms = _extract_legal_terms(state["query"])
    if not terms:
        return {
            "logs": _append_log(state, "DefinitionAgent",
                                "No legal terms detected — skipping definition lookup.")
        }

    try:
        nodes = await _db.find_definitions(terms)
    except Exception as exc:
        logger.error("[DefinitionAgent] Neo4j error: %s", exc)
        nodes = []

    new_nodes = [n for n in nodes if n.id not in state["visited_node_ids"]]
    new_ids   = {n.id for n in new_nodes}

    return {
        "current_nodes":    [*state["current_nodes"], *new_nodes],
        "visited_node_ids": state["visited_node_ids"] | new_ids,
        "logs": _append_log(
            state,
            "DefinitionAgent",
            f"Terms detected: {terms}. "
            f"Added {len(new_nodes)} definition node(s).",
        ),
    }


async def initial_search_agent(state: AgentState) -> dict:
    """Hybrid BM25 + vector retrieval for the top-K most relevant articles."""
    try:
        results = await _db.hybrid_search(state["query"], top_k=TOP_K_SEARCH)
    except Exception as exc:
        logger.error("[InitialSearchAgent] Search error: %s", exc)
        results = []

    new_nodes = [n for n in results if n.id not in state["visited_node_ids"]]
    new_ids   = {n.id for n in new_nodes}
    all_nodes = [*state["current_nodes"], *new_nodes]

    return {
        "current_nodes":    all_nodes,
        "visited_node_ids": state["visited_node_ids"] | new_ids,
        "logs": _append_log(
            state,
            "InitialSearchAgent",
            f"Hybrid search → {len(new_nodes)} new node(s). "
            f"Context: ~{_total_tokens(all_nodes)} tokens.",
        ),
    }


async def supervisor_agent(state: AgentState) -> dict:
    """Dedup nodes by id, prune to MAX_CONTEXT_TOKENS, log summary."""
    nodes = state["current_nodes"]

    seen:    set[str]   = set()
    deduped: list[Node] = []
    for n in nodes:
        if n.id not in seen:
            seen.add(n.id)
            deduped.append(n)
    dropped_dedup = len(nodes) - len(deduped)

    pruned        = sorted(deduped, key=lambda n: n.relevance_score, reverse=True)
    dropped_prune = 0
    while _total_tokens(pruned) > MAX_CONTEXT_TOKENS and len(pruned) > 1:
        pruned.pop()
        dropped_prune += 1

    return {
        "current_nodes": pruned,
        "logs": _append_log(
            state, "SupervisorAgent",
            f"Dedup: -{dropped_dedup} | Prune: -{dropped_prune} | "
            f"Remaining: {len(pruned)} node(s) "
            f"(~{_total_tokens(pruned)}/{MAX_CONTEXT_TOKENS} tokens).",
        ),
    }


async def router_agent(state: AgentState) -> dict:
    """Ask DeepSeek whether context is sufficient; decide STOP or KEEP_LOOKING."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return {
            "next_step":       "stop",
            "iteration_count": state["iteration_count"] + 1,
            "logs": _append_log(
                state, "RouterAgent",
                f"Max iterations ({MAX_ITERATIONS}) reached — forcing STOP.",
            ),
        }

    context_preview = _build_context(state["current_nodes"], max_chars=3_000)
    schema_desc     = _schema_description(RouterDecision)

    system_prompt = (
        "You are a legal routing agent for a Kazakhstani legislation RAG system.\n"
        "Decide whether the provided context is sufficient to answer the user query fully.\n"
        "Rules:\n"
        "- Be strict: only choose STOP if ALL legal aspects of the query are covered.\n"
        "- Choose KEEP_LOOKING when key articles, definitions, or sanctions are missing.\n"
        "- List each missing aspect as a short phrase in 'missing_aspects'.\n\n"
        f"{schema_desc}\n"
        "The 'decision' field MUST be exactly the string \"STOP\" or \"KEEP_LOOKING\"."
    )
    user_prompt = (
        f"User query:\n{state['query']}\n\n"
        f"Retrieved context ({len(state['current_nodes'])} articles):\n"
        f"{context_preview}\n\n"
        "Is this context sufficient for a complete legal answer? "
        "Reply with the JSON object only."
    )

    decision = RouterDecision(
        reasoning="API unavailable — defaulting to STOP.",
        decision="STOP",
        missing_aspects=[],
    )

    try:
        response = await _llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=_JSON_OBJECT_FORMAT,
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or "{}"
        decision = RouterDecision.model_validate_json(raw)

    except Exception as exc:
        logger.error("[RouterAgent] DeepSeek error: %s — defaulting to STOP.", exc)

    next_step = "keep_looking" if decision.decision == "KEEP_LOOKING" else "stop"

    return {
        "next_step":       next_step,
        "iteration_count": state["iteration_count"] + 1,
        "logs": _append_log(
            state, "RouterAgent",
            f"Decision: {decision.decision} | "
            f"Reasoning: {decision.reasoning[:100]} | "
            f"Missing: {decision.missing_aspects}",
        ),
    }


async def recursive_retrieval_agent(state: AgentState) -> dict:
    """One-hop graph expansion via BASED_ON / CAUSED_BY / REFERENCES edges."""
    source_ids = [
        n.id for n in state["current_nodes"]
        if n.relevance_score >= 0.8
        or n.id not in state["visited_node_ids"]
    ]

    if not source_ids:
        return {
            "logs": _append_log(
                state, "RecursiveRetrievalAgent",
                "No eligible source nodes for expansion — skipping hop.",
            )
        }

    try:
        related = await _db.get_related_nodes(
            source_ids=source_ids,
            relation_types=["BASED_ON", "CAUSED_BY", "REFERENCES"],
            top_k=TOP_K_RELATED,
        )
    except Exception as exc:
        logger.error("[RecursiveRetrievalAgent] Neo4j error: %s", exc)
        related = []

    new_nodes = [n for n in related if n.id not in state["visited_node_ids"]]
    new_ids   = {n.id for n in new_nodes}
    all_nodes = [*state["current_nodes"], *new_nodes]

    return {
        "current_nodes":    all_nodes,
        "visited_node_ids": state["visited_node_ids"] | new_ids,
        "logs": _append_log(
            state,
            "RecursiveRetrievalAgent",
            f"Hop #{state['iteration_count']}: "
            f"expanded {len(source_ids)} source(s) → "
            f"{len(new_nodes)} new node(s) found. "
            f"Total visited: {len(state['visited_node_ids'] | new_ids)}.",
        ),
    }


async def answering_agent(state: AgentState) -> dict:
    """Generate a citation-backed legal answer from accumulated context."""
    nodes   = state["current_nodes"]
    context = _build_context(nodes)
    cited   = [n.id for n in nodes]

    schema_desc = _schema_description(FinalAnswer)
    system_prompt = (
        "Ты — профессиональный юридический ассистент по законодательству "
        "Республики Казахстан. Правила:\n"
        "1. Отвечай СТРОГО на основе предоставленных статей.\n"
        "2. Каждый тезис подкрепляй ссылкой на id статьи в квадратных скобках, например [a1b2c3d4].\n"
        "3. Если информации недостаточно — прямо укажи это в ответе.\n"
        "4. Структура ответа: краткий вывод → детальный анализ → ссылки.\n"
        "5. Поле 'answer_ru' — обязательно на русском языке.\n"
        "6. Поле 'answer_kz' — перевод ответа на казахский язык (если возможно, иначе null).\n"
        "7. Поле 'cited_articles' — список id всех статей, на которые ты ссылаешься.\n"
        "8. Поле 'confidence_score' — от 0.0 до 1.0, насколько контекст покрывает вопрос.\n\n"
        f"{schema_desc}"
    )
    user_prompt = (
        f"Вопрос:\n{state['query']}\n\n"
        f"Статьи законодательства ({len(nodes)} шт.):\n"
        f"{context}\n\n"
        "Ответь ТОЛЬКО JSON-объектом."
    )

    fallback_ru = (
        f"На основе {len(nodes)} извлечённых статей:\n\n"
        + "\n".join(
            f"• [{n.id}] {n.content_ru[:150].strip()}…"
            for n in nodes[:5]
        )
        + "\n\nДля точного ответа требуется подключение к DeepSeek API."
    )
    answer = FinalAnswer(
        answer_ru=fallback_ru,
        answer_kz=None,
        cited_articles=cited,
        confidence_score=0.4,
    )

    try:
        response = await _llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=_JSON_OBJECT_FORMAT,
            temperature=0.2,
            max_tokens=4_096,
        )
        raw = response.choices[0].message.content or "{}"
        answer = FinalAnswer.model_validate_json(raw)

    except Exception as exc:
        logger.error("[AnsweringAgent] DeepSeek error: %s — using fallback.", exc)

    return {
        "final_answer": answer,
        "logs": _append_log(
            state, "AnsweringAgent",
            f"Answer generated. "
            f"Confidence: {answer.confidence_score:.2f} | "
            f"Citations: {len(answer.cited_articles)}.",
        ),
    }
