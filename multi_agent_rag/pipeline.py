from __future__ import annotations
"""LangGraph StateGraph pipeline assembly, caching, and public query API."""

import argparse
import asyncio
from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph

from .config import MAX_ITERATIONS, DEEPSEEK_MODEL, logger
from .models import Node, FinalAnswer, AgentState, _build_context
from .database import _llm
from .agents import (
    definition_agent,
    initial_search_agent,
    supervisor_agent,
    router_agent,
    recursive_retrieval_agent,
    answering_agent,
)


def _route_decision(state: AgentState) -> Literal["stop", "keep_looking"]:
    """Conditional edge: return branch key after router_agent."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "stop"
    return state.get("next_step", "stop")                              


def build_graph():
    """Assemble and compile the full LangGraph StateGraph (with answering_agent)."""
    g = StateGraph(AgentState)

    g.add_node("definition_agent",          definition_agent)
    g.add_node("initial_search_agent",      initial_search_agent)
    g.add_node("supervisor_agent",          supervisor_agent)
    g.add_node("router_agent",              router_agent)
    g.add_node("recursive_retrieval_agent", recursive_retrieval_agent)
    g.add_node("answering_agent",           answering_agent)

    g.add_edge(START,                  "definition_agent")
    g.add_edge("definition_agent",     "initial_search_agent")
    g.add_edge("initial_search_agent", "supervisor_agent")
    g.add_edge("supervisor_agent",     "router_agent")

    g.add_conditional_edges(
        "router_agent",
        _route_decision,
        {
            "stop":         "answering_agent",
            "keep_looking": "recursive_retrieval_agent",
        },
    )

    g.add_edge("recursive_retrieval_agent", "supervisor_agent")
    g.add_edge("answering_agent", END)

    return g.compile()


_FULL_PIPELINE:    Any | None = None
_CONTEXT_PIPELINE: Any | None = None


def _get_full_pipeline() -> Any:
    """Return the cached full pipeline (definition → answer)."""
    global _FULL_PIPELINE
    if _FULL_PIPELINE is None:
        _FULL_PIPELINE = build_graph()
        logger.info("Full pipeline compiled and cached.")
    return _FULL_PIPELINE


def _build_context_graph():
    """Assemble the context-only graph (stops at router — no answering_agent)."""
    g = StateGraph(AgentState)

    g.add_node("definition_agent",          definition_agent)
    g.add_node("initial_search_agent",      initial_search_agent)
    g.add_node("supervisor_agent",          supervisor_agent)
    g.add_node("router_agent",              router_agent)
    g.add_node("recursive_retrieval_agent", recursive_retrieval_agent)

    g.add_edge(START,                  "definition_agent")
    g.add_edge("definition_agent",     "initial_search_agent")
    g.add_edge("initial_search_agent", "supervisor_agent")
    g.add_edge("supervisor_agent",     "router_agent")

    g.add_conditional_edges(
        "router_agent",
        _route_decision,
        {
            "stop":         END,
            "keep_looking": "recursive_retrieval_agent",
        },
    )
    g.add_edge("recursive_retrieval_agent", "supervisor_agent")
    return g.compile()


def _get_context_pipeline() -> Any:
    """Return the cached context-only pipeline (definition → router, no answering)."""
    global _CONTEXT_PIPELINE
    if _CONTEXT_PIPELINE is None:
        _CONTEXT_PIPELINE = _build_context_graph()
        logger.info("Context pipeline compiled and cached.")
    return _CONTEXT_PIPELINE


async def run_query(query: str, bilingual: bool = False) -> FinalAnswer | None:
    """Execute the full multi-agent pipeline for a legal query."""
    pipeline = _get_full_pipeline()

    initial_state: AgentState = {
        "query":            query + (" (ответ на двух языках)" if bilingual else ""),
        "current_nodes":    [],
        "visited_node_ids": set(),
        "next_step":        "keep_looking",
        "logs":             [],
        "iteration_count":  0,
        "final_answer":     None,
    }

    logger.info("=" * 65)
    logger.info("Pipeline start | Query: %r", query[:80])
    logger.info("=" * 65)

    try:
        final_state: AgentState = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline crashed: %s", exc, exc_info=True)
        return None

    logger.info("─" * 65)
    logger.info("Execution trace (%d steps):", len(final_state.get("logs", [])))
    for entry in final_state.get("logs", []):
        logger.info("  %s", entry)

    answer: FinalAnswer | None = final_state.get("final_answer")
    if answer:
        logger.info("─" * 65)
        logger.info("ANSWER (RU):\n%s", answer.answer_ru)
        if answer.answer_kz:
            logger.info("ANSWER (KZ):\n%s", answer.answer_kz)
        logger.info("CITED: %s", answer.cited_articles)
        logger.info("CONFIDENCE: %.2f", answer.confidence_score)
    else:
        logger.warning("No answer produced.")

    return answer


async def get_context_nodes(
    query: str,
    status_cb: "Callable[[str], None] | None" = None,
) -> tuple[list[Node], list[str]]:
    """Run retrieval agents without generating the final answer."""
    pipeline = _get_context_pipeline()

    initial_state: AgentState = {
        "query":            query,
        "current_nodes":    [],
        "visited_node_ids": set(),
        "next_step":        "keep_looking",
        "logs":             [],
        "iteration_count":  0,
        "final_answer":     None,
    }

    try:
        final_state: AgentState = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("get_context_nodes failed: %s", exc)
        return [], [f"[ERROR] {exc}"]

    logs = final_state.get("logs", [])
    if status_cb:
        for entry in logs:
            status_cb(entry)

    return final_state.get("current_nodes", []), logs


async def stream_answer(query: str, nodes: list[Node]):
    """Async generator yielding DeepSeek response tokens as they arrive."""
    if not nodes:
        yield "⚠ Релевантные статьи не найдены. Попробуйте переформулировать запрос."
        return

    context = _build_context(nodes)

    system_prompt = (
        "Ты — профессиональный юридический ассистент по законодательству "
        "Республики Казахстан. Правила:\n"
        "1. Отвечай СТРОГО на основе предоставленных статей.\n"
        "2. Каждый тезис подкрепляй ссылкой (id статьи в квадратных скобках).\n"
        "3. Структура: краткий вывод → детальный анализ → ссылки.\n"
        "4. Если информации недостаточно — явно укажи это.\n"
        "Отвечай на русском языке."
    )
    user_prompt = (
        f"Вопрос пользователя:\n{query}\n\n"
        f"Статьи законодательства ({len(nodes)} шт.):\n{context}"
    )

    try:
        stream = await _llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            stream=True,
            temperature=0.2,
            max_tokens=4_096,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as exc:
        logger.error("stream_answer error: %s", exc)
        yield f"\n\n⚠ Ошибка при генерации ответа: {exc}"


async def _main(query: str | None) -> None:
    """Function _main."""
    test_queries = query and [query] or [
        "Какова ответственность за нарушение договора по гражданскому кодексу?",
        "Каковы права работника при незаконном увольнении по трудовому кодексу?",
    ]
    for q in test_queries:
        await run_query(q)
        print()


def main() -> None:
    """Function main."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Graph-RAG pipeline for Kazakhstani legislation.",
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Legal question to answer. Omit to run built-in demo queries.",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.query))


if __name__ == "__main__":
    main()
