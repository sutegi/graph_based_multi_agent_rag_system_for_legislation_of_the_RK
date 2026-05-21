"""
multi_agent_rag — Kazakhstan Legal RAG system.

Quick-start
-----------
    from multi_agent_rag import run, Session

    session = Session()
    result  = await run("Можно ли уволить беременную женщину?", session)

    print(result.answer.answer)
    print(f"Confidence: {result.answer.confidence:.0%}")
    for c in result.answer.citations:
        print(f"  • {c.codex_prefix} ст.{c.number} — {c.name_ru}")

Shutdown
--------
    await close_driver()   # call once before process exit
"""
from .database import close_driver
from .llm import AnswerResult, Citation, IntentResult, Keyword
from .pipeline import PipelineResult, run
from .retriever import ArticleScore, RetrievalResult
from .session import Session

__all__ = [
    # Main entry points
    "run",
    "Session",
    "close_driver",
    # Result types
    "PipelineResult",
    "IntentResult",
    "AnswerResult",
    "Citation",
    "Keyword",
    "ArticleScore",
    "RetrievalResult",
]
