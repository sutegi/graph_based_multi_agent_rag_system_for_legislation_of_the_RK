"""
Three-phase RAG pipeline  —  no LangGraph dependency.

  Phase 1 — UNDERSTAND  : LLM intent analysis  →  IntentResult
  Phase 2 — RETRIEVE    : Weighted BM25 + graph enrichment  →  RetrievalResult
  Phase 3 — ANSWER      : LLM answer synthesis  →  AnswerResult

  Optional retry:
    If confidence < CONFIDENCE_THRESHOLD a second attempt is made with
    the codex filter removed (broader corpus).  The attempt with the
    higher confidence score is returned.

Public API:
    result = await run(question, session)   →  PipelineResult
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import CONFIDENCE_THRESHOLD, MAX_RETRIES, logger
from .llm import AnswerResult, IntentResult, analyse_intent, synthesise_answer
from .retriever import RetrievalResult, retrieve
from .session import Session


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    question: str
    intent: IntentResult
    retrieval: RetrievalResult
    answer: AnswerResult
    retried: bool = False
    elapsed_ms: int = 0

    @property
    def is_low_confidence(self) -> bool:
        return self.answer.confidence < CONFIDENCE_THRESHOLD


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run(question: str, session: Session) -> PipelineResult:
    """
    Execute the full pipeline for *question* given conversation *session*.

    Always returns a PipelineResult.  Never raises.
    """
    t0          = time.monotonic()
    history_ctx = session.format_for_llm()

    # ── Phase 1: UNDERSTAND ───────────────────────────────────────────────────
    logger.info("Pipeline ▶ phase 1 — intent analysis")
    intent = await analyse_intent(question, history_ctx=history_ctx)

    # Short-circuit: clearly not a legal question
    if not intent.is_legal_question:
        off_topic = AnswerResult(
            answer=(
                "Ваш вопрос не относится к правовой сфере Республики Казахстан. "
                "Пожалуйста, задайте юридический вопрос о законодательстве РК."
            ),
            citations=[],
            confidence=1.0,
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        return PipelineResult(
            question=question,
            intent=intent,
            retrieval=RetrievalResult(),
            answer=off_topic,
            elapsed_ms=elapsed,
        )

    # ── Phase 2: RETRIEVE ─────────────────────────────────────────────────────
    logger.info(
        "Pipeline ▶ phase 2 — retrieval  codexes=%s  kws=%d",
        intent.codex_slugs, len(intent.keywords),
    )
    retrieval = await retrieve(intent)

    # ── Phase 3: ANSWER ───────────────────────────────────────────────────────
    logger.info("Pipeline ▶ phase 3 — answer synthesis")
    answer = await synthesise_answer(
        question,
        context_text=retrieval.context_text,
        history_ctx=history_ctx,
    )

    retried = False

    # ── Optional retry with broader search ────────────────────────────────────
    if answer.confidence < CONFIDENCE_THRESHOLD and MAX_RETRIES > 0:
        logger.info(
            "Pipeline ↩ retry — conf %.2f < threshold %.2f — dropping codex filter",
            answer.confidence, CONFIDENCE_THRESHOLD,
        )
        retried = True

        # Drop codex filter → search across the entire corpus
        broad_intent = IntentResult(
            language=intent.language,
            is_legal_question=True,
            codex_slugs=[],
            keywords=intent.keywords,
        )
        broad_retrieval = await retrieve(broad_intent)
        broad_answer    = await synthesise_answer(
            question,
            context_text=broad_retrieval.context_text,
            history_ctx=history_ctx,
        )

        # Keep whichever attempt produced higher confidence
        if broad_answer.confidence >= answer.confidence:
            retrieval = broad_retrieval
            answer    = broad_answer
            logger.info(
                "Pipeline ↩ retry improved: %.2f → %.2f",
                answer.confidence, broad_answer.confidence,
            )

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Pipeline ✓  conf=%.2f  retried=%s  elapsed=%dms",
        answer.confidence, retried, elapsed,
    )

    # Persist this turn to session history
    session.add(question, answer.answer, answer.confidence)

    return PipelineResult(
        question=question,
        intent=intent,
        retrieval=retrieval,
        answer=answer,
        retried=retried,
        elapsed_ms=elapsed,
    )
