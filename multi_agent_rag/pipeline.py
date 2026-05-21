"""Three-phase RAG pipeline: intent analysis → retrieval → answer synthesis."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import CONFIDENCE_THRESHOLD, MAX_RETRIES, logger
from .llm import AnswerResult, IntentResult, analyse_intent, synthesise_answer
from .retriever import RetrievalResult, retrieve
from .session import Session


@dataclass
class PipelineResult:
    """Full output of one pipeline run including intent, retrieval, and answer."""

    question: str
    intent: IntentResult
    retrieval: RetrievalResult
    answer: AnswerResult
    retried: bool = False
    elapsed_ms: int = 0

    @property
    def is_low_confidence(self) -> bool:
        """Return True if answer confidence is below the configured threshold."""
        return self.answer.confidence < CONFIDENCE_THRESHOLD


async def run(question: str, session: Session) -> PipelineResult:
    """Execute the full pipeline for question given session. Always returns. Never raises."""
    t0          = time.monotonic()
    history_ctx = session.format_for_llm()

    logger.info("Pipeline ▶ phase 1 — intent analysis")
    intent = await analyse_intent(question, history_ctx=history_ctx)

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

    logger.info(
        "Pipeline ▶ phase 2 — retrieval  codexes=%s  kws=%d",
        intent.codex_slugs, len(intent.keywords),
    )
    retrieval = await retrieve(intent)

    logger.info("Pipeline ▶ phase 3 — answer synthesis")
    answer = await synthesise_answer(
        question,
        context_text=retrieval.context_text,
        history_ctx=history_ctx,
    )

    retried = False

    if answer.confidence < CONFIDENCE_THRESHOLD and MAX_RETRIES > 0:
        logger.info(
            "Pipeline ↩ retry — conf %.2f < threshold %.2f — dropping codex filter",
            answer.confidence, CONFIDENCE_THRESHOLD,
        )
        retried = True

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

    session.add(question, answer.answer, answer.confidence)

    return PipelineResult(
        question=question,
        intent=intent,
        retrieval=retrieval,
        answer=answer,
        retried=retried,
        elapsed_ms=elapsed,
    )
