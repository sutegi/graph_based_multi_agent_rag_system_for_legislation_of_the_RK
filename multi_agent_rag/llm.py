"""
LLM layer — structured JSON in, typed dataclasses out.

All OpenAI-compatible API calls live here.  No business logic.

Two public coroutines:
    analyse_intent(question, history_ctx)  →  IntentResult
    synthesise_answer(question, context, history_ctx)  →  AnswerResult
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LEGAL_CODEXES,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    SYSTEM_PROMPT,
    logger,
    resolve_codex,
    validate_codex_slugs,
)

# ── Client singleton ──────────────────────────────────────────────────────────

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )
    return _client


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Keyword:
    """One search keyword with an LLM-assigned importance weight."""

    text: str
    weight: float  # 0.1 – 1.0, descending in IntentResult.keywords

    def __repr__(self) -> str:
        return f"Keyword({self.text!r}, w={self.weight:.2f})"


@dataclass
class IntentResult:
    """Output of intent analysis: language, codex slugs, ranked keywords."""

    language: str            # "ru" | "kz" | "mixed"
    is_legal_question: bool
    codex_slugs: list[str]   # validated slugs from LEGAL_CODEXES, may be empty
    keywords: list[Keyword]  # sorted by weight descending (highest first)
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class Citation:
    """Reference to one article used in the synthesised answer."""

    article_id: str
    codex_prefix: str
    number: str
    name_ru: str


@dataclass
class AnswerResult:
    """Output of answer synthesis: text, citations, confidence."""

    answer: str
    citations: list[Citation]
    confidence: float  # 0.0 – 1.0
    raw: dict = field(default_factory=dict, repr=False)


# ── Shared LLM call ───────────────────────────────────────────────────────────

async def _chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """
    Single DeepSeek call in JSON mode.

    Strips markdown code-fences if the model wraps the JSON.
    Raises on network/API error or JSON parse failure.
    """
    client = get_client()
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        ),
        timeout=LLM_TIMEOUT,
    )
    raw: str = (resp.choices[0].message.content or "{}").strip()

    # Strip possible ``` fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        # drop first line (```json or ```) and last ``` line
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw = "\n".join(inner).strip()

    return json.loads(raw)


# ── Intent analysis ───────────────────────────────────────────────────────────

_VALID_SLUGS = ", ".join(sorted(LEGAL_CODEXES.keys()))

_INTENT_PROMPT = """\
Проанализируй юридический вопрос о законодательстве Республики Казахстан.

Допустимые значения codex_slugs: {codex_list}

{history}

Вопрос: {question}

Верни JSON строго по этой схеме (без markdown):
{{
  "language": "ru",
  "is_legal_question": true,
  "codex_slugs": ["slug1"],
  "keywords": [
    {{"text": "юридический термин", "weight": 0.95}},
    {{"text": "второй термин",      "weight": 0.75}}
  ]
}}

Правила:
- language       : "ru" | "kz" | "mixed"  (язык вопроса)
- is_legal_question : true если вопрос касается права / законодательства РК
- codex_slugs    : только slug-и из списка выше, которые ПРЯМО регулируют вопрос; [] если неясно
- keywords       : 4–10 юридических терминов на РУССКОМ, sorted by weight DESC
- weight         : 1.0 = самый важный; 0.1 = наименее важный
- Предпочитай словосочетания (2–3 слова) одиночным общим словам
- Включай конкретные правовые термины И ситуационные слова из вопроса
"""


async def analyse_intent(
    question: str,
    history_ctx: str = "",
) -> IntentResult:
    """
    Classify the question: language, legal domain, codex slugs, ranked keywords.

    Never raises — on any failure returns a safe minimal IntentResult so
    retrieval can still proceed.
    """
    user_msg = _INTENT_PROMPT.format(
        codex_list=_VALID_SLUGS,
        history=history_ctx,
        question=question,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    try:
        data = await _chat_json(messages)
    except Exception as exc:
        logger.error("Intent analysis failed: %s", exc)
        return IntentResult(
            language="ru",
            is_legal_question=True,
            codex_slugs=[],
            keywords=[Keyword(text=question[:80].strip(), weight=1.0)],
        )

    # ── language ─────────────────────────────────────────────────────────────
    language = str(data.get("language") or "ru").lower().strip()
    if language not in ("ru", "kz", "mixed"):
        language = "ru"

    # ── is_legal_question ─────────────────────────────────────────────────────
    is_legal = bool(data.get("is_legal_question", True))

    # ── codex_slugs ───────────────────────────────────────────────────────────
    raw_slugs: list = data.get("codex_slugs") or []
    validated: list[str] = []
    for item in raw_slugs:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item in LEGAL_CODEXES:
            validated.append(item)
        else:
            # LLM may have returned a Russian codex name instead of a slug
            validated.extend(resolve_codex(item))
    # deduplicate while preserving order, then validate
    seen_set: set[str] = set()
    deduped: list[str] = []
    for s in validated:
        if s not in seen_set:
            seen_set.add(s)
            deduped.append(s)
    validated = validate_codex_slugs(deduped)

    # ── keywords ──────────────────────────────────────────────────────────────
    raw_kws: list = data.get("keywords") or []
    keywords: list[Keyword] = []
    seen_kw: set[str] = set()

    for item in raw_kws:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text or text.lower() in seen_kw:
            continue
        seen_kw.add(text.lower())

        try:
            weight = float(item.get("weight") or 0.5)
        except (TypeError, ValueError):
            weight = 0.5
        weight = max(0.1, min(1.0, weight))
        keywords.append(Keyword(text=text, weight=weight))

    # Ensure descending order
    keywords.sort(key=lambda k: k.weight, reverse=True)

    # Fallback: use trimmed question text
    if not keywords:
        keywords = [Keyword(text=question[:80].strip(), weight=1.0)]

    logger.info(
        "Intent: lang=%s  legal=%s  slugs=%s  kws=%d  top=%s",
        language, is_legal, validated, len(keywords),
        [(k.text, round(k.weight, 2)) for k in keywords[:3]],
    )

    return IntentResult(
        language=language,
        is_legal_question=is_legal,
        codex_slugs=validated,
        keywords=keywords,
        raw=data,
    )


# ── Answer synthesis ──────────────────────────────────────────────────────────

_ANSWER_PROMPT = """\
Ответь на юридический вопрос, используя ТОЛЬКО приведённый ниже контекст.

{history}

Вопрос: {question}

=== КОНТЕКСТ (статьи законодательства РК) ===
{context}
==============================================

Верни JSON строго по этой схеме (без markdown):
{{
  "answer": "развёрнутый ответ на языке вопроса",
  "citations": [
    {{
      "article_id":   "внутренний id",
      "codex_prefix": "slug кодекса",
      "number":       "номер статьи",
      "name_ru":      "название статьи"
    }}
  ],
  "confidence": 0.85
}}

Правила:
- answer     : полный ответ на ТОМ ЖЕ языке, что и вопрос (русский или казахский)
- citations  : только статьи из контекста, которые реально использованы в ответе
- confidence : 0.0–1.0
    · 0.85–0.95 — контекст полностью отвечает на вопрос с конкретными статьями
    · 0.60–0.80 — контекст частично отвечает
    · <0.40     — контекст недостаточен
- Если контекста недостаточно: начни answer со строки
  "НЕДОСТАТОЧНО ПРАВОВЫХ ОСНОВАНИЙ:" и объясни, чего не хватает
- НЕ придумывай статьи и НЕ используй обучающие данные
- Не используй markdown в поле answer
"""


async def synthesise_answer(
    question: str,
    context_text: str,
    history_ctx: str = "",
) -> AnswerResult:
    """
    Generate a cited, structured answer from retrieved context.

    Never raises — on any failure returns a low-confidence error answer.
    """
    if not context_text.strip():
        return AnswerResult(
            answer=(
                "НЕДОСТАТОЧНО ПРАВОВЫХ ОСНОВАНИЙ: Не удалось найти релевантные "
                "статьи в базе данных. Рекомендуется обратиться к "
                "квалифицированному юристу или изучить нормы на adilet.zan.kz."
            ),
            citations=[],
            confidence=0.1,
        )

    user_msg = _ANSWER_PROMPT.format(
        history=history_ctx,
        question=question,
        context=context_text,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    try:
        data = await _chat_json(messages, temperature=0.15)
    except Exception as exc:
        logger.error("Answer synthesis failed: %s", exc)
        return AnswerResult(
            answer="Техническая ошибка при формировании ответа. Попробуйте повторить запрос.",
            citations=[],
            confidence=0.0,
        )

    # ── answer text ───────────────────────────────────────────────────────────
    answer = str(data.get("answer") or "").strip()
    if not answer:
        answer = "Не удалось сформировать ответ на основе предоставленного контекста."

    # ── citations ─────────────────────────────────────────────────────────────
    raw_cits: list = data.get("citations") or []
    citations: list[Citation] = []
    seen_cit: set[str] = set()

    for c in raw_cits:
        if not isinstance(c, dict):
            continue
        key = f"{c.get('codex_prefix', '')}:{c.get('number', '')}"
        if key in seen_cit:
            continue
        seen_cit.add(key)
        citations.append(Citation(
            article_id=str(c.get("article_id") or ""),
            codex_prefix=str(c.get("codex_prefix") or ""),
            number=str(c.get("number") or ""),
            name_ru=str(c.get("name_ru") or ""),
        ))

    # ── confidence ────────────────────────────────────────────────────────────
    try:
        confidence = float(data.get("confidence") or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Structural cap: penalise answers with too few citations
    if not citations:
        confidence = min(confidence, 0.35)
    elif len(citations) == 1:
        confidence = min(confidence, 0.65)

    # Penalise "insufficient" answers regardless of LLM self-report
    if answer.startswith("НЕДОСТАТОЧНО"):
        confidence = min(confidence, 0.35)

    logger.info(
        "Answer: conf=%.2f  citations=%d  answer_len=%d",
        confidence, len(citations), len(answer),
    )

    return AnswerResult(
        answer=answer,
        citations=citations,
        confidence=confidence,
        raw=data,
    )
