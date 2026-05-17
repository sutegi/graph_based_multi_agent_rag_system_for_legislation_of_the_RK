from __future__ import annotations
"""Pydantic models and shared state utilities for the multi-agent RAG pipeline."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from .config import MAX_CONTEXT_TOKENS, logger


class Node(BaseModel):
    """Legal graph node: Article / Clause / Section / Keyword."""

    id:              str
    type:            str             = Field(description="Article | Section | Keyword | Clause")
    content_ru:      str             = Field(default="")
    content_kz:      str             = Field(default="")
    metadata:        dict[str, Any]  = Field(default_factory=dict)
    relevance_score: float           = Field(default=0.0, ge=0.0, le=1.0)

    def token_estimate(self) -> int:
        """Function token_estimate."""
        return (len(self.content_ru) + len(self.content_kz)) // 4

    def as_context_block(self) -> str:
        """Function as_context_block."""
        body   = self.content_ru.strip() or self.content_kz.strip()
        codex  = self.metadata.get("codex_prefix", "")
        num    = self.metadata.get("number", "")
        header = f"[{self.type} | id={self.id} | {codex} ст.{num}]"
        return f"{header}\n{body}"


class RouterDecision(BaseModel):
    """Structured output from Router Agent."""

    reasoning:       str                        = Field(description="Explain whether the context covers the query fully.")
    decision:        Literal["STOP", "KEEP_LOOKING"] = Field(description="STOP if context is sufficient; KEEP_LOOKING if gaps remain.")
    missing_aspects: list[str]                  = Field(default_factory=list, description="Specific legal aspects still missing (only when KEEP_LOOKING).")


class FinalAnswer(BaseModel):
    """Structured output from Answering Agent."""

    answer_ru:        str        = Field(description="Full legal answer in Russian.")
    answer_kz:        str | None = Field(default=None, description="Answer in Kazakh (bilingual mode).")
    cited_articles:   list[str]  = Field(description="Article IDs or titles cited in the answer.")
    confidence_score: float      = Field(ge=0.0, le=1.0, description="Estimated completeness of the answer (0–1).")


class AgentState(TypedDict):
    """Shared state passed through all LangGraph nodes."""

    query:            str
    current_nodes:    list[Node]
    visited_node_ids: set[str]
    next_step:        str
    logs:             list[str]
    iteration_count:  int
    final_answer:     FinalAnswer | None


def _total_tokens(nodes: list[Node]) -> int:
    """Function _total_tokens."""
    return sum(n.token_estimate() for n in nodes)


def _append_log(state: AgentState, agent: str, msg: str) -> list[str]:
    """Append a log entry and return the updated list."""
    entry = f"[{agent}] {msg}"
    logger.info(entry)
    return [*state["logs"], entry]


def _build_context(nodes: list[Node], max_chars: int = 12_000) -> str:
    """Concatenate node context blocks respecting a char budget."""
    parts: list[str] = []
    budget = max_chars
    for node in nodes:
        block = node.as_context_block()
        if len(block) > budget:
            break
        parts.append(block)
        budget -= len(block)
    return "\n\n───\n\n".join(parts)


def _extract_legal_terms(query: str) -> list[str]:
    """Lightweight keyword extractor over a fixed legal vocabulary."""
    vocabulary = [
        "правонарушение", "ответственность", "договор", "право собственности",
        "наследование", "алименты", "налог", "штраф", "арест", "дефиниция",
        "понятие", "уголовная", "гражданская", "трудовой", "кодекс",
        "статья", "закон", "иск", "обязательство", "субъект",
    ]
    q = query.lower()
    return [t for t in vocabulary if t in q]
