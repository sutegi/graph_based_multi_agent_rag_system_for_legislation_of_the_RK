"""Conversation session — tracks turn history for multi-turn LLM context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

MAX_HISTORY: int = 6


class Turn(NamedTuple):
    question: str
    answer: str
    confidence: float


@dataclass
class Session:
    """Mutable conversation state for a single user session."""

    history: list[Turn] = field(default_factory=list)

    def add(self, question: str, answer: str, confidence: float) -> None:
        """Append a completed turn; trim oldest entries beyond MAX_HISTORY."""
        self.history.append(Turn(question, answer, confidence))
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    def format_for_llm(self) -> str:
        """Return compact history string for LLM prompt injection; empty string if no history."""
        if not self.history:
            return ""
        lines = ["История разговора:"]
        for i, turn in enumerate(self.history, 1):
            lines.append(f"В{i}: {turn.question}")
            short = turn.answer[:250] + ("…" if len(turn.answer) > 250 else "")
            lines.append(f"О{i}: {short}")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        """Return number of completed turns in the current session."""
        return len(self.history)

    def clear(self) -> None:
        """Erase all conversation history."""
        self.history.clear()
