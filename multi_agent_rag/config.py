"""
Configuration for the Kazakhstan Legal RAG system.

All tunables, codex definitions, and shared prompts live here.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────

_ENV = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV if _ENV.exists() else None)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("legal_rag")

# ── LLM (DeepSeek via OpenAI-compatible API) ──────────────────────────────────

DEEPSEEK_API_KEY: str  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL: str    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

LLM_TIMEOUT: int       = 60    # seconds per call
LLM_MAX_RETRIES: int   = 2     # retry on transient errors

# ── Neo4j ─────────────────────────────────────────────────────────────────────

NEO4J_URI:      str = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER:     str = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Retrieval ─────────────────────────────────────────────────────────────────

# Base BM25 result limit per keyword. Actual limit = round(BASE × (0.5 + weight)).
TOP_K_BASE: int          = 10
# Maximum codex slugs to filter by (avoids combinatorial explosion).
MAX_CODEX_FILTERS: int   = 3
# How many top-scored articles to expand via graph traversal.
GRAPH_SEED: int          = 8
# Maximum graph traversal depth.
GRAPH_DEPTH: int         = 1
# Limit on graph neighbours per seed article.
GRAPH_LIMIT: int         = 5
# Score decay applied to graph-discovered articles.
GRAPH_DECAY: float       = 0.75
# Maximum top-weighted keywords used for definition lookup.
DEFINITION_KW_COUNT: int = 3
# Definition results per term.
DEFINITION_TOP_K: int    = 5
# Maximum characters of context passed to the answer LLM.
CONTEXT_MAX_CHARS: int   = 18_000

# ── Pipeline ──────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD: float = 0.55   # below this, one retry is attempted
MAX_RETRIES: int            = 1      # max additional attempts after first

# ── Kazakhstan legal codexes ──────────────────────────────────────────────────
# Keys are the codex_prefix values stored on Article nodes in Neo4j.
# Values are (Russian name, Kazakh name).

LEGAL_CODEXES: dict[str, tuple[str, str]] = {
    "admin_offenses":  ("Кодекс об административных правонарушениях",          "Әкімшілік құқық бұзушылықтар туралы кодекс"),
    "admin_proc":      ("Административный процедурно-процессуальный кодекс",    "Әкімшілік рәсімдік-процестік кодекс"),
    "budget":          ("Бюджетный кодекс",                                    "Бюджет кодексі"),
    "civil_general":   ("Гражданский кодекс (Общая часть)",                    "Азаматтық кодекс (Жалпы бөлім)"),
    "civil_proc":      ("Гражданский процессуальный кодекс",                   "Азаматтық процестік кодекс"),
    "civil_special":   ("Гражданский кодекс (Особенная часть)",                "Азаматтық кодекс (Ерекше бөлім)"),
    "construction":    ("Градостроительный кодекс",                            "Қала құрылысы кодексі"),
    "criminal":        ("Уголовный кодекс",                                    "Қылмыстық кодекс"),
    "criminal_exec":   ("Уголовно-исполнительный кодекс",                      "Қылмыстық-атқару кодексі"),
    "criminal_proc":   ("Уголовно-процессуальный кодекс",                      "Қылмыстық-процестік кодекс"),
    "customs":         ("Кодекс о таможенном регулировании",                   "Кедендік реттеу туралы кодекс"),
    "digital":         ("Цифровой кодекс",                                     "Цифрлық кодекс"),
    "entrepreneurial": ("Предпринимательский кодекс",                          "Кәсіпкерлік кодекс"),
    "environmental":   ("Экологический кодекс",                                "Экологиялық кодекс"),
    "family":          ("Кодекс о браке (супружестве) и семье",                "Неке (ерлі-зайыптылық) және отбасы туралы кодекс"),
    "forest":          ("Лесной кодекс",                                       "Орман кодексі"),
    "health":          ("Кодекс о здоровье народа и системе здравоохранения",  "Халық денсаулығы және денсаулық сақтау жүйесі туралы кодекс"),
    "labor":           ("Трудовой кодекс",                                     "Еңбек кодексі"),
    "land":            ("Земельный кодекс",                                    "Жер кодексі"),
    "social":          ("Социальный кодекс",                                   "Әлеуметтік кодекс"),
    "subsoil":         ("Кодекс о недрах и недропользовании",                  "Жер қойнауы және жер қойнауын пайдалану туралы кодекс"),
    "tax_code":        ("Налоговый кодекс",                                    "Салық кодексі"),
    "tax_payments":    ("Кодекс о налогах и других обязательных платежах в бюджет", "Салықтар және бюджетке төленетін басқа да міндетті төлемдер туралы кодекс"),
    "water":           ("Водный кодекс",                                       "Су кодексі"),
}

# Derived: Russian name → slug (for display)
CODEX_RU: dict[str, str] = {v[0]: k for k, v in LEGAL_CODEXES.items()}

# ── Codex name resolution ─────────────────────────────────────────────────────

def resolve_codex(name: str) -> list[str]:
    """Map a partial Russian or Kazakh codex name to validated slug(s).

    Case-insensitive substring match against both language names.
    Returns [] if no match.

    Examples:
        resolve_codex("Трудовой кодекс")   → ["labor"]
        resolve_codex("Гражданский")       → ["civil_general", "civil_special", "civil_proc"]
        resolve_codex("Уголовный")         → ["criminal", "criminal_proc", "criminal_exec"]
    """
    if not name:
        return []
    needle = name.strip().lower()
    return [
        slug
        for slug, (ru, kz) in LEGAL_CODEXES.items()
        if needle in ru.lower() or needle in kz.lower()
    ]


def validate_codex_slugs(slugs: list[str]) -> list[str]:
    """Return only slugs that exist in LEGAL_CODEXES (drop LLM hallucinations)."""
    return [s for s in (slugs or []) if s in LEGAL_CODEXES]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = (
    "Вы — точный правовой аналитик, специализирующийся исключительно на законодательстве "
    "Республики Казахстан.\n\n"
    "Обязательные правила:\n"
    "1. Ссылайтесь ТОЛЬКО на статьи, явно приведённые в контексте пользователя. "
    "Никогда не вспоминайте, не придумывайте и не перефразируйте статьи из обучающих данных.\n"
    "2. Отвечайте ТОЛЬКО валидным JSON, точно соответствующим схеме из запроса пользователя. "
    "Не используйте markdown-обёртки.\n"
    "3. Если контекст недостаточен, укажите это в JSON и установите confidence ниже 0.4.\n"
    "4. Отвечайте на языке запроса пользователя (русский или казахский).\n"
    "5. Не добавляйте пояснений, преамбул или текста вне структуры JSON.\n"
)
