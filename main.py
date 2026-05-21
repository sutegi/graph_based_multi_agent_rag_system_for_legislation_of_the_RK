"""Kazakhstan Legal AI Assistant — real-time Rich terminal chat interface."""
from __future__ import annotations

import asyncio
import sys

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from multi_agent_rag import Session, close_driver, run
from multi_agent_rag.config import LEGAL_CODEXES
from multi_agent_rag.database import ping
from multi_agent_rag.pipeline import PipelineResult

console = Console(highlight=False)

_BANNER = """\
[bold cyan]\
╔══════════════════════════════════════════════════════════╗
║   Казахстанский Правовой ИИ-Ассистент                    ║
║   Kazakhstan Legal AI Assistant                          ║
╚══════════════════════════════════════════════════════════╝\
[/bold cyan]

[dim]Задавайте вопросы о законодательстве РК на русском или казахском языке.
Команды: /help  /clear  /stats  /quit[/dim]
"""

_HELP = """\
[bold white]Команды:[/bold white]
  [yellow]/help[/yellow]   — эта справка
  [yellow]/clear[/yellow]  — очистить историю разговора
  [yellow]/stats[/yellow]  — подробная статистика последнего поиска
  [yellow]/quit[/yellow]   — выйти

[bold white]О системе:[/bold white]
  База данных: {:d} кодексов · ~9 587 статей · законодательство РК
  Поиск: BM25 fulltext + one-hop graph enrichment + DeepSeek LLM
  Многоходовой диалог: история последних {:d} вопросов передаётся в LLM
""".format(len(LEGAL_CODEXES), 6)

_last_result: PipelineResult | None = None


def _conf_colour(conf: float) -> str:
    """Return Rich colour name corresponding to the confidence level."""
    if conf >= 0.75:
        return "bright_green"
    if conf >= 0.45:
        return "yellow"
    return "red"


def _answer_panel(result: PipelineResult) -> Panel:
    """Build the main answer Rich Panel with confidence and timing metadata."""
    conf   = result.answer.confidence
    colour = _conf_colour(conf)
    s      = result.retrieval.stats

    meta: list[str] = [f"[{colour}]уверенность {conf:.0%}[/{colour}]"]
    if result.retried:
        meta.append("[yellow]↩ расширенный поиск[/yellow]")
    meta.append(f"[dim]{result.elapsed_ms} мс[/dim]")
    if s:
        meta.append(
            f"[dim]статей: {s.get('included_in_context', 0)}"
            f"/{s.get('total_candidates', 0)}[/dim]"
        )

    return Panel(
        Text(result.answer.answer),
        title="[bold white]Ответ[/bold white]",
        subtitle="  ·  ".join(meta),
        border_style="blue",
        padding=(1, 2),
    )


def _citations_table(result: PipelineResult) -> Table | None:
    """Return a Rich Table of cited articles, or None if no citations."""
    cits = result.answer.citations
    if not cits:
        return None

    tbl = Table(
        title="Использованные статьи",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )
    tbl.add_column("Кодекс",   style="cyan",       no_wrap=True, max_width=42)
    tbl.add_column("Статья",   style="bold white",  no_wrap=True)
    tbl.add_column("Название", style="white",       max_width=55)

    seen: set[str] = set()
    for c in cits:
        key = f"{c.codex_prefix}:{c.number}"
        if key in seen:
            continue
        seen.add(key)
        codex_name = (LEGAL_CODEXES.get(c.codex_prefix) or ("",))[0] or c.codex_prefix
        tbl.add_row(codex_name[:42], f"Ст. {c.number}", (c.name_ru or "")[:55])

    return tbl if seen else None


def _stats_table(result: PipelineResult) -> Table:
    """Return a Rich Table with detailed retrieval statistics for the last query."""
    tbl = Table(
        title="Статистика последнего запроса",
        box=box.SIMPLE_HEAD,
        show_header=False,
        border_style="dim",
        padding=(0, 1),
    )
    tbl.add_column("Параметр", style="dim cyan", no_wrap=True)
    tbl.add_column("Значение", style="white")

    s      = result.retrieval.stats
    intent = result.intent

    tbl.add_row("Язык вопроса",    intent.language)
    tbl.add_row(
        "Кодексы",
        ", ".join(intent.codex_slugs) if intent.codex_slugs else "(авто — без фильтра)",
    )
    tbl.add_row(
        "Ключевые слова",
        "  ".join(f"{k.text}({k.weight:.1f})" for k in intent.keywords[:6]),
    )
    tbl.add_row("BM25 хитов",       str(s.get("bm25_hits",           "—")))
    tbl.add_row("Граф-соседей",     str(s.get("graph_hits",           "—")))
    tbl.add_row("Всего кандидатов", str(s.get("total_candidates",     "—")))
    tbl.add_row("В контексте LLM",  str(s.get("included_in_context",  "—")))
    tbl.add_row("Определений",      str(s.get("definitions_found",    "—")))
    tbl.add_row("Уверенность",      f"{result.answer.confidence:.0%}")
    tbl.add_row("Расш. поиск",      "да" if result.retried else "нет")
    tbl.add_row("Время",            f"{result.elapsed_ms} мс")

    return tbl


def _intent_hint(result: PipelineResult) -> str:
    """Return a one-line summary of detected codexes and top keywords."""
    parts: list[str] = []
    if result.intent.codex_slugs:
        names = [
            (LEGAL_CODEXES.get(s) or (s,))[0]
            for s in result.intent.codex_slugs[:2]
        ]
        parts.append("Кодекс: " + ", ".join(names))
    if result.intent.keywords:
        kw_str = ", ".join(k.text for k in result.intent.keywords[:4])
        parts.append(f"Термины: {kw_str}")
    return "  |  ".join(parts)


async def _process(question: str, session: Session) -> None:
    """Run the RAG pipeline behind a Rich spinner and render the result to console."""
    global _last_result

    with Live(
        Spinner("dots", text="[cyan]Анализирую запрос…[/cyan]"),
        console=console,
        refresh_per_second=12,
        transient=True,
    ):
        result = await run(question, session)

    _last_result = result

    hint = _intent_hint(result)
    if hint:
        console.print(f"  [dim]↳ {hint}[/dim]")

    console.print(_answer_panel(result))

    cit_tbl = _citations_table(result)
    if cit_tbl:
        console.print(cit_tbl)

    console.print()


async def main() -> None:
    """Entry point — verify DB connection, then run the interactive chat loop."""
    console.print(_BANNER)

    console.print("[dim]Подключение к Neo4j…[/dim]", end=" ")
    if await ping():
        console.print("[bright_green]✓ подключено[/bright_green]\n")
    else:
        console.print("[red]✗ недоступно[/red]")
        console.print(
            "[yellow]Убедитесь, что Neo4j запущен и .env настроен верно.[/yellow]"
        )
        sys.exit(1)

    session = Session()

    while True:
        try:
            raw = console.input("[bold cyan]Вы:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]До свидания![/dim]")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            cmd = raw.lower().split()[0]

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]До свидания![/dim]")
                break

            elif cmd == "/clear":
                session.clear()
                console.print(Rule(style="dim"))
                console.print("[bright_green]История разговора очищена.[/bright_green]\n")

            elif cmd == "/help":
                console.print(_HELP)

            elif cmd == "/stats":
                if _last_result is not None:
                    console.print(_stats_table(_last_result))
                    console.print()
                else:
                    console.print("[dim]Нет данных — сначала задайте вопрос.[/dim]\n")

            else:
                console.print(
                    f"[red]Неизвестная команда:[/red] {raw}  [dim](введите /help)[/dim]\n"
                )
            continue

        try:
            await _process(raw, session)
        except KeyboardInterrupt:
            console.print("\n[dim]Запрос прерван.[/dim]\n")
        except Exception as exc:
            console.print(f"[red]Ошибка:[/red] {exc}\n")

    await close_driver()


if __name__ == "__main__":
    asyncio.run(main())
