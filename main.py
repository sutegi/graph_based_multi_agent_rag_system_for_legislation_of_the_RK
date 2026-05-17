                      
"""main.py  —  Real-time chat interface for the Kazakhstani Legal RAG system."""

from __future__ import annotations

import asyncio
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

                                                                               
try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

                                                                               
from multi_agent_rag import run_query, get_context_nodes, stream_answer, Node

                                                                             
        
                                                                             

ASSISTANT_NAME = "Юр. Ассистент"
USER_NAME      = "Вы"
MAX_HISTORY    = 20                            

COMMANDS = {
    "/exit":    "Завершить сеанс",
    "/clear":   "Очистить экран",
    "/history": "Показать историю вопросов",
    "/nodes":   "Показать статьи последнего ответа",
    "/help":    "Показать эту справку",
}

WELCOME = """
╔══════════════════════════════════════════════════════════════════╗
║      Юридический ассистент — Законодательство РК (RU + KZ)      ║
║   Введите вопрос на русском или казахском языке и нажмите Enter  ║
╚══════════════════════════════════════════════════════════════════╝
"""


                                                                             
               
                                                                             

class ChatEntry(NamedTuple):
    """Class ChatEntry."""
    timestamp: str
    query:     str
    answer:    str
    nodes:     list[Node]


class Session:
    """Class Session."""
    def __init__(self) -> None:
        """Function __init__."""
        self.history:    list[ChatEntry] = []
        self.last_nodes: list[Node]      = []
        self.console:    "Console | None" = Console(highlight=False) if HAS_RICH else None

    def add(self, query: str, answer: str, nodes: list[Node]) -> None:
        """Function add."""
        entry = ChatEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            query=query,
            answer=answer,
            nodes=nodes,
        )
        self.history.append(entry)
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self.last_nodes = nodes


                                                                             
            
                                                                             

def _print(session: Session, text: str = "", **kwargs) -> None:
    """Function _print."""
    if session.console:
        session.console.print(text, **kwargs)
    else:
        print(text)


def _rule(session: Session, title: str = "") -> None:
    """Function _rule."""
    if session.console:
        session.console.print(Rule(title, style="dim"))
    else:
        print("─" * 60)


def _print_welcome(session: Session) -> None:
    """Function _print_welcome."""
    if session.console:
        session.console.print(
            Panel(
                Text.assemble(
                    ("Юридический ассистент\n", "bold white"),
                    ("Законодательство Республики Казахстан (RU + KZ)\n\n", "cyan"),
                    ("Введите вопрос и нажмите Enter\n", "dim"),
                    ("Команды: ", "dim"),
                    ("/help  /history  /nodes  /clear  /exit", "dim yellow"),
                ),
                border_style="blue",
                padding=(1, 4),
            )
        )
    else:
        print(WELCOME)


def _print_help(session: Session) -> None:
    """Function _print_help."""
    if session.console:
        tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        tbl.add_column("cmd",  style="yellow")
        tbl.add_column("desc", style="dim")
        for cmd, desc in COMMANDS.items():
            tbl.add_row(cmd, desc)
        session.console.print(tbl)
    else:
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd:<12} {desc}")


def _print_history(session: Session) -> None:
    """Function _print_history."""
    if not session.history:
        _print(session, "[dim]История пуста.[/dim]" if session.console else "История пуста.")
        return
    if session.console:
        for i, entry in enumerate(session.history, 1):
            session.console.print(
                f"[dim]{i:2}. [{entry.timestamp}][/dim] [cyan]{entry.query}[/cyan]"
            )
    else:
        for i, entry in enumerate(session.history, 1):
            print(f"  {i:2}. [{entry.timestamp}] {entry.query}")


def _print_nodes(session: Session) -> None:
    """Function _print_nodes."""
    nodes = session.last_nodes
    if not nodes:
        _print(session, "[dim]Нет данных — задайте вопрос сначала.[/dim]"
               if session.console else "Нет данных.")
        return
    if session.console:
        tbl = Table(
            "№", "ID", "Кодекс", "Статья", "Релевантность",
            box=box.SIMPLE_HEAVY, show_lines=False,
        )
        for i, n in enumerate(nodes, 1):
            tbl.add_row(
                str(i),
                n.id[:12] + "…",
                n.metadata.get("codex_prefix", "—"),
                n.metadata.get("number", "—"),
                f"{n.relevance_score:.2f}",
            )
        session.console.print(tbl)
    else:
        for i, n in enumerate(nodes, 1):
            print(f"  {i}. [{n.id[:12]}] {n.metadata.get('codex_prefix','?')} "
                  f"ст.{n.metadata.get('number','?')}  score={n.relevance_score:.2f}")


def _get_input(session: Session) -> str:
    """Read user input with a styled prompt."""
    if session.console:
        return session.console.input("\n[bold cyan]Вы:[/bold cyan] ").strip()
    else:
        return input("\nВы: ").strip()


                                                                             
                    
                                                                             

async def handle_query(query: str, session: Session) -> None:
    """Two-phase execution:
      Phase 1 — Context collection with live spinner showing agent progress."""

                                                                           
    nodes: list[Node] = []
    logs:  list[str]  = []
    agent_status = {"text": "Инициализация..."}

    def _status_cb(log_entry: str) -> None:
                                                        
                                           
        """Function _status_cb."""
        if log_entry.startswith("["):
            end = log_entry.find("]")
            agent_status["text"] = log_entry[1:end] if end > 0 else log_entry

    if session.console:
                                                                 
        spinner_text = Text()
        with session.console.status(
            "[bold yellow]Поиск по законодательству…[/bold yellow]",
            spinner="dots",
        ):
            nodes, logs = await get_context_nodes(query, status_cb=_status_cb)
        _print(session, f"[dim]Найдено статей: {len(nodes)} | "
               f"Агентов отработало: {len(logs)}[/dim]")
    else:
        print("Поиск по законодательству…", flush=True)
        nodes, logs = await get_context_nodes(query)
        print(f"Найдено статей: {len(nodes)}")

                                                                           
    _rule(session)

    if session.console:
        session.console.print(f"[bold green]{ASSISTANT_NAME}:[/bold green]")
    else:
        print(f"\n{ASSISTANT_NAME}:")

    full_answer_parts: list[str] = []

                                                             
                                                                     
    async for token in stream_answer(query, nodes):
        sys.stdout.write(token)
        sys.stdout.flush()
        full_answer_parts.append(token)

    sys.stdout.write("\n\n")
    sys.stdout.flush()

    full_answer = "".join(full_answer_parts)

                                                                           
    if nodes and session.console:
        cited = " · ".join(
            f"[dim]{n.metadata.get('codex_prefix','?')} ст.{n.metadata.get('number','?')}[/dim]"
            for n in nodes[:6]
        )
        session.console.print(f"[dim]📎 Источники: {cited}"
                               + (" …" if len(nodes) > 6 else "") + "[/dim]")
    elif nodes:
        sources = ", ".join(
            f"{n.metadata.get('codex_prefix','?')} ст.{n.metadata.get('number','?')}"
            for n in nodes[:6]
        )
        print(f"Источники: {sources}")

    _rule(session)

                                                                           
    session.add(query, full_answer, nodes)


                                                                             
                    
                                                                             

async def dispatch_command(cmd: str, session: Session) -> bool:
    """Handle slash commands."""
    cmd = cmd.lower().strip()

    if cmd == "/exit":
        _print(session, "\n[dim]До свидания![/dim]" if session.console else "\nДо свидания!")
        return True

    if cmd == "/clear":
        if session.console:
            session.console.clear()
        else:
            print("\033[2J\033[H", end="")
        _print_welcome(session)
        return False

    if cmd == "/help":
        _print_help(session)
        return False

    if cmd == "/history":
        _print_history(session)
        return False

    if cmd == "/nodes":
        _print_nodes(session)
        return False

    _print(session,
           f"[red]Неизвестная команда:[/red] {cmd}  (введите [yellow]/help[/yellow])"
           if session.console else f"Неизвестная команда: {cmd}")
    return False


                                                                             
           
                                                                             

async def chat_loop(session: Session) -> None:
    """Function chat_loop."""
    _print_welcome(session)

    while True:
        try:
            raw = _get_input(session)
        except (KeyboardInterrupt, EOFError):
            _print(session, "\n[dim]Прерывание — выход.[/dim]"
                   if session.console else "\nВыход.")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            should_exit = await dispatch_command(raw, session)
            if should_exit:
                break
            continue

        try:
            await handle_query(raw, session)
        except KeyboardInterrupt:
            _print(session, "\n[dim]Запрос прерван.[/dim]"
                   if session.console else "\nЗапрос прерван.")
        except Exception as exc:
            _print(session,
                   f"[red]Ошибка:[/red] {exc}" if session.console else f"Ошибка: {exc}")


                                                                             
             
                                                                             

def main() -> None:
    """Function main."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Real-time legal RAG chat for Kazakhstani legislation.",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable Rich formatting (plain terminal output).",
    )
    args = parser.parse_args()

    session = Session()
    if args.no_color:
        session.console = None                            

    asyncio.run(chat_loop(session))


if __name__ == "__main__":
    main()
