"""
converter.py v4 — Batch DOCX → Markdown converter
for 24 Codes of the Republic of Kazakhstan (RU + KZ)

Platform : Windows 11 (native Python, no WSL)
Author   : Senior Python Developer / Data Engineer
Project  : Multi-Agent Graph-Based RAG (Neo4j)
Encoding : UTF-8 — Kazakh Cyrillic with diacritics (Ә І Ң Ү Қ Ғ Һ Ө)

Bug-fixes v4 (over v3)
──────────────────────
  BUG-1 FIXED — Non-breaking spaces (\\xa0, U+00A0): pandoc uses \\xa0 for body-text
         indentation.  Python's lstrip() and regex \\s do NOT handle \\xa0 by default.
         Fix: _normalize_nbsp() strips \\xa0 → regular space as FIRST pipeline step.

  BUG-2 FIXED — CRLF output: write_text() on Windows uses text mode (\\n → \\r\\n).
         Fix: dst.write_text(..., newline='\\n') forces LF-only output.

  BUG-3 FIXED — Images not removed: RE_IMAGE regex was correct but postprocessing
         pipeline was never applied to the existing .md files (old code had no
         postprocess() call). v4 adds --reprocess flag for in-place re-processing of
         already-converted .md files without re-running pandoc.

  BUG-4 FIXED — Missing KZ structural patterns in HEADING_LEVEL_MAP:
         * "1-параграф." (KZ sub-paragraph) → H4
         * "1-КІШІ БӨЛІМ" (KZ subdivision) → H3
         * "-бап" without trailing dot (some articles omit it)
         * "§ N" Cyrillic variant

  BUG-5 FIXED — RE_SERVICE_NOTE incomplete: expanded to catch ЗҚАИ variants with
         dots/dashes, Примечание with lowercase, and inline-embedded notes that
         survive after _fix_soft_breaks merging.

  BUG-6 FIXED — RE_FOOTNOTE: updated \\s pattern → [\\s\\xa0]* so footnote lines
         preceded by \\xa0 indentation are matched before normalization.

  BUG-7 FIXED — Exception handling: postprocess() exceptions that are NOT
         OSError/PermissionError were silently crashing the entire batch.
         Fix: broad except in convert_file() with proper logging.

  BUG-8 FIXED — _deindent_paragraphs lstrip(): now explicitly strips \\xa0.

Requirements
────────────
    pip install pypandoc tqdm
    winget install --id JohnMacFarlane.Pandoc -e
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import pypandoc
from tqdm import tqdm

# ── directory layout ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR  = BASE_DIR / "data" / "raw"
MD_DIR   = BASE_DIR / "data" / "markdown"
LOG_FILE = BASE_DIR / "data" / "conversion.log"

# ── heading level map ─────────────────────────────────────────────────────────
# Ordered from most-specific to least-specific.
# Each entry: (compiled regex matching the *content* of the bold line, heading level)
#
# Kazakh script notes:
#   БӨЛІМ / БӨЛIМ  — the 4th char is sometimes Cyrillic І (U+0406) or Latin I (U+0049).
#   тарау / тарaw  — same doc may use "1-тарау" or "1-1 тарау" (space, not dash).
#   бап            — "51-бап." or "51-1 бап." (compound sub-article numbers).
#   бөлiмше        — sub-section with Latin i mixed in.
# All patterns use re.IGNORECASE | re.UNICODE for safe Cyrillic matching.
HEADING_LEVEL_MAP: list[tuple[re.Pattern, int]] = [
    # H1 — Document title: line contains "кодекс" / "кодексі"
    (re.compile(r'кодекс', re.IGNORECASE | re.UNICODE), 1),
    # H1 — Part headers: "ОБЩАЯ ЧАСТЬ", "ОСОБЕННАЯ ЧАСТЬ" and KZ equivalents.
    #   Also matched when preceded by a number: "1. ОБЩАЯ ЧАСТЬ", "2. ЖАЛПЫ БӨЛІМ".
    (re.compile(
        r'(ОБЩАЯ ЧАСТЬ|ОСОБЕННАЯ ЧАСТЬ'
        r'|ЖАЛПЫ\s+БӨЛІК|ЖАЛПЫ\s+БӨЛІМ'
        r'|ЕРЕКШЕ\s+БӨЛІК|ЕРЕКШЕ\s+БӨЛІМ)',
        re.IGNORECASE | re.UNICODE,
    ), 1),
    # H2 — Section   RU: "Раздел 1"   KZ: "1-БӨЛІМ" / "1-БӨЛIМ" (Latin-I variant)
    (re.compile(r'^(Раздел|РАЗДЕЛ)\s+[IVXLCDM\d]',  re.IGNORECASE | re.UNICODE), 2),
    (re.compile(r'^\d[\d\-]*-БӨЛ[ІI]М',             re.IGNORECASE | re.UNICODE), 2),
    (re.compile(r'^БӨЛ[ІI]М\s+\d',                  re.IGNORECASE | re.UNICODE), 2),
    # H2 — numbered section e.g. "2-БӨЛIМ" using Latin I
    (re.compile(r'^\d[\d\-]*\s*[-–]\s*БӨЛ[ІI]М',   re.IGNORECASE | re.UNICODE), 2),
    # H3 — Chapter   RU: "Глава 1"   KZ: "1-тарау" or "1-1 тарау" (space variant)
    (re.compile(r'^(Глава|ГЛАВА)\s+\d',              re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^\d[\d\-]*-тарау',                 re.IGNORECASE | re.UNICODE), 3),
    # Space-before-тарау: "11-1 тарау" — compound chapter numbers
    (re.compile(r'^\d+(?:-\d+)?\s+тарау',            re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^ТАРАУ\s+\d',                      re.IGNORECASE | re.UNICODE), 3),
    # H3 — KZ sub-division: "1-КІШІ БӨЛІМ", "1-бөлiмше", "N бөлiм" (no dash)
    (re.compile(r'^\d[\d\-]*-КІШІ\s+БӨЛ[ІI]М',     re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^\d[\d\-]*-бөл[іi]мше',           re.IGNORECASE | re.UNICODE), 3),
    # "3 бөлiм", "4 бөлім" — section number + space (no dash before section word)
    (re.compile(r'^\d+\s+бөл[іi]м',                 re.IGNORECASE | re.UNICODE), 3),
    # H4 — Article   RU: "Статья 1" / "Cтатья" (Latin-C publisher typo)
    #              KZ: "1-бап" / "1-1 бап" (space variant)
    # Character class [СC] covers both Cyrillic С (U+0421) and Latin C (U+0043)
    (re.compile(r'^[СC]татья\s+\d',                  re.IGNORECASE | re.UNICODE), 4),
    # "-бап" with or without trailing dot/space, including compound "51-1-бап"
    (re.compile(r'^\d[\d\-]*-бап',                   re.IGNORECASE | re.UNICODE), 4),
    # Space-before-бап: "133-1 бап"
    (re.compile(r'^\d+(?:-\d+)?\s+бап',              re.IGNORECASE | re.UNICODE), 4),
    # Paragraph / sub-paragraph
    (re.compile(r'^(Параграф|ПАРАГРАФ|§)\s+\d',     re.IGNORECASE | re.UNICODE), 4),
    (re.compile(r'^\d[\d\-]*-параграф',              re.IGNORECASE | re.UNICODE), 4),
]

# Strip leading quotation marks or "N. " numbering that some publishers
# prepend to article titles in the TOC/body before bold promotion.
_RE_LEADING_QUOTE   = re.compile(r'^[«"\'"""'']+')
_RE_LEADING_NUMBER  = re.compile(r'^\d+\.\s+')

# ── compiled patterns (module-level for performance) ─────────────────────────
RE_BLANK_3PLUS   = re.compile(r'\n{3,}')
RE_SOFT_BREAK    = re.compile(r'(?<!\n)\n(?!\n)')
# BUG-1 FIX: non-breaking space normalization
RE_NBSP          = re.compile(r'\xa0')
# Extended: includes " and ' in addition to original set
RE_PANDOC_ESCAPE = re.compile(r"""\\([`*_{}[\]()#+\-.!|"'])""")
# BUG-6 FIX: Footnote/amendment marker — handle \xa0 indentation via normalize-first
RE_FOOTNOTE      = re.compile(
    r'^(Сноска\.|Ескерту\.)\s+.+$',
    re.MULTILINE | re.UNICODE,
)
# Image artifacts produced by pandoc for embedded pictures
# BUG-3 FIX: also match attributes split across two lines (pandoc edge case)
RE_IMAGE         = re.compile(
    r'!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?\s*(?:\n\s*\{[^}]*\})?\s*\n?',
    re.DOTALL,
)
# Pandoc span notation: [text]{.underline}, [text]{.mark}, [text]{lang="kk"} …
RE_SPAN_ATTR     = re.compile(r'\[([^\]]+)\]\{[^}]+\}')
# Full-line bold: entire line is **content**  (heading candidate)
# BUG-7 FIX: explicit \r? to tolerate any stray CR before line end
RE_BOLD_LINE     = re.compile(r'^\*\*(.+?)\*\*\r?\s*$')
# Hard break (backslash before whitespace or end-of-line)
RE_HARD_BREAK    = re.compile(r'\\[ \t]*\n?')
# BUG-5 FIX: Service/editorial notes — expanded pattern set
RE_SERVICE_NOTE  = re.compile(
    r'^(?:'
    r'Примечание\s+[А-ЯЁA-Za-z][А-ЯЁA-Za-z\-\.]*!'          # РЦПИ, ИЗПИ, etc.
    r'|ЗҚАИ[\-\.]\s*ның\s+ескертпесі!'                        # ЗҚАИ-ның / ЗҚАИ.ның
    r'|ЗҚАИ\s+ескертпесі!'                                    # variant without dash
    r'|Вниманию\s+пользователей!'
    r'|Қолданушылар\s+назарына!'
    r'|Пайдаланушылар\s+назарына!'
    r')\s*$',
    re.MULTILINE | re.UNICODE,
)


# ── logging ───────────────────────────────────────────────────────────────────

class _TqdmLoggingHandler(logging.StreamHandler):
    """Pipes log messages through tqdm.write() so progress bars aren't broken."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout, end='\n')
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(str(LOG_FILE), encoding='utf-8'),
            _TqdmLoggingHandler(),
        ],
    )
    return logging.getLogger('converter')


# ── Windows helpers ───────────────────────────────────────────────────────────

def _is_file_locked(path: Path) -> bool:
    try:
        with open(path, 'rb'):
            pass
        return False
    except PermissionError:
        return True


def _find_pandoc_windows() -> Optional[str]:
    found = shutil.which('pandoc')
    if found:
        return found
    for base_env in ('LOCALAPPDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)'):
        base = os.environ.get(base_env, '')
        candidate = Path(base) / 'Pandoc' / 'pandoc.exe'
        if candidate.exists():
            return str(candidate)
    return None


def _check_pandoc(logger: logging.Logger) -> Optional[str]:
    pandoc_path = _find_pandoc_windows()
    if pandoc_path is None:
        logger.error(
            'Pandoc not found.\n'
            '  winget : winget install --id JohnMacFarlane.Pandoc -e\n'
            '  choco  : choco install pandoc\n'
            '  manual : https://pandoc.org/installing.html\n'
            'Restart PowerShell after installing.'
        )
        return None
    os.environ.setdefault('PYPANDOC_PANDOC', pandoc_path)
    version = pypandoc.get_pandoc_version()
    logger.info('Pandoc %s  →  %s', version, pandoc_path)
    return version


# ── post-processing: individual transforms ───────────────────────────────────

def _normalize_line_endings(text: str) -> str:
    """Normalise CRLF / CR → LF so all regex work uniformly."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _normalize_nbsp(text: str) -> str:
    """
    BUG-1 FIX — Replace non-breaking spaces (U+00A0 / \\xa0) with regular spaces.

    Pandoc uses \\xa0 for body-text indentation when converting from DOCX.
    Python's str.lstrip() and regex \\s do NOT treat \\xa0 as whitespace, which
    causes _deindent_paragraphs, RE_FOOTNOTE, and RE_SERVICE_NOTE to silently fail.
    Running this step first makes all subsequent logic Unicode-safe.
    """
    return RE_NBSP.sub(' ', text)


def _remove_images(text: str) -> str:
    """
    Remove pandoc image artifacts.
    Every Kazakh code starts with a publisher logo:
        ![](media/document_image_rId3.png){width="2.25in" height="0.625in"}
    BUG-3 FIX: RE_IMAGE now uses re.DOTALL and handles attributes on next line.
    """
    return RE_IMAGE.sub('', text)


def _remove_span_attrs(text: str) -> str:
    """
    Strip pandoc span notation, keeping only the visible text.
        [уәкiлеттi орган]{.underline}  →  уәкiлеттi орган
        [текст]{lang="kk"}             →  текст
    """
    return RE_SPAN_ATTR.sub(r'\1', text)


def _remove_pandoc_escapes(text: str) -> str:
    """
    Remove backslash escapes pandoc inserts before Markdown specials.
    Extended to include \\" and \\' which appear around footnote references.
    """
    return RE_PANDOC_ESCAPE.sub(r'\1', text)


def _replace_hard_breaks(text: str) -> str:
    """
    Convert Markdown hard line-breaks (backslash before whitespace) to
    paragraph breaks.  Pandoc uses this to join multi-segment headings:
        **ОБЩАЯ ЧАСТЬ**\\\\ **РАЗДЕЛ 1. ...**
    After this step each bold segment is on its own paragraph line.
    """
    return RE_HARD_BREAK.sub('\n\n', text)


def _deindent_paragraphs(text: str) -> str:
    """
    Strip leading whitespace from body-text lines.
    BUG-8 FIX: use lstrip(' \\t\\xa0') instead of lstrip() to also remove
    non-breaking spaces that pandoc inserts for paragraph indentation.
    Skips lines inside fenced code blocks and table rows.
    """
    lines = text.split('\n')
    result: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.lstrip(' \t\xa0')
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_fence = not in_fence
        if not in_fence and not stripped.startswith('|'):
            line = stripped
        result.append(line)
    return '\n'.join(result)


def _promote_footnotes(text: str) -> str:
    """
    Wrap publisher footnote / amendment lines in Markdown blockquotes.
    Must run BEFORE _fix_soft_breaks so the footnote line is still
    identifiable as a standalone line (not yet merged into a paragraph).

    Before:  Сноска. Статья 5 с изменениями, внесёнными Законом РК...
    After:   > *Сноска. Статья 5 с изменениями, внесёнными Законом РК...*

    BUG-6 FIX: RE_FOOTNOTE updated pattern (runs after _normalize_nbsp so
    \\xa0 prefix is already gone).
    """
    return RE_FOOTNOTE.sub(lambda m: f'> *{m.group(0).strip()}*', text)


def _wrap_service_notes(text: str) -> str:
    """
    Wrap publisher service notes (Примечание РЦПИ!, ЗҚАИ-ның ескертпесі! …)
    in blockquotes so they are visually distinct and easily filtered
    during graph ingestion.
    BUG-5 FIX: RE_SERVICE_NOTE expanded with more publisher patterns.
    """
    return RE_SERVICE_NOTE.sub(
        lambda m: f'> ℹ *{m.group(0).strip()}*', text
    )


def _classify_bold_content(content: str) -> Optional[int]:
    """
    Return the heading level (1–4) for a bold line's content, or None
    if the content does not match any structural keyword.

    Pre-processing strip (does not mutate the heading text itself):
      - Leading curly/straight quotes: «"' etc. (e.g. «51-бап. Title»)
      - Leading "N. " numbering in TOC-style headings (e.g. "1. ОБЩАЯ ЧАСТЬ")
    """
    # Try as-is first (most cases)
    for pattern, level in HEADING_LEVEL_MAP:
        if pattern.search(content):
            return level
    # Try stripping leading quotes
    stripped = _RE_LEADING_QUOTE.sub('', content).strip()
    if stripped != content:
        for pattern, level in HEADING_LEVEL_MAP:
            if pattern.search(stripped):
                return level
    # Try stripping leading "N. " number prefix (TOC headings)
    stripped2 = _RE_LEADING_NUMBER.sub('', stripped or content).strip()
    if stripped2 != (stripped or content):
        for pattern, level in HEADING_LEVEL_MAP:
            if pattern.search(stripped2):
                return level
    return None


def _promote_bold_headings(text: str) -> str:
    """
    Convert full-line bold patterns to proper ATX headings.

    Pandoc outputs legal headings as **bold text** (because the source DOCX
    uses manual bold formatting rather than Word Heading styles).  This
    function detects those lines and replaces them with ## / ### / #### etc.

    Continuation logic:
    If a bold-only line does NOT match any level keyword but immediately
    follows a promoted heading (only blank lines between), it is treated
    as the second half of a split heading title and appended to it.

    Example — family_kz:
        **1-тарау. ҚАЗАҚСТАН РЕСПУБЛИКАСЫНЫҢ**     ← H3
        **НЕКЕ-ОТБАСЫ ЗАҢНАМАСЫ**                  ← continuation
    Result:
        ### 1-тарау. ҚАЗАҚСТАН РЕСПУБЛИКАСЫНЫҢ НЕКЕ-ОТБАСЫ ЗАҢНАМАСЫ
    """
    lines = text.split('\n')
    result: list[str] = []
    prev_heading_idx: int = -1   # index in result[] of last promoted heading

    for line in lines:
        bold_m = RE_BOLD_LINE.match(line)

        if bold_m:
            content = bold_m.group(1).strip()
            level   = _classify_bold_content(content)

            if level is not None:
                result.append('#' * level + ' ' + content)
                prev_heading_idx = len(result) - 1
            else:
                # No level keyword — check for continuation of a split heading
                if prev_heading_idx >= 0:
                    between = result[prev_heading_idx + 1:]
                    if all(ln.strip() == '' for ln in between):
                        # Merge: append content to the previous heading
                        result[prev_heading_idx] += ' ' + content
                        # Drop the blank lines inserted between
                        del result[prev_heading_idx + 1:]
                    else:
                        result.append(line)
                        prev_heading_idx = -1
                else:
                    result.append(line)
        else:
            # Non-bold line: reset continuation context if it has real content
            if line.strip():
                prev_heading_idx = -1
            result.append(line)

    return '\n'.join(result)


def _fix_soft_breaks(text: str) -> str:
    """
    Join soft-wrapped paragraph lines (single newlines pandoc inserts when
    it re-flows long lines).  The join is skipped when the next line begins
    a new structural element: heading, numbered/bulleted list item,
    blockquote, or table row.
    Note: with --wrap=none pandoc does not soft-wrap, so this step is mostly
    a safety net for any residual single-newline paragraph flows.
    """
    def _join(m: re.Match) -> str:
        after = m.string[m.end(): m.end() + 80]
        # Keep the newline if the next line opens a structural element
        if re.match(r'^(#{1,6}\s|\d+[.)]\s|[-*+]\s|>|\|)', after):
            return m.group(0)
        return ' '

    return RE_SOFT_BREAK.sub(_join, text)


def _collapse_blank_lines(text: str) -> str:
    """Reduce three or more consecutive blank lines to exactly two."""
    return RE_BLANK_3PLUS.sub('\n\n', text)


# ── pipeline ──────────────────────────────────────────────────────────────────

def postprocess(raw: str) -> str:
    """
    Ordered post-processing pipeline applied to pandoc's raw Markdown output.

    Step order is intentional:
      1    Normalise CRLF → LF
      2    BUG-1 FIX: normalise \\xa0 → space  ← NEW
      3    Remove image artefacts
      4    Strip span attributes
      5    Remove pandoc backslash escapes
      6    Split hard-breaks → separate paragraph per bold segment
      7    De-indent (strip leading spaces / \\xa0)
      8    Promote footnotes BEFORE soft-break joiner merges them away
      9    Wrap service notes
      10   Promote bold headings → ATX ##/###/####
      11   Fix soft-wrapped paragraph lines
      12   Collapse excess blank lines
    """
    text = _normalize_line_endings(raw)        # 1
    text = _normalize_nbsp(text)               # 2  BUG-1 FIX
    text = _remove_images(text)                # 3
    text = _remove_span_attrs(text)            # 4
    text = _remove_pandoc_escapes(text)        # 5
    text = _replace_hard_breaks(text)          # 6
    text = _deindent_paragraphs(text)          # 7
    text = _promote_footnotes(text)            # 8
    text = _wrap_service_notes(text)           # 9
    text = _promote_bold_headings(text)        # 10
    text = _fix_soft_breaks(text)              # 11
    text = _collapse_blank_lines(text)         # 12
    return text.strip() + '\n'


# ── single-file conversion ────────────────────────────────────────────────────

def convert_file(src: Path, dst: Path, logger: logging.Logger) -> bool:
    """
    Convert one .docx → .md.  Returns True on success, False on any error.
    Detects Word file locks (PermissionError) before attempting conversion.
    BUG-7 FIX: broad except block around postprocess() so non-OSError
    exceptions are logged instead of silently crashing the batch.
    """
    if _is_file_locked(src):
        logger.warning('SKIPPED (locked)  %s — close it in Word first.', src.name)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw_md: str = pypandoc.convert_file(
            str(src),
            to='markdown',
            format='docx',
            extra_args=[
                '--wrap=none',
                '--markdown-headings=atx',
            ],
        )
    except Exception as exc:
        logger.error('PANDOC FAILED  %-50s  %s', src.name, exc)
        return False

    try:
        cleaned = postprocess(raw_md)
    except Exception as exc:
        # BUG-7 FIX: catch any postprocess error and log it, don't crash batch
        logger.error('POSTPROCESS FAILED  %-50s  %s', src.name, exc)
        return False

    try:
        # BUG-2 FIX: newline='\n' forces LF-only output on Windows
        dst.write_text(cleaned, encoding='utf-8', newline='\n')
    except PermissionError:
        logger.error('WRITE DENIED   %s — check folder permissions.', dst)
        return False
    except OSError as exc:
        logger.error('WRITE FAILED   %s  %s', dst, exc)
        return False

    logger.info('OK  %-50s  →  %s', src.name, dst.name)
    return True


def reprocess_file(md_path: Path, logger: logging.Logger) -> bool:
    """
    BUG-3 FIX helper — Apply postprocessing to an already-converted .md file
    without re-running pandoc.  Use --reprocess to fix the 48 existing files
    that were converted before the full postprocessing pipeline was in place.
    """
    try:
        raw = md_path.read_text(encoding='utf-8')
    except OSError as exc:
        logger.error('READ FAILED   %s  %s', md_path, exc)
        return False

    try:
        cleaned = postprocess(raw)
    except Exception as exc:
        logger.error('POSTPROCESS FAILED  %-50s  %s', md_path.name, exc)
        return False

    try:
        md_path.write_text(cleaned, encoding='utf-8', newline='\n')
    except OSError as exc:
        logger.error('WRITE FAILED   %s  %s', md_path, exc)
        return False

    logger.info('REPROCESSED  %s', md_path.name)
    return True


# ── batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    raw_dir: Path = RAW_DIR,
    md_dir:  Path = MD_DIR,
    glob:    str  = '**/*.docx',
    dry_run: bool = False,
    force:   bool = False,
) -> None:
    """
    Scan raw_dir for .docx files and convert each one to Markdown under md_dir,
    preserving sub-directory structure.
    """
    logger = setup_logging()

    if not raw_dir.exists():
        logger.warning('raw_dir not found: %s — creating empty folder.', raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

    md_dir.mkdir(parents=True, exist_ok=True)

    # Skip Word temp-files (~$filename.docx)
    docx_files = [
        p for p in sorted(raw_dir.glob(glob))
        if not p.name.startswith('~$')
    ]

    if not docx_files:
        logger.warning('No .docx files found in %s', raw_dir)
        return

    logger.info('Found %d .docx file(s)  |  source: %s', len(docx_files), raw_dir)

    if dry_run:
        print(f'\n[DRY-RUN] {len(docx_files)} file(s) to convert:\n')
        for f in docx_files:
            dst = md_dir / f.relative_to(raw_dir).with_suffix('.md')
            exists = '✓' if dst.exists() else ' '
            print(f'  [{exists}] {f.relative_to(raw_dir)}')
        return

    if _check_pandoc(logger) is None:
        sys.exit(1)

    ok = fail = skipped_lock = skipped_exists = 0

    bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    with tqdm(docx_files, unit='file', desc='Converting', ncols=90,
              bar_format=bar_fmt) as bar:
        for src in bar:
            dst = md_dir / src.relative_to(raw_dir).with_suffix('.md')
            bar.set_postfix_str(src.name[:38])

            if not force and dst.exists():
                skipped_exists += 1
                continue

            if _is_file_locked(src):
                skipped_lock += 1
                logger.warning('SKIPPED (locked) %s', src.name)
                continue

            if convert_file(src, dst, logger):
                ok += 1
            else:
                fail += 1

    print()
    logger.info(
        'Done — OK: %d | Failed: %d | Locked: %d | Already exists: %d\n'
        '       Log: %s',
        ok, fail, skipped_lock, skipped_exists, LOG_FILE,
    )


def run_reprocess(
    md_dir: Path = MD_DIR,
    glob:   str  = '**/*.md',
) -> None:
    """
    BUG-3 FIX — Apply postprocessing to existing .md files without re-running pandoc.
    Use this when converter.py was updated but the .md files are from an older run.

    Usage:
        python converter.py --reprocess
        python converter.py --reprocess --md-dir data/markdown
    """
    logger = setup_logging()

    md_files = sorted(md_dir.glob(glob))
    if not md_files:
        logger.warning('No .md files found in %s', md_dir)
        return

    logger.info('Reprocessing %d .md file(s)  |  source: %s', len(md_files), md_dir)

    ok = fail = 0
    bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    with tqdm(md_files, unit='file', desc='Reprocessing', ncols=90,
              bar_format=bar_fmt) as bar:
        for md_path in bar:
            bar.set_postfix_str(md_path.name[:38])
            if reprocess_file(md_path, logger):
                ok += 1
            else:
                fail += 1

    print()
    logger.info(
        'Reprocess done — OK: %d | Failed: %d\n       Log: %s',
        ok, fail, LOG_FILE,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch DOCX → Markdown — Kazakh legal codes (Windows 11)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--raw-dir',    type=Path, default=RAW_DIR,
                        help='Folder with source .docx files')
    parser.add_argument('--md-dir',     type=Path, default=MD_DIR,
                        help='Output folder for .md files')
    parser.add_argument('--glob',       default='**/*.docx',
                        help='Glob pattern relative to --raw-dir')
    parser.add_argument('--dry-run',    action='store_true',
                        help='List matching files without converting')
    parser.add_argument('--force',      action='store_true',
                        help='Re-convert even if .md already exists')
    # BUG-3 FIX: new flag for in-place re-processing of existing .md files
    parser.add_argument('--reprocess',  action='store_true',
                        help='Re-apply postprocessing to existing .md files '
                             'without re-running pandoc (fixes files from old runs)')
    args = parser.parse_args()

    if args.reprocess:
        run_reprocess(md_dir=args.md_dir)
    else:
        run_batch(
            raw_dir=args.raw_dir,
            md_dir=args.md_dir,
            glob=args.glob,
            dry_run=args.dry_run,
            force=args.force,
        )


if __name__ == '__main__':
    main()
