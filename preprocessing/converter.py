"""converter.py v4 — Batch DOCX → Markdown converter
for 24 Codes of the Republic of Kazakhstan (RU + KZ)

Platform : Windows 11 (native Python, no WSL)
Author   : Senior Python Developer / Data Engineer
Project  : Multi-Agent Graph-Based RAG (Neo4j)
Encoding : UTF-8 — Kazakh Cyrillic with diacritics (Ә І Ң Ү Қ Ғ Һ Ө)

Bug-fixes v4 (over v3)
──────────────────────
  BUG-1 FIXED — Non-breaking spaces (\xa0, U+00A0): pandoc uses \xa0 for body-text
         indentation."""

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

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR  = BASE_DIR / "data" / "raw"
MD_DIR   = BASE_DIR / "data" / "markdown"
LOG_FILE = BASE_DIR / "data" / "conversion.log"

HEADING_LEVEL_MAP: list[tuple[re.Pattern, int]] = [
    (re.compile(r'кодекс', re.IGNORECASE | re.UNICODE), 1),
    (re.compile(
        r'(ОБЩАЯ ЧАСТЬ|ОСОБЕННАЯ ЧАСТЬ'
        r'|ЖАЛПЫ\s+БӨЛІК|ЖАЛПЫ\s+БӨЛІМ'
        r'|ЕРЕКШЕ\s+БӨЛІК|ЕРЕКШЕ\s+БӨЛІМ)',
        re.IGNORECASE | re.UNICODE,
    ), 1),
    (re.compile(r'^(Раздел|РАЗДЕЛ)\s+[IVXLCDM\d]',  re.IGNORECASE | re.UNICODE), 2),
    (re.compile(r'^\d[\d\-]*-БӨЛ[ІI]М',             re.IGNORECASE | re.UNICODE), 2),
    (re.compile(r'^БӨЛ[ІI]М\s+\d',                  re.IGNORECASE | re.UNICODE), 2),
    (re.compile(r'^\d[\d\-]*\s*[-–]\s*БӨЛ[ІI]М',   re.IGNORECASE | re.UNICODE), 2),
    (re.compile(r'^(Глава|ГЛАВА)\s+\d',              re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^\d[\d\-]*-тарау',                 re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^\d+(?:-\d+)?\s+тарау',            re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^ТАРАУ\s+\d',                      re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^\d[\d\-]*-КІШІ\s+БӨЛ[ІI]М',     re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^\d[\d\-]*-бөл[іi]мше',           re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^\d+\s+бөл[іi]м',                 re.IGNORECASE | re.UNICODE), 3),
    (re.compile(r'^[СC]татья\s+\d',                  re.IGNORECASE | re.UNICODE), 4),
    (re.compile(r'^\d[\d\-]*-бап',                   re.IGNORECASE | re.UNICODE), 4),
    (re.compile(r'^\d+(?:-\d+)?\s+бап',              re.IGNORECASE | re.UNICODE), 4),
    (re.compile(r'^(Параграф|ПАРАГРАФ|§)\s+\d',     re.IGNORECASE | re.UNICODE), 4),
    (re.compile(r'^\d[\d\-]*-параграф',              re.IGNORECASE | re.UNICODE), 4),
]

_RE_LEADING_QUOTE   = re.compile(r'^[«"\'"""'']+')
_RE_LEADING_NUMBER  = re.compile(r'^\d+\.\s+')

RE_BLANK_3PLUS   = re.compile(r'\n{3,}')
RE_SOFT_BREAK    = re.compile(r'(?<!\n)\n(?!\n)')
RE_NBSP          = re.compile(r'\xa0')
RE_PANDOC_ESCAPE = re.compile(r"""\\([`*_{}[\]()#+\-.!|"'])""")
RE_FOOTNOTE      = re.compile(
    r'^(Сноска\.|Ескерту\.)\s+.+$',
    re.MULTILINE | re.UNICODE,
)
RE_IMAGE         = re.compile(
    r'!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?\s*(?:\n\s*\{[^}]*\})?\s*\n?',
    re.DOTALL,
)
RE_SPAN_ATTR     = re.compile(r'\[([^\]]+)\]\{[^}]+\}')
RE_BOLD_LINE     = re.compile(r'^\*\*(.+?)\*\*\r?\s*$')
RE_HARD_BREAK    = re.compile(r'\\[ \t]*\n?')
RE_SERVICE_NOTE  = re.compile(
    r'^(?:'
    r'Примечание\s+[А-ЯЁA-Za-z][А-ЯЁA-Za-z\-\.]*!'
    r'|ЗҚАИ[\-\.]\s*ның\s+ескертпесі!'
    r'|ЗҚАИ\s+ескертпесі!'
    r'|Вниманию\s+пользователей!'
    r'|Қолданушылар\s+назарына!'
    r'|Пайдаланушылар\s+назарына!'
    r')\s*$',
    re.MULTILINE | re.UNICODE,
)


class _TqdmLoggingHandler(logging.StreamHandler):
    """Pipes log messages through tqdm.write() so progress bars aren't broken."""
    def emit(self, record: logging.LogRecord) -> None:
        """Function emit."""
        try:
            tqdm.write(self.format(record), file=sys.stdout, end='\n')
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging() -> logging.Logger:
    """Function setup_logging."""
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


def _is_file_locked(path: Path) -> bool:
    """Function _is_file_locked."""
    try:
        with open(path, 'rb'):
            pass
        return False
    except PermissionError:
        return True


def _find_pandoc_windows() -> Optional[str]:
    """Function _find_pandoc_windows."""
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
    """Function _check_pandoc."""
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


def _normalize_line_endings(text: str) -> str:
    """Normalise CRLF / CR → LF so all regex work uniformly."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _normalize_nbsp(text: str) -> str:
    """BUG-1 FIX — Replace non-breaking spaces (U+00A0 / \xa0) with regular spaces."""
    return RE_NBSP.sub(' ', text)


def _remove_images(text: str) -> str:
    """Remove pandoc image artifacts."""
    return RE_IMAGE.sub('', text)


def _remove_span_attrs(text: str) -> str:
    """Strip pandoc span notation, keeping only the visible text."""
    return RE_SPAN_ATTR.sub(r'\1', text)


def _remove_pandoc_escapes(text: str) -> str:
    """Remove backslash escapes pandoc inserts before Markdown specials."""
    return RE_PANDOC_ESCAPE.sub(r'\1', text)


def _replace_hard_breaks(text: str) -> str:
    """Convert Markdown hard line-breaks (backslash before whitespace) to
    paragraph breaks."""
    return RE_HARD_BREAK.sub('\n\n', text)


def _deindent_paragraphs(text: str) -> str:
    """Strip leading whitespace from body-text lines."""
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
    """Wrap publisher footnote / amendment lines in Markdown blockquotes."""
    return RE_FOOTNOTE.sub(lambda m: f'> *{m.group(0).strip()}*', text)


def _wrap_service_notes(text: str) -> str:
    """Wrap publisher service notes (Примечание РЦПИ!, ЗҚАИ-ның ескертпесі!"""
    return RE_SERVICE_NOTE.sub(
        lambda m: f'> ℹ *{m.group(0).strip()}*', text
    )


def _classify_bold_content(content: str) -> Optional[int]:
    """Return the heading level (1–4) for a bold line's content, or None
    if the content does not match any structural keyword."""
    for pattern, level in HEADING_LEVEL_MAP:
        if pattern.search(content):
            return level
    stripped = _RE_LEADING_QUOTE.sub('', content).strip()
    if stripped != content:
        for pattern, level in HEADING_LEVEL_MAP:
            if pattern.search(stripped):
                return level
    stripped2 = _RE_LEADING_NUMBER.sub('', stripped or content).strip()
    if stripped2 != (stripped or content):
        for pattern, level in HEADING_LEVEL_MAP:
            if pattern.search(stripped2):
                return level
    return None


def _promote_bold_headings(text: str) -> str:
    """Convert full-line bold patterns to proper ATX headings."""
    lines = text.split('\n')
    result: list[str] = []
    prev_heading_idx: int = -1

    for line in lines:
        bold_m = RE_BOLD_LINE.match(line)

        if bold_m:
            content = bold_m.group(1).strip()
            level   = _classify_bold_content(content)

            if level is not None:
                result.append('#' * level + ' ' + content)
                prev_heading_idx = len(result) - 1
            else:
                if prev_heading_idx >= 0:
                    between = result[prev_heading_idx + 1:]
                    if all(ln.strip() == '' for ln in between):
                        result[prev_heading_idx] += ' ' + content
                        del result[prev_heading_idx + 1:]
                    else:
                        result.append(line)
                        prev_heading_idx = -1
                else:
                    result.append(line)
        else:
            if line.strip():
                prev_heading_idx = -1
            result.append(line)

    return '\n'.join(result)


def _fix_soft_breaks(text: str) -> str:
    """Join soft-wrapped paragraph lines (single newlines pandoc inserts when
    it re-flows long lines)."""
    def _join(m: re.Match) -> str:
        """Function _join."""
        after = m.string[m.end(): m.end() + 80]
        if re.match(r'^(#{1,6}\s|\d+[.)]\s|[-*+]\s|>|\|)', after):
            return m.group(0)
        return ' '

    return RE_SOFT_BREAK.sub(_join, text)


def _collapse_blank_lines(text: str) -> str:
    """Reduce three or more consecutive blank lines to exactly two."""
    return RE_BLANK_3PLUS.sub('\n\n', text)


def postprocess(raw: str) -> str:
    """Ordered post-processing pipeline applied to pandoc's raw Markdown output."""
    text = _normalize_line_endings(raw)
    text = _normalize_nbsp(text)
    text = _remove_images(text)
    text = _remove_span_attrs(text)
    text = _remove_pandoc_escapes(text)
    text = _replace_hard_breaks(text)
    text = _deindent_paragraphs(text)
    text = _promote_footnotes(text)
    text = _wrap_service_notes(text)
    text = _promote_bold_headings(text)
    text = _fix_soft_breaks(text)
    text = _collapse_blank_lines(text)
    return text.strip() + '\n'


def convert_file(src: Path, dst: Path, logger: logging.Logger) -> bool:
    """Convert one .docx → .md."""
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
        logger.error('POSTPROCESS FAILED  %-50s  %s', src.name, exc)
        return False

    try:
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
    """BUG-3 FIX helper — Apply postprocessing to an already-converted .md file
    without re-running pandoc."""
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


def run_batch(
    raw_dir: Path = RAW_DIR,
    md_dir:  Path = MD_DIR,
    glob:    str  = '**/*.docx',
    dry_run: bool = False,
    force:   bool = False,
) -> None:
    """Scan raw_dir for .docx files and convert each one to Markdown under md_dir,
    preserving sub-directory structure."""
    logger = setup_logging()

    if not raw_dir.exists():
        logger.warning('raw_dir not found: %s — creating empty folder.', raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

    md_dir.mkdir(parents=True, exist_ok=True)

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
    """BUG-3 FIX — Apply postprocessing to existing .md files without re-running pandoc."""
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


def main() -> None:
    """Function main."""
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
