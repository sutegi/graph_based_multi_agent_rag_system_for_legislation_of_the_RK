from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path
from typing import Any
from uuid import uuid5, NAMESPACE_X500
from tqdm import tqdm


class _TqdmLoggingHandler(logging.StreamHandler):
    """Pipes log messages through tqdm.write() so progress bars aren't broken."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout, end='\n')
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_TqdmLoggingHandler()],
)
logger = logging.getLogger("convert_to_json")

DEFAULT_INPUT_DIR  = Path("./data/markdown")
DEFAULT_OUTPUT_DIR = Path("./data/structured")

_RE_ATX   = re.compile(r"^#{1,6}\s+")
_RE_BWRAP = re.compile(r"^\*{1,3}(.*?)\*{1,3}$", re.DOTALL)
_RE_BINL  = re.compile(r"\*{1,3}(.*?)\*{1,3}", re.DOTALL)
_RE_UNDER = re.compile(r"_{1,2}(.*?)_{1,2}", re.DOTALL)
_RE_CODE  = re.compile(r"`(.+?)`")
_RE_HTML  = re.compile(r"<[^>]+>")
_RE_MSP   = re.compile(r"  +")


def strip_markdown(text):
    if not text:
        return ""
    t = text.replace("\xa0", " ")
    t = _RE_HTML.sub("", t)
    t = _RE_ATX.sub("", t)
    t = _RE_BINL.sub(r"\1", t)
    t = _RE_UNDER.sub(r"\1", t)
    t = _RE_CODE.sub(r"\1", t)
    t = _RE_MSP.sub(" ", t)
    return t.strip()


def bare(block):
    first = block.splitlines()[0].strip()
    first = _RE_ATX.sub("", first)
    first = first.replace("\xa0", " ")
    first = _RE_HTML.sub("", first)
    m = _RE_BWRAP.match(first)
    if m:
        first = m.group(1).strip()
    return first.strip()


# Article: RU  (C or S-Cyrillic + tatya + number)
_ART_RU = re.compile(
    r"^[СC]татья\s+([\d]+(?:-[\d]+)*)[\.\s]",
    re.I | re.U,
)
# Article: KZ  (number + -bap)
_ART_KZ = re.compile(
    r"^([\d]+(?:-[\d]+)*)-(?:бап|баn|бабы|баптың)[\.\s]?",
    re.I | re.U,
)
# Paragraph: KZ  N-paragraf
_PAR_KZ = re.compile(
    r"^[\d][\d\-]*-параграф[\.\s]",
    re.I | re.U,
)
# Paragraph: RU  Paragraf N  or  sign N
_PAR_RU = re.compile(
    r"^(?:§\s*\d+|Параграф\s+\d+)",
    re.I | re.U,
)
# Section: RAZDEL / BOLIM
_SEC = re.compile(
    r"\b(?:РАЗДЕЛ|Б[ӨО]Л[І2Ii]М)\b",
    re.I | re.U,
)
# Chapter: Glava / tarau
_CH = re.compile(
    r"\b(?:Глава|тарау)\b",
    re.I | re.U,
)


def classify(b):
    m = _ART_RU.match(b)
    if m: return "article", m.group(1)
    m = _ART_KZ.match(b)
    if m: return "article", m.group(1)
    if _PAR_KZ.match(b) or _PAR_RU.match(b): return "paragraph", ""
    if _SEC.search(b): return "section", ""
    if _CH.search(b):  return "chapter", ""
    return "content", ""


def make_uuid(*parts):
    return str(uuid5(NAMESPACE_X500, "|".join(str(p) for p in parts)))


def parse_lang_and_code(filename):
    stem = Path(filename).stem
    if stem.endswith("_ru"): return stem[:-3], "ru"
    if stem.endswith("_kz"): return stem[:-3], "kz"
    return stem, "unknown"


def preprocess_md(path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.error("Cannot read %s: %s", path, e)
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = text.splitlines()
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.endswith("\\") and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            b2  = re.sub(r"[*_`#]", "", nxt)
            if b2 and b2[0].islower():
                merged.append(line[:-1].rstrip() + " " + nxt)
                i += 2; continue
            else:
                merged.append(line[:-1].rstrip())
                merged.append("")
        else:
            merged.append(line)
        i += 1
    raw = "\n".join(merged).split("\n\n")
    blocks = []
    for blk in raw:
        blk = blk.strip()
        if not blk: continue
        bl = blk.splitlines()
        if any(l.strip().startswith("|") for l in bl):
            blocks.append("\n".join(bl))
        else:
            blocks.append(" ".join(l.strip() for l in bl if l.strip()))
    return blocks


def parse_blocks(blocks, code_name, lang):
    articles = []
    hier = {"section": None, "chapter": None, "paragraph": None}
    cur  = None

    def flush():
        nonlocal cur
        if cur is not None:
            articles.append(cur)
            cur = None

    for blk in blocks:
        blk = _RE_HTML.sub("", blk).replace("\xa0", " ").strip()
        if not blk or blk.startswith(">"): continue
        b = bare(blk)
        if not b: continue
        kind, num = classify(b)

        if kind == "section":
            flush()
            hier = {"section": strip_markdown(b), "chapter": None, "paragraph": None}
        elif kind == "chapter":
            flush()
            hier["chapter"]   = strip_markdown(b)
            hier["paragraph"] = None
        elif kind == "paragraph":
            flush()
            hier["paragraph"] = strip_markdown(b)
        elif kind == "article":
            flush()
            title    = strip_markdown(blk.splitlines()[0])
            ctx_rest = " ".join(l.strip() for l in blk.splitlines()[1:] if l.strip())
            h_path   = " / ".join(p for p in [hier["section"], hier["chapter"], hier["paragraph"]] if p) or None
            cur = {
                "id":        make_uuid("article", code_name, lang, num),
                "number":    num,
                "title":     title,
                "hierarchy": h_path,
                "section":   hier["section"],
                "chapter":   hier["chapter"],
                "paragraph": hier["paragraph"],
                "context":   ctx_rest,
            }
        else:
            if cur is not None:
                sep = "\n\n" if cur["context"] else ""
                cur["context"] = cur["context"] + sep + blk

    flush()
    return articles


def _save_json(data, path):
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError as e:
        logger.error("Cannot save %s: %s", path, e)
        tmp.unlink(missing_ok=True)
        raise


def process_file(path, output_dir):
    code_name, lang = parse_lang_and_code(path.name)
    blocks   = preprocess_md(path)
    articles = parse_blocks(blocks, code_name, lang)
    _save_json(articles, output_dir / f"{path.stem}.json")
    return len(articles)


def run(input_dir, output_dir, single_file=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    md_files = [input_dir / single_file] if single_file else sorted(input_dir.glob("*.md"))
    if not md_files:
        logger.error("No .md files found in %s", input_dir); sys.exit(1)
    logger.info("Processing %d file(s): %s -> %s", len(md_files), input_dir, output_dir)
    total = 0; failed = 0
    for p in tqdm(md_files, desc="Files", unit="file"):
        try:
            n = process_file(p, output_dir)
            logger.info("  %-40s -> %d articles", p.name, n)
            total += n
        except Exception as e:
            logger.error("FAILED: %s - %s", p.name, e, exc_info=True)
            failed += 1
    logger.info("Done. %d files | %d articles | %d failed",
                len(md_files) - failed, total, failed)


def parse_args():
    p = argparse.ArgumentParser(
        description="Parse Kazakh legal .md files into structured article JSON.")
    p.add_argument("--input",  "-i", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--file",   "-f", default=None,
                   help="Single filename relative to --input (e.g. family_ru.md)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(input_dir=args.input, output_dir=args.output, single_file=args.file)
