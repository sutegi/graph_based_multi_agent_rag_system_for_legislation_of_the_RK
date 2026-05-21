"""Extracts bilingual keywords and typed relations via DeepSeek API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv
except ImportError:
    print("[ERROR] python-dotenv not installed.  Run: pip install python-dotenv")
    sys.exit(1)

load_dotenv()

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    import pydantic
    if int(pydantic.VERSION.split(".")[0]) < 2:
        raise ImportError("Pydantic v2 required.")
except ImportError as e:
    print(f"[ERROR] {e}\n  Run: pip install 'pydantic>=2.0'")
    sys.exit(1)

try:
    from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError
except ImportError:
    print("[ERROR] openai not installed.  Run: pip install 'openai>=1.0'")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("deepseek_extractor")


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"
MAX_RETRIES       = 3
RETRY_DELAY_SEC   = 5

MAX_TOKENS = 8192

MAX_ARTICLES_PER_BATCH = 15

MAX_CHARS_PER_BATCH = 60_000

KW_SOFT_CAP = 20


ALLOWED_RELATION_TYPES = Literal[
    "CAUSED_BY",
    "BASED_ON",
    "MANIFESTED_IN",
    "CORRELATED_WITH",
    "EXTENDS",
    "REFERENCES",
]


def _trim_kw(v: list) -> list:
    """Deduplicate, lowercase, cap at KW_SOFT_CAP."""
    seen: set[str] = set()
    result: list[str] = []
    for kw in v:
        norm = str(kw).strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
            if len(result) == KW_SOFT_CAP:
                break
    return result


class BilingualKeywords(BaseModel):
    """Keyword lists in both languages."""
    ru: list[str] = Field(
        ..., min_length=1,
        description="5-15 legal terms in Russian (lowercase). Max 20. REQUIRED.",
    )
    kz: list[str] = Field(
        ..., min_length=1,
        description="5-15 legal terms in Kazakh (lowercase). Max 20. REQUIRED.",
    )

    @field_validator("ru", "kz", mode="before")
    @classmethod
    def _trim(cls, v: list) -> list:
        """Function _trim."""
        return _trim_kw(v)


class ArticleKeywords(BaseModel):
    """Class ArticleKeywords."""
    article_id: str = Field(..., description="Exact id from input article header.")
    keywords: BilingualKeywords = Field(
        ..., description='{"ru": [...], "kz": [...]}. Both lists REQUIRED.',
    )


class RelationMetadata(BaseModel):
    """Class RelationMetadata."""
    source_kz: str | None = Field(default=None,
        description="Kazakh label for source node (e.g. '45-bap').")
    target_kz: str | None = Field(default=None,
        description="Kazakh label for target node (e.g. '188-bap').")
    ref_type: Literal["internal", "external"] | None = Field(default=None,
        description="REFERENCES only: 'internal'=same doc, 'external'=other act.")
    target_label: str | None = Field(default=None,
        description="REFERENCES only: Russian label, e.g. 'Statya 14'.")


class Relation(BaseModel):
    """Class Relation."""
    source_id: str = Field(..., description="Exact id of source article.")
    target_id: str = Field(..., description="Exact id of target article or external act.")
    type: ALLOWED_RELATION_TYPES = Field(..., description="Semantic relationship type.")
    evidence: str = Field(..., description="Verbatim substring from source article text.")
    metadata: RelationMetadata = Field(
        default_factory=RelationMetadata,
        description="Always present. source_kz/target_kz for all types. "
                    "ref_type/target_label ONLY for REFERENCES.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "Relation":
        """Function _validate."""
        if self.type == "REFERENCES" and self.metadata.ref_type is None:
            raise ValueError("metadata.ref_type required when type='REFERENCES'.")
        if self.type != "REFERENCES":
            object.__setattr__(self, "metadata",
                RelationMetadata(
                    source_kz=self.metadata.source_kz,
                    target_kz=self.metadata.target_kz,
                ))
        return self


class ExtractionResult(BaseModel):
    """Class ExtractionResult."""
    keywords:  list[ArticleKeywords] = Field(..., description="Bilingual keywords per article.")
    relations: list[Relation]        = Field(..., description="Typed relationships.")


SYSTEM_PROMPT = """\
You are a senior legal analyst specialising in Kazakhstani legislation."""

CORRECTION_PROMPT = """\
The following articles have EMPTY keyword arrays, which is invalid (every graph node \
needs keywords in both languages):

{problems}

Please return the COMPLETE corrected JSON with all empty arrays filled."""


class DeepSeekExtractor:
    """Sends article batches to DeepSeek; returns validated ExtractionResult."""

    def __init__(self, api_key: str, model: str = DEEPSEEK_MODEL) -> None:
        """Function __init__."""
        self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self._model  = model
        self._schema = ExtractionResult.model_json_schema()

    def _build_messages(self, user_text: str) -> list[dict[str, str]]:
        """Function _build_messages."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]

    def _call_with_schema(self, messages: list[dict]) -> tuple[str, str]:
        """json_schema without strict=True (strict causes HTTP 400 with anyOf schemas)."""
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=MAX_TOKENS,
            temperature=0.0,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "ExtractionResult",
                                             "schema": self._schema}},
        )
        c = resp.choices[0]
        return c.message.content or "", c.finish_reason or ""

    def _call_json_object(self, messages: list[dict]) -> tuple[str, str]:
        """Fallback plain JSON mode."""
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=MAX_TOKENS,
            temperature=0.0, response_format={"type": "json_object"},
        )
        c = resp.choices[0]
        return c.message.content or "", c.finish_reason or ""

    @staticmethod
    def _parse_raw(raw: str) -> dict:
        """Strip markdown fences and parse JSON."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)

    @staticmethod
    def _filter_relations(raw_obj: dict) -> dict:
        """Type 3 fix: drop relations missing required fields BEFORE Pydantic validation."""
        REQUIRED = {"source_id", "target_id", "type", "evidence"}
        original = raw_obj.get("relations", [])
        valid = [r for r in original if REQUIRED.issubset(r.keys())]
        dropped = len(original) - len(valid)
        if dropped:
            logger.warning("  Dropped %d relation(s) with missing required fields.", dropped)
        raw_obj["relations"] = valid
        return raw_obj

    def _correction_turn(
        self,
        raw_obj: dict,
        messages: list[dict],
    ) -> dict:
        """Type 1 fix: if any article has empty ru or kz keyword list,
        send a correction turn asking the model to fill them."""
        problems: list[str] = []
        for kw in raw_obj.get("keywords", []):
            art_id = kw.get("article_id", "?")
            kws = kw.get("keywords", {})
            if not kws.get("ru"):
                problems.append(f"  - article '{art_id}': 'ru' is empty []")
            if not kws.get("kz"):
                problems.append(f"  - article '{art_id}': 'kz' is empty []")

        if not problems:
            return raw_obj

        logger.warning(
            "  Empty keyword lists detected (%d). Sending correction turn ...",
            len(problems),
        )

        correction_messages = messages + [
            {"role": "assistant",
             "content": json.dumps(raw_obj, ensure_ascii=False)},
            {"role": "user",
             "content": CORRECTION_PROMPT.format(problems="\n".join(problems))},
        ]

        try:
            raw_fixed, finish_reason = self._call_json_object(correction_messages)
            if finish_reason == "length" or not raw_fixed.strip():
                logger.warning("  Correction turn failed (length/empty). Keeping original.")
                return raw_obj
            fixed_obj = self._parse_raw(raw_fixed)
            broken_ids = {
                kw.get("article_id")
                for kw in raw_obj.get("keywords", [])
                if not kw.get("keywords", {}).get("ru")
                   or not kw.get("keywords", {}).get("kz")
            }
            fixed_map = {
                kw.get("article_id"): kw
                for kw in fixed_obj.get("keywords", [])
                if kw.get("article_id") in broken_ids
            }
            merged_kw = [
                fixed_map.get(kw.get("article_id"), kw)
                for kw in raw_obj.get("keywords", [])
            ]
            raw_obj["keywords"] = merged_kw
            logger.info("  Correction turn applied (%d articles fixed).", len(fixed_map))
        except Exception as exc:
            logger.warning("  Correction turn error: %s. Keeping original.", exc)

        return raw_obj

    def extract(self, user_text: str) -> ExtractionResult:
        """Call DeepSeek with retries; return validated ExtractionResult."""
        messages  = self._build_messages(user_text)
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("  API call attempt %d/%d ...", attempt, MAX_RETRIES)

                try:
                    raw, finish_reason = self._call_with_schema(messages)
                    logger.debug("  Used json_schema mode.")
                except APIStatusError as e:
                    if e.status_code == 400:
                        logger.warning("  json_schema rejected -- trying json_object.")
                        raw, finish_reason = self._call_json_object(messages)
                    else:
                        raise

                if finish_reason == "length":
                    raise RuntimeError(
                        "finish_reason='length': output truncated. "
                        "Auto-splitting batch."
                    )

                if not raw.strip():
                    raise ValueError("Empty response from API.")

                raw_obj = self._parse_raw(raw)

                raw_obj = self._correction_turn(raw_obj, messages)

                raw_obj = self._filter_relations(raw_obj)

                result = ExtractionResult.model_validate(raw_obj)
                logger.info("  -> %d kw-sets, %d relations.",
                            len(result.keywords), len(result.relations))
                return result

            except RuntimeError as exc:
                if "finish_reason='length'" in str(exc):
                    raise
                logger.warning("  Runtime error on attempt %d: %s", attempt, exc)
                last_exc = exc
            except (APIConnectionError, APITimeoutError) as exc:
                logger.warning("  Network error on attempt %d: %s", attempt, exc)
                last_exc = exc
            except APIStatusError as exc:
                if exc.status_code in (401, 403):
                    logger.error("  Auth failed -- check DEEPSEEK_API_KEY.")
                    raise
                logger.warning("  API error %d on attempt %d: %s",
                               exc.status_code, attempt, exc)
                last_exc = exc
            except Exception as exc:
                logger.warning("  Error on attempt %d: %s", attempt, exc)
                last_exc = exc

            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_SEC * (2 ** (attempt - 1))
                logger.info("  Retrying in %d s ...", delay)
                time.sleep(delay)

        raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last: {last_exc}")


def _extract_with_autosplit(
    extractor: DeepSeekExtractor,
    batch: list[dict[str, str]],
) -> list[ExtractionResult]:
    """Type 2 fix: if batch causes finish_reason='length', split in half and
    retry each sub-batch recursively until it fits or reaches single article."""
    try:
        return [extractor.extract(format_batch(batch))]
    except RuntimeError as exc:
        if "finish_reason='length'" not in str(exc):
            raise
        if len(batch) == 1:
            raise RuntimeError(
                f"Single article '{batch[0]['id']}' exceeds output token limit. "
                "Article may be too long -- consider splitting manually."
            ) from exc
        mid = len(batch) // 2
        logger.warning(
            "  Auto-split: batch of %d -> %d + %d",
            len(batch), mid, len(batch) - mid,
        )
        left  = _extract_with_autosplit(extractor, batch[:mid])
        right = _extract_with_autosplit(extractor, batch[mid:])
        return left + right


def load_articles(path: Path, json_field: str | None) -> list[dict[str, str]]:
    """Load full-text articles from *_merged.json or plain .txt."""
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Expected a JSON array.")

        articles: list[dict[str, str]] = []
        for r in records:
            art_id   = str(r.get("article_id") or r.get("id") or "unknown")
            number   = str(r.get("number", ""))
            title_ru = str(r.get("title_ru") or "")
            title_kz = str(r.get("title_kz") or "")

            if json_field:
                body_ru, body_kz = str(r.get(json_field) or ""), ""
            else:
                body_ru = str(r.get("context_ru") or r.get("content") or "")
                body_kz = str(r.get("context_kz") or "")

            articles.append({
                "id": art_id, "number": number,
                "title_ru": title_ru, "title_kz": title_kz,
                "body_ru": body_ru,   "body_kz": body_kz,
                "has_kz": "1" if body_kz.strip() else "0",
            })
        return articles

    return [{"id": "text_input", "number": "",
             "title_ru": path.stem, "title_kz": "",
             "body_ru": path.read_text(encoding="utf-8"),
             "body_kz": "", "has_kz": "0"}]


def format_batch(articles: list[dict[str, str]]) -> str:
    """Render bilingual article dicts into user-message string."""
    parts: list[str] = []
    for a in articles:
        lines = ["### ARTICLE", f"id: {a['id']}", f"number: {a['number']}"]
        if a["title_ru"]:
            lines.append(f"title_ru: {a['title_ru']}")
        if a["title_kz"]:
            lines.append(f"title_kz: {a['title_kz']}")
        if a["body_ru"]:
            lines.append(f"text_ru:\n{a['body_ru']}")
        if a["body_kz"]:
            lines.append(f"text_kz:\n{a['body_kz']}")
        elif a.get("has_kz") == "0":
            lines.append(
                "text_kz: [NOT AVAILABLE -- translate Russian legal terms into Kazakh "
                "for the 'kz' keyword array. Do NOT return an empty list.]"
            )
        lines.append("")
        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


def batch_articles(
    articles: list[dict[str, str]],
    max_chars: int = MAX_CHARS_PER_BATCH,
    max_per_batch: int = MAX_ARTICLES_PER_BATCH,
) -> list[list[dict[str, str]]]:
    """Group articles bounded by BOTH max_per_batch (output tokens) AND
    max_chars combined RU+KZ (input tokens)."""
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_chars = 0

    for art in articles:
        art_chars = (len(art["body_ru"]) + len(art["body_kz"])
                     + len(art["id"]) + len(art["title_ru"]) + len(art["title_kz"]) + 120)
        over_count = len(current) >= max_per_batch
        over_chars = current and current_chars + art_chars > max_chars
        if over_count or over_chars:
            batches.append(current)
            current = [art]
            current_chars = art_chars
        else:
            current.append(art)
            current_chars += art_chars

    if current:
        batches.append(current)
    return batches


def _atomic_write(obj: Any, path: Path) -> None:
    """Atomic write (.tmp -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot write {path}: {exc}") from exc


def _load_checkpoint(path: Path) -> dict:
    """Function _load_checkpoint."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"keywords": {}, "relations": []}


def _save_checkpoint(kw_acc: dict, rel_acc: list, path: Path) -> None:
    """Function _save_checkpoint."""
    _atomic_write({"keywords": kw_acc, "relations": rel_acc}, path)


def process_file(
    input_path: Path,
    out_dir: Path,
    extractor: DeepSeekExtractor,
    json_field: str | None,
    max_articles: int | None,
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    dry_run: bool,
) -> dict[str, int]:
    """Process one *_merged.json with incremental checkpointing and auto-split."""
    articles = load_articles(input_path, json_field)
    if max_articles:
        articles = articles[:max_articles]

    no_kz = sum(1 for a in articles if a.get("has_kz") == "0")
    total_chars = sum(len(a["body_ru"]) + len(a["body_kz"]) for a in articles)
    logger.info("  %d article(s) | %d body chars | %d without KZ text",
                len(articles), total_chars, no_kz)

    batches = batch_articles(articles, max_chars=max_chars_per_batch,
                             max_per_batch=max_articles_per_batch)
    logger.info("  %d batch(es) [<= %d arts OR %d chars each]",
                len(batches), max_articles_per_batch, max_chars_per_batch)

    if dry_run:
        for idx, b in enumerate(batches, 1):
            b_chars = sum(len(a["body_ru"]) + len(a["body_kz"]) for a in b)
            est_out = len(b) * 386 + 50
            flag = "(!)" if est_out > 7500 else ""
            logger.info("    Batch %d: %d arts, %d chars in, ~%d tok out %s",
                        idx, len(b), b_chars, est_out, flag)
        if batches:
            print(f"\n{'='*60}\nBATCH 1 PREVIEW:\n{'='*60}")
            print(format_batch(batches[0])[:2000])
        return {"articles": len(articles), "batches": len(batches),
                "keywords": 0, "relations": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "_checkpoint.json"
    ckpt = _load_checkpoint(ckpt_path)
    kw_acc:  dict[str, dict] = ckpt.get("keywords", {})
    rel_acc: list[dict]      = ckpt.get("relations", [])
    completed_ids: set[str]  = set(kw_acc.keys())

    batches_done = 0
    for i, batch in enumerate(batches, 1):
        batch_ids = {a["id"] for a in batch}
        if batch_ids.issubset(completed_ids):
            logger.info("  Batch %d/%d -- SKIP (checkpoint)", i, len(batches))
            batches_done += 1
            continue

        ids_p = list(batch_ids)
        logger.info("  Batch %d/%d -- %d art(s): %s%s",
                    i, len(batches), len(batch),
                    ", ".join(ids_p[:4]), " ..." if len(ids_p) > 4 else "")

        results = _extract_with_autosplit(extractor, batch)

        for result in results:
            for kw in result.keywords:
                if kw.article_id not in kw_acc:
                    kw_acc[kw.article_id] = {
                        "ru": kw.keywords.ru,
                        "kz": kw.keywords.kz,
                    }
                else:
                    logger.warning("  Duplicate article_id: %s (keeping first)", kw.article_id)
            rel_acc.extend([r.model_dump(exclude_none=True) for r in result.relations])

        _save_checkpoint(kw_acc, rel_acc, ckpt_path)
        batches_done += 1
        total_kw = sum(len(r.keywords) for r in results)
        total_rel = sum(len(r.relations) for r in results)
        logger.info("  Batch %d done: +%d kw-sets, +%d relations | checkpoint saved",
                    i, total_kw, total_rel)

    _atomic_write(kw_acc, out_dir / "keywords.json")
    _atomic_write(rel_acc, out_dir / "relations.json")
    ckpt_path.unlink(missing_ok=True)

    return {
        "articles":  len(articles),
        "batches":   batches_done,
        "keywords":  len(kw_acc),
        "relations": len(rel_acc),
    }


def _parse() -> argparse.Namespace:
    """Function _parse."""
    p = argparse.ArgumentParser(
        description=(
            "Extract bilingual keywords & typed relations from legal articles via DeepSeek.\n"
            "Default: all *_merged.json in data/merged -> data/llm_results/<code>/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input-dir", "-d", type=Path, default=Path("data/merged"),
                     help="Directory with *_merged.json files.")
    src.add_argument("--input", "-i", type=Path, default=None,
                     help="Single input file. Overrides --input-dir.")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("data/llm_results"),
                   help="Root output dir. Each code -> <out-dir>/<code>/.")
    p.add_argument("--json-field", default=None,
                   help="JSON body field (default: context_ru -> context_kz -> content).")
    p.add_argument("--model", default=DEEPSEEK_MODEL)
    p.add_argument("--max-articles", type=int, default=None,
                   help="First N articles per file (smoke-test).")
    p.add_argument("--max-articles-per-batch", type=int, default=MAX_ARTICLES_PER_BATCH,
                   help="Max articles per API call (output token guard).")
    p.add_argument("--max-chars-per-batch", type=int, default=MAX_CHARS_PER_BATCH,
                   help="Max combined RU+KZ chars per API call (input token guard).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip files with existing output (default: on).")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Re-process all files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show batch plan with token estimates; no API calls.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    """Function main."""
    args = _parse()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.input is not None:
        if not args.input.exists():
            logger.error("Input file not found: %s", args.input)
            sys.exit(1)
        input_files = [args.input]
    else:
        if not args.input_dir.is_dir():
            logger.error("Input directory not found: %s", args.input_dir)
            sys.exit(1)
        input_files = sorted(args.input_dir.glob("*_merged.json"))
        if not input_files:
            logger.error("No *_merged.json files found in %s", args.input_dir)
            sys.exit(1)

    logger.info("Found %d file(s) | output: %s | batch: <=%d arts OR <=%d chars",
                len(input_files), args.out_dir,
                args.max_articles_per_batch, args.max_chars_per_batch)
    if args.dry_run:
        logger.info("DRY RUN -- no API calls.")

    api_key = ""
    if not args.dry_run:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            logger.error("DEEPSEEK_API_KEY not set.")
            sys.exit(1)

    extractor: DeepSeekExtractor | None = (
        DeepSeekExtractor(api_key=api_key, model=args.model) if not args.dry_run else None
    )

    summary: list[dict] = []

    for file_idx, input_path in enumerate(input_files, 1):
        code         = input_path.stem.replace("_merged", "")
        file_out_dir = args.out_dir / code
        logger.info("\n[%d/%d] %s", file_idx, len(input_files), code.upper())

        kw_path  = file_out_dir / "keywords.json"
        rel_path = file_out_dir / "relations.json"
        if args.resume and not args.dry_run and kw_path.exists() and rel_path.exists():
            n = len(json.loads(kw_path.read_text(encoding="utf-8")))
            logger.info("  SKIP (%d articles done). --no-resume to redo.", n)
            summary.append({"code": code, "status": "skipped", "articles": n})
            continue

        try:
            stats = process_file(
                input_path=input_path, out_dir=file_out_dir,
                extractor=extractor, json_field=args.json_field,
                max_articles=args.max_articles,
                max_chars_per_batch=args.max_chars_per_batch,
                max_articles_per_batch=args.max_articles_per_batch,
                dry_run=args.dry_run,
            )
            summary.append({"code": code, "status": "ok", **stats})
            logger.info("  Done: %d kw-sets, %d relations, %d batches",
                        stats["keywords"], stats["relations"], stats["batches"])
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            summary.append({"code": code, "status": "failed", "error": str(exc)})

    ok      = [s for s in summary if s["status"] == "ok"]
    skipped = [s for s in summary if s["status"] == "skipped"]
    failed  = [s for s in summary if s["status"] == "failed"]
    logger.info("\n%s", "=" * 65)
    logger.info("DONE  processed=%d  skipped=%d  failed=%d",
                len(ok), len(skipped), len(failed))
    if ok:
        logger.info("Total: %d articles | %d kw-sets | %d relations",
                    sum(s.get("articles", 0) for s in ok),
                    sum(s.get("keywords", 0) for s in ok),
                    sum(s.get("relations", 0) for s in ok))
    for s in failed:
        logger.error("  FAILED: %s -- %s", s["code"], s.get("error", ""))
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
