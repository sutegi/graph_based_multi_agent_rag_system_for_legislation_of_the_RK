"""merge_bilingual.py
==================
Aligns and merges paired Russian/Kazakh legal JSON files from parser.py output."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("merge_bilingual")


BILINGUAL_FIELDS: list[str] = [
    "title",
    "section",
    "chapter",
    "paragraph",
    "hierarchy",
    "context",
]


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Function _load_json."""
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def _save_json(obj: list[dict[str, Any]], path: Path) -> None:
    """Atomic write: .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot save {path}: {exc}") from exc


def _find_pairs(input_dir: Path) -> list[tuple[str, Path, Path]]:
    """Scan input_dir for *_ru.json / *_kz.json pairs."""
    ru_files: dict[str, Path] = {}
    kz_files: dict[str, Path] = {}

    for p in sorted(input_dir.glob("*.json")):
        stem = p.stem
        if stem.endswith("_ru"):
            ru_files[stem[:-3]] = p
        elif stem.endswith("_kz"):
            kz_files[stem[:-3]] = p

    pairs: list[tuple[str, Path, Path]] = []
    for prefix, ru_path in ru_files.items():
        kz_path = kz_files.get(prefix)
        if kz_path is None:
            logger.warning("No KZ counterpart for %s -- skipped.", ru_path.name)
            continue
        pairs.append((prefix, ru_path, kz_path))

    for prefix in kz_files:
        if prefix not in ru_files:
            logger.warning("No RU counterpart for %s_kz.json -- skipped.", prefix)

    return pairs


def _build_index(
    records: list[dict[str, Any]],
    key_field: str,
    lang: str,
) -> dict[str, dict[str, Any]]:
    """Build {key_value: record} map for O(1) lookup."""
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        k = rec.get(key_field)
        if k is None:
            logger.debug("  [%s] Record missing key field '%s' -- skipped.", lang, key_field)
            continue
        k = str(k)
        if k in index:
            logger.debug("  [%s] Duplicate key %r -- later record overwrites.", lang, k)
        index[k] = rec
    return index


def _extract_lang(
    rec: dict[str, Any] | None,
    field: str,
) -> Any:
    """Return field value from rec, or None if rec is missing."""
    return rec.get(field) if rec is not None else None


def _build_bilingual_fields(
    ru_rec: dict[str, Any] | None,
    kz_rec: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build all bilingual field objects from RU and KZ records."""
    return {
        field: {
            "ru": _extract_lang(ru_rec, field),
            "kz": _extract_lang(kz_rec, field),
        }
        for field in BILINGUAL_FIELDS
    }


def _merge_pair(
    prefix: str,
    ru_path: Path,
    kz_path: Path,
    key_field: str,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Align RU and KZ records by key_field and produce fully bilingual merged list."""
    ru_records = _load_json(ru_path)
    kz_records = _load_json(kz_path)

    ru_index = _build_index(ru_records, key_field, "ru")
    kz_index = _build_index(kz_records, key_field, "kz")

    merged: list[dict[str, Any]] = []
    seen_kz_keys: set[str] = set()
    matched = ru_only = 0

    for ru_rec in ru_records:
        k = str(ru_rec.get(key_field, ""))
        kz_rec = kz_index.get(k) if k else None

        if kz_rec is not None:
            matched += 1
            seen_kz_keys.add(k)
            source = "both"
        else:
            ru_only += 1
            source = "ru_only"
            if verbose:
                logger.debug("  [%s] RU-only article: number=%r", prefix, k)

        record: dict[str, Any] = {
            "article_id": ru_rec.get("id") or (kz_rec.get("id") if kz_rec else None),
            "number":     ru_rec.get(key_field),
            "source":     source,
        }
        record.update(_build_bilingual_fields(ru_rec, kz_rec))
        merged.append(record)

    kz_only = 0
    for k, kz_rec in kz_index.items():
        if k in seen_kz_keys:
            continue
        kz_only += 1
        if verbose:
            logger.debug("  [%s] KZ-only article: number=%r", prefix, k)

        record = {
            "article_id": kz_rec.get("id"),
            "number":     kz_rec.get(key_field),
            "source":     "kz_only",
        }
        record.update(_build_bilingual_fields(None, kz_rec))
        merged.append(record)

    logger.info(
        "  %-30s  matched=%d  ru_only=%d  kz_only=%d  total=%d",
        prefix, matched, ru_only, kz_only, len(merged),
    )
    return merged


def run(
    input_dir: Path,
    output_dir: Path,
    key_field: str = "number",
    verbose: bool = False,
) -> None:
    """Function run."""
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_pairs(input_dir)
    if not pairs:
        logger.error("No matched RU/KZ pairs found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d pair(s) in %s", len(pairs), input_dir)
    total_articles = 0
    failed = 0

    for prefix, ru_path, kz_path in pairs:
        try:
            merged = _merge_pair(prefix, ru_path, kz_path,
                                 key_field=key_field, verbose=verbose)
            out_path = output_dir / f"{prefix}_merged.json"
            _save_json(merged, out_path)
            total_articles += len(merged)
        except Exception as exc:
            logger.error("FAILED [%s]: %s", prefix, exc)
            failed += 1

    logger.info(
        "Done. %d pair(s) merged | %d article(s) total | %d failed",
        len(pairs) - failed, total_articles, failed,
    )


def parse_args() -> argparse.Namespace:
    """Function parse_args."""
    p = argparse.ArgumentParser(
        description="Align and merge RU/KZ legal JSON pairs (all fields, bilingual).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   "-i", type=Path, default=Path("data/structured"),
                   help="Directory containing *_ru.json and *_kz.json files.")
    p.add_argument("--output",  "-o", type=Path, default=Path("data/merged"),
                   help="Output directory for merged JSON files.")
    p.add_argument("--key",     "-k", default="number",
                   help="Field used as the alignment key (must be unique per file).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Log each RU-only / KZ-only article during merge.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dir  = args.input,
        output_dir = args.output,
        key_field  = args.key,
        verbose    = args.verbose,
    )
