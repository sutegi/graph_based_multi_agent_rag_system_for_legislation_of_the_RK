"""
keyword_extractor.py
====================
Hybrid BM25Plus + KeyBERT keyword extraction for bilingual (RU/KZ) legal texts.
No LLM. Pure statistical + semantic signal.

Quality-first design decisions
--------------------------------
BM25:
  • BM25Plus (delta=0.5) instead of Okapi — avoids zero-score on absent terms,
    handles the long legal articles characteristic of this corpus.
  • Sublinear TF scaling: 1 + log(tf) — dampens term-frequency explosion
    in verbose articles (Civil Code articles can be 5 000+ chars).
  • Title-boost: title tokens count 3× in TF — surface article topics reliably.
  • Two-level IDF:
      local  — per-code IDF (what makes this article unique within Criminal Code?)
      global — corpus-wide IDF (penalises legal boilerplate shared across codes)
    FinalBM25 = 0.6 × local_score + 0.4 × global_score

KeyBERT:
  • paraphrase-multilingual-mpnet-base-v2 by default (better quality than MiniLM;
    ~2× slower, ~2× better precision for legal terminology).
    Fallback to MiniLM-L12-v2 if GPU OOM or user preference.
  • Extract separately from RU text and KZ text, then merge — cross-lingual
    consensus: a term appearing in both languages gets a 1.25× score boost.
  • use_mmr=True + diversity=0.72 — avoids repetitive synonym clusters.
  • ngram_range=(1, 3) — captures both atomic terms and multi-word legal phrases.

Hybrid fusion:
  FinalScore = α × BM25_norm + β × KeyBERT_sim
  Default α=0.30, β=0.70 — semantic signal dominates for legal domain precision.
  Per-article min-max normalisation with IQR-robust outlier clipping.

Bonus post-processing:
  • Legal citation extractor: Статья N / N-бап pulled via regex → always a keyword.
  • Stopword filter: language-aware (RU + KZ sets, NLTK augmentation if installed).

Input  : ./data/merged/*_merged.json          (24 files)
Output : ./data/with_keywords_merged/*.json   (24 files, one per code)

Usage
-----
  python keyword_extractor.py                                  # full run
  python keyword_extractor.py --max-articles 50               # smoke test
  python keyword_extractor.py --model paraphrase-multilingual-MiniLM-L12-v2
  python keyword_extractor.py --alpha 0.4 --beta 0.6
  python keyword_extractor.py --no-fp16                       # force FP32
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from keybert import KeyBERT
    from rank_bm25 import BM25Plus
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError as _err:
    print(
        f"\n[ERROR] Missing dependency: {_err}"
        "\n\nInstall:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "  pip install -r requirements.txt\n",
        file=sys.stdout,
    )
    sys.stdout.flush()
    sys.exit(1)


# ── Logging ───────────────────────────────────────────────────────────────────

class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout, end="\n")
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_TqdmHandler()],
)
logger = logging.getLogger("keyword_extractor")


# ── Stopwords (language-aware) ────────────────────────────────────────────────

_SW_RU: frozenset[str] = frozenset({
    "статья", "пункт", "подпункт", "часть", "раздел", "глава", "параграф",
    "закон", "кодекс", "настоящий", "настоящего", "настоящем", "настоящей",
    "настоящее", "данный", "указанный", "соответствии", "порядке",
    "установленном", "установленного", "предусмотренных", "предусмотренном",
    "республика", "казахстан", "республики", "казахстана",
    "который", "которая", "которые", "которых", "которому", "которой", "которым",
    "после", "дней", "лет", "года", "году", "также", "либо", "если",
    "должен", "должна", "должно", "должны", "является", "являются",
    "будет", "быть", "было", "были", "этого", "этом", "этот", "эта",
    "один", "одно", "одна", "может", "могут", "иной", "иное", "иных",
    "своих", "своим", "вводится", "действие", "действия", "опубликования",
    "официального", "принятого", "сноска", "изменениями", "внесенными",
    "законами", "законом", "редакции", "введение", "настоящим",
    "но", "или", "что", "как", "при", "до", "со", "из", "для", "его",
    "они", "она", "оно", "мы", "вы", "он", "ей", "им", "их", "нет",
    "более", "менее", "такой", "такая", "такие", "такого", "такому",
    "этой", "этому", "тот", "той", "том", "того", "тем", "тех",
})

_SW_KZ: frozenset[str] = frozenset({
    "бап", "бабы", "бапқа", "баптың", "баптар", "баптарда", "баптарының",
    "тармақ", "тармағы", "тармақшасы", "тармақтарда", "тармақтарының",
    "ескерту", "кодексі", "кодексінің", "кодексіне", "кодексінде",
    "республикасы", "қазақстан", "қазақстанның", "қазақстанда",
    "осы", "болып", "арқылы", "үшін", "және", "сәйкес", "бойынша",
    "туралы", "заңымен", "заңы", "заңның", "заңда", "заңдарымен",
    "енгізіледі", "енгізілген", "жылы", "жылдан", "бастап",
    "дейін", "ретінде", "болған", "белгіленген", "көзделген",
    "мен", "мына", "осыған", "оның", "олар", "оларға", "оларда", "олардың",
    "бұл", "бұған", "егер", "немесе", "де", "да", "кейін",
    "алдында", "жоқ", "бар", "болады", "болуы", "болса",
    "берілген", "жүргізілетін", "жасалатын", "қолданылатын",
})

_SW_ALL: frozenset[str] = _SW_RU | _SW_KZ

# Optional NLTK augmentation
try:
    import nltk  # type: ignore
    try:
        from nltk.corpus import stopwords as _nltk_sw  # type: ignore
        _SW_RU = _SW_RU | frozenset(_nltk_sw.words("russian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as _nltk_sw  # type: ignore
        _SW_RU = _SW_RU | frozenset(_nltk_sw.words("russian"))
except Exception:
    pass

_KZ_CHARS: frozenset[str] = frozenset("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
_TOKEN_RE = re.compile(r"[а-яёА-ЯЁa-zA-ZәіңғүұқөһӘІҢҒҮҰҚӨҺ]{3,}")


def _lang(text: str) -> str:
    if not text:
        return "ru"
    kz = sum(1 for c in text if c in _KZ_CHARS)
    return "kz" if kz / max(len(text), 1) > 0.01 else "ru"


def _tokenize(text: str, lang: str = "auto") -> list[str]:
    if lang == "auto":
        lang = _lang(text)
    sw = _SW_KZ if lang == "kz" else _SW_RU
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in sw]


# ── Legal citation extractor ──────────────────────────────────────────────────

_CITE_RU = re.compile(
    r"стать[яеию]\s+([\d]+(?:-[\d]+)*)",
    re.IGNORECASE | re.UNICODE,
)
_CITE_KZ = re.compile(
    r"([\d]+(?:-[\d]+)*)-(?:бап|баптың|бабы|бабының)",
    re.IGNORECASE | re.UNICODE,
)


def _extract_citations(text: str) -> list[str]:
    """Return normalised citation strings: ['статья 5', '14-бап', ...]"""
    results: list[str] = []
    for m in _CITE_RU.finditer(text):
        results.append(f"статья {m.group(1)}")
    for m in _CITE_KZ.finditer(text):
        results.append(f"{m.group(1)}-бап")
    return results


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    input_dir:  str = "data/merged"
    output_dir: str = "data/with_keywords_merged"

    # BM25Plus
    bm25_k1:          float = 1.5    # term saturation (1.2–2.0 typical)
    bm25_b:           float = 0.75   # length normalisation
    bm25_delta:       float = 0.5    # BM25+ smoothing — the key difference from Okapi
    bm25_min_len:     int   = 3      # minimum token length kept
    bm25_top_n:       int   = 30     # candidates from BM25 before fusion
    local_weight:     float = 0.6    # weight of per-code IDF vs global IDF
    global_weight:    float = 0.4

    # KeyBERT
    kb_model:     str   = "paraphrase-multilingual-mpnet-base-v2"
    kb_top_n:     int   = 20         # candidates from KeyBERT before fusion
    kb_ngram:     tuple = (1, 3)
    kb_diversity: float = 0.72       # MMR diversity (0=greedy, 1=max-diverse)
    kb_batch:     int   = 32         # articles per GPU batch
    kb_max_chars: int   = 2000       # truncation for KeyBERT (covers most articles)
    use_fp16:     bool  = True

    # Hybrid fusion
    alpha:    float = 0.30   # BM25 weight
    beta:     float = 0.70   # KeyBERT weight
    final_n:  int   = 25     # keywords kept per article after fusion
    cross_lingual_boost: float = 1.25  # boost when term found in both RU and KZ

    # Runtime
    max_articles: int = 0    # 0 = all


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class Article:
    article_id: str
    number:     str
    title_ru:   str
    title_kz:   str
    hierarchy:  str | None
    context_ru: str
    context_kz: str
    code_name:  str
    _combined:  str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._combined = " ".join(p for p in (self.context_ru, self.context_kz) if p)

    @property
    def combined_text(self) -> str:
        return self._combined


def load_file(path: Path) -> tuple[str, list[Article]]:
    """Load one *_merged.json → (code_name, [Article,...])"""
    code_name = path.stem.replace("_merged", "")
    records = json.loads(path.read_text(encoding="utf-8"))
    articles = [
        Article(
            article_id = r.get("article_id") or "",
            number     = str(r.get("number") or ""),
            title_ru   = r.get("title_ru") or "",
            title_kz   = r.get("title_kz") or "",
            hierarchy  = r.get("hierarchy"),
            context_ru = r.get("context_ru") or "",
            context_kz = r.get("context_kz") or "",
            code_name  = code_name,
        )
        for r in records
    ]
    return code_name, articles


# ── BM25Plus with sublinear TF + title boost ──────────────────────────────────

class BM25Index:
    """
    BM25Plus index with:
      - sublinear TF scaling: tf_scaled = 1 + log(1 + raw_tf)
      - title boost: title tokens appear 3× in TF computation
      - per-article scored dict via smoothed TF-IDF
    """

    def __init__(
        self,
        articles: list[Article],
        cfg: Config,
        label: str = "",
    ) -> None:
        self._cfg   = cfg
        self._n     = len(articles)
        label_str   = f"[{label}] " if label else ""

        # Build boosted token lists
        tokenized: list[list[str]] = []
        for art in articles:
            lang   = _lang(art.context_ru or art.context_kz or "")
            body   = _tokenize(art.combined_text, lang)
            title  = _tokenize(art.title_ru + " " + art.title_kz, lang)
            # Title tokens appear 3× to boost their TF
            tokenized.append(body + title * 3)

        self._bm25      = BM25Plus(tokenized, k1=cfg.bm25_k1, b=cfg.bm25_b, delta=cfg.bm25_delta)
        self._tokenized = tokenized

        # Smoothed TF-IDF per article
        df: dict[str, int] = defaultdict(int)
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1

        self._tfidf: list[dict[str, float]] = []
        for toks in tokenized:
            doc_len = max(len(toks), 1)
            raw_tf: dict[str, int] = defaultdict(int)
            for t in toks:
                raw_tf[t] += 1
            scores: dict[str, float] = {}
            for term, cnt in raw_tf.items():
                tf  = 1.0 + math.log(1.0 + cnt / doc_len)   # sublinear
                idf = math.log((self._n + 1) / (df[term] + 1)) + 1.0  # smoothed
                scores[term] = tf * idf
            self._tfidf.append(scores)

        vocab_size = len({t for toks in tokenized for t in toks})
        logger.info("  %sBM25Plus ready | docs=%d | vocab=%d", label_str, self._n, vocab_size)

    def scored(self, idx: int) -> dict[str, float]:
        """Return {term: tfidf_score} for article *idx*, min_len filtered."""
        min_len = self._cfg.bm25_min_len
        return {t: s for t, s in self._tfidf[idx].items() if len(t) >= min_len}


# ── KeyBERT stage ─────────────────────────────────────────────────────────────

class KeyBERTStage:
    """
    Batched KeyBERT with:
      - separate extraction for RU and KZ text
      - cross-lingual consensus boost (terms in both → +25%)
    """

    def __init__(self, cfg: Config) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading KeyBERT model '%s' on %s …", cfg.kb_model, device)
        try:
            st = SentenceTransformer(cfg.kb_model, device=device)
            if cfg.use_fp16 and device == "cuda":
                st = st.half()
            self._kb = KeyBERT(model=st)
            logger.info("  KeyBERT ready (device=%s, fp16=%s).", device, cfg.use_fp16 and device == "cuda")
        except Exception as exc:
            fallback = "paraphrase-multilingual-MiniLM-L12-v2"
            logger.warning("Model load failed (%s). Falling back to %s.", exc, fallback)
            st = SentenceTransformer(fallback, device=device)
            self._kb = KeyBERT(model=st)
        self._cfg    = cfg
        self._device = device

    def _extract(self, texts: list[str]) -> list[list[tuple[str, float]]]:
        """Extract keyphrases from a batch of texts."""
        cfg     = self._cfg
        results: list[list[tuple[str, float]]] = []
        batch   = cfg.kb_batch

        for start in range(0, len(texts), batch):
            chunk = [t[:cfg.kb_max_chars] for t in texts[start : start + batch]]
            raw = self._kb.extract_keywords(
                chunk,
                keyphrase_ngram_range = cfg.kb_ngram,
                top_n      = cfg.kb_top_n,
                use_mmr    = True,
                diversity  = cfg.kb_diversity,
            )
            # Single string → list[tuple]; list → list[list[tuple]]
            if chunk and not isinstance(raw[0], list):
                raw = [raw]
            results.extend(raw)
        return results

    def extract_bilingual(
        self, articles: list[Article]
    ) -> list[dict[str, list[tuple[str, float]]]]:
        """
        For each article return:
          {"ru": [(phrase, score),...], "kz": [...], "merged": [...]}
        merged applies cross-lingual boost.
        """
        ru_texts = [art.context_ru[:self._cfg.kb_max_chars] for art in articles]
        kz_texts = [art.context_kz[:self._cfg.kb_max_chars] for art in articles]

        logger.info("  KeyBERT pass 1/2 — RU texts …")
        ru_res = self._extract(ru_texts)
        logger.info("  KeyBERT pass 2/2 — KZ texts …")
        kz_res = self._extract(kz_texts)

        boost = self._cfg.cross_lingual_boost
        combined: list[dict[str, list[tuple[str, float]]]] = []

        for ru_kws, kz_kws in zip(ru_res, kz_res):
            # Build score maps
            ru_map: dict[str, float] = {kw.lower(): s for kw, s in ru_kws}
            kz_map: dict[str, float] = {kw.lower(): s for kw, s in kz_kws}

            # Union with cross-lingual boost for shared terms/cognates
            all_terms = set(ru_map) | set(kz_map)
            merged_map: dict[str, float] = {}
            for term in all_terms:
                r = ru_map.get(term, 0.0)
                k = kz_map.get(term, 0.0)
                score = max(r, k)
                if r > 0 and k > 0:
                    score = min(score * boost, 1.0)
                merged_map[term] = score

            merged_sorted = sorted(merged_map.items(), key=lambda x: x[1], reverse=True)
            combined.append({
                "ru":     ru_kws,
                "kz":     kz_kws,
                "merged": merged_sorted,
            })

        if self._device == "cuda":
            torch.cuda.empty_cache()
        return combined


# ── Hybrid scorer ─────────────────────────────────────────────────────────────

class HybridScorer:
    """
    FinalScore = α × BM25_norm + β × KeyBERT_sim

    BM25_norm uses IQR-robust clipping before min-max to handle outliers:
      values above Q75 + 1.5*IQR are clipped → prevents one dominant term
      from collapsing all others to near-zero.

    Two-level BM25: FinalBM25 = local_w × local_score + global_w × global_score
    """

    def __init__(self, cfg: Config) -> None:
        if abs(cfg.alpha + cfg.beta - 1.0) > 1e-6:
            raise ValueError(f"alpha + beta must equal 1.0, got {cfg.alpha + cfg.beta}")
        self._cfg = cfg

    @staticmethod
    def _robust_norm(scores: dict[str, float]) -> dict[str, float]:
        """IQR-clipped min-max normalisation."""
        if not scores:
            return {}
        vals = np.array(list(scores.values()), dtype=float)
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr  = q75 - q25
        clip = q75 + 1.5 * iqr if iqr > 0 else vals.max()
        lo, hi = vals.min(), min(vals.max(), clip)
        span = hi - lo or 1.0
        return {t: float(min(s, clip) - lo) / span for t, s in scores.items()}

    def fuse(
        self,
        local_bm25:  dict[str, float],    # {term: tfidf} within-code
        global_bm25: dict[str, float],    # {term: tfidf} corpus-wide
        kb_merged:   list[tuple[str, float]],  # [(phrase, sim)]
        stopwords:   frozenset[str],
        citations:   list[str],
    ) -> list[tuple[str, float]]:
        cfg = self._cfg

        # Combine two BM25 levels
        all_terms = set(local_bm25) | set(global_bm25)
        combined_bm25 = {
            t: cfg.local_weight  * local_bm25.get(t, 0.0)
             + cfg.global_weight * global_bm25.get(t, 0.0)
            for t in all_terms
        }
        bm25_norm = self._robust_norm(combined_bm25)

        # KeyBERT: already in [0, 1], expand multi-word phrases
        kb_map: dict[str, float] = {}
        for phrase, sim in kb_merged:
            p = phrase.lower().strip()
            if p and p not in stopwords:
                kb_map[p] = max(kb_map.get(p, 0.0), float(sim))
                # Index individual words at 80% score so they can combine with BM25
                for word in p.split():
                    if len(word) >= 3 and word not in stopwords:
                        kb_map[word] = max(kb_map.get(word, 0.0), float(sim) * 0.80)

        union = set(bm25_norm) | set(kb_map)
        scored: list[tuple[str, float]] = []
        for term in union:
            if term in stopwords or len(term) < 3:
                continue
            b = bm25_norm.get(term, 0.0)
            k = kb_map.get(term, 0.0)
            final = cfg.alpha * b + cfg.beta * k
            if final > 0.0:
                scored.append((term, round(final, 6)))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = scored[: cfg.final_n]

        # Prepend citation keywords with max score (always preserve)
        seen = {t for t, _ in result}
        for cite in citations:
            c = cite.lower().strip()
            if c and c not in seen:
                result.insert(0, (c, 1.0))
                seen.add(c)

        return result


# ── Output serialisation ──────────────────────────────────────────────────────

def _to_record(art: Article, scored: list[tuple[str, float]],
               bm25_top: list[str], kb_top: list[str]) -> dict[str, Any]:
    return {
        "article_id":       art.article_id,
        "number":           art.number,
        "title_ru":         art.title_ru,
        "title_kz":         art.title_kz,
        "hierarchy":        art.hierarchy,
        "context_ru":       art.context_ru,
        "context_kz":       art.context_kz,
        "code_name":        art.code_name,
        "keywords":         [t for t, _ in scored],
        "keywords_scored":  [[t, s] for t, s in scored],
        "keywords_bm25":    bm25_top,
        "keywords_keybert": kb_top,
    }


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot save {path}: {exc}") from exc


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    t0        = time.perf_counter()
    input_dir = Path(cfg.input_dir)
    out_dir   = Path(cfg.output_dir)

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    files = sorted(input_dir.glob("*_merged.json"))
    if not files:
        logger.error("No *_merged.json files in %s", input_dir)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Input : %s (%d files)", input_dir, len(files))
    logger.info("Output: %s", out_dir)

    # ── Load ALL articles for global BM25 index ───────────────────────────────
    logger.info("Loading all articles for global BM25 index …")
    all_articles: list[Article] = []
    file_slices:  list[tuple[str, int, int]] = []   # (code, start, end)

    for f in files:
        code_name, arts = load_file(f)
        if cfg.max_articles and len(all_articles) >= cfg.max_articles:
            break
        start = len(all_articles)
        if cfg.max_articles:
            arts = arts[: cfg.max_articles - start]
        all_articles.extend(arts)
        file_slices.append((code_name, start, len(all_articles)))

    logger.info("Loaded %d articles total.", len(all_articles))

    # ── Global BM25Plus index ─────────────────────────────────────────────────
    logger.info("Building global BM25Plus index …")
    global_idx = BM25Index(all_articles, cfg, label="global")

    # ── KeyBERT (single GPU pass over all articles) ───────────────────────────
    kb_stage = KeyBERTStage(cfg)
    logger.info("Running bilingual KeyBERT extraction (%d articles) …", len(all_articles))
    kb_results = kb_stage.extract_bilingual(all_articles)

    # ── Per-file processing ───────────────────────────────────────────────────
    scorer = HybridScorer(cfg)
    total_keywords = 0
    failed = 0

    for code_name, start, end in tqdm(file_slices, desc="Files", unit="file"):
        arts    = all_articles[start:end]
        n_local = len(arts)

        # Local BM25Plus index (within-code IDF — discriminative keywords)
        local_idx = BM25Index(arts, cfg, label=code_name)

        records: list[dict[str, Any]] = []

        for local_i, (art, kb_res) in enumerate(
            zip(arts, kb_results[start:end])
        ):
            global_i = start + local_i
            lang     = _lang(art.context_ru or art.context_kz or "")
            sw       = _SW_KZ if lang == "kz" else _SW_RU

            local_bm25  = local_idx.scored(local_i)
            global_bm25 = global_idx.scored(global_i)
            kb_merged   = kb_res["merged"]

            # Legal citation extraction (Статья N, N-бап)
            citations = _extract_citations(art.context_ru + " " + art.context_kz)

            # Top-N from BM25 for debug field (sorted by local score)
            bm25_top = sorted(local_bm25, key=local_bm25.get, reverse=True)[: cfg.bm25_top_n]

            # Top-N from KeyBERT for debug field
            kb_top = [kw for kw, _ in kb_merged[: cfg.kb_top_n]]

            try:
                scored = scorer.fuse(local_bm25, global_bm25, kb_merged, sw, citations)
                records.append(_to_record(art, scored, bm25_top, kb_top))
                total_keywords += len(scored)
            except Exception as exc:
                logger.warning("[%s] Article %s failed: %s", code_name, art.number, exc)
                records.append(_to_record(art, [], bm25_top, kb_top))
                failed += 1

        out_path = out_dir / f"{code_name}_keywords.json"
        try:
            _save(records, out_path)
            logger.info(
                "  %-25s → %4d articles | avg %.1f keywords/article",
                code_name, n_local,
                sum(len(r["keywords"]) for r in records) / max(n_local, 1),
            )
        except Exception as exc:
            logger.error("Cannot save %s: %s", out_path, exc)
            failed += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "\nDone in %.1f sec | %d articles | %d total keywords | avg %.1f/art | %d failed",
        elapsed, len(all_articles), total_keywords,
        total_keywords / max(len(all_articles), 1), failed,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid BM25Plus + KeyBERT keyword extractor for bilingual legal texts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", default="data/merged")
    p.add_argument("--output", "-o", default="data/with_keywords_merged")
    p.add_argument("--model",  "-m",
                   default="paraphrase-multilingual-mpnet-base-v2",
                   help="SentenceTransformer model for KeyBERT.")
    p.add_argument("--alpha",  type=float, default=0.30, help="BM25 weight (α).")
    p.add_argument("--beta",   type=float, default=0.70, help="KeyBERT weight (β).")
    p.add_argument("--final-n",    type=int,   default=25,   help="Keywords per article.")
    p.add_argument("--kb-batch",   type=int,   default=32,   help="KeyBERT batch size.")
    p.add_argument("--max-articles", type=int, default=0,    help="Limit (0=all).")
    p.add_argument("--no-fp16",    action="store_true",      help="Disable FP16.")
    p.add_argument("--diversity",  type=float, default=0.72, help="MMR diversity.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if abs(args.alpha + args.beta - 1.0) > 1e-4:
        print(f"[ERROR] --alpha + --beta must equal 1.0 (got {args.alpha + args.beta})")
        sys.exit(1)
    run(Config(
        input_dir    = args.input,
        output_dir   = args.output,
        kb_model     = args.model,
        alpha        = args.alpha,
        beta         = args.beta,
        final_n      = args.final_n,
        kb_batch     = args.kb_batch,
        max_articles = args.max_articles,
        use_fp16     = not args.no_fp16,
        kb_diversity = args.diversity,
    ))
