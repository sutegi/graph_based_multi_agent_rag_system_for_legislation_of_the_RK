#!/usr/bin/env python3
"""
evaluation_metrics.py
=====================
Comprehensive evaluation suite for the Multi-Agent Graph-Based
Retrieval-Augmented Generation System applied to the Legislation
of the Republic of Kazakhstan.

Diploma: "Development of a Multi-Agent Graph-Based Retrieval-Augmented
          Generation System Applied to the Legislation of the Republic
          of Kazakhstan"

Metrics computed
----------------
Retrieval  — Precision@K, Recall@K, F1@K, MRR, NDCG@K, Hit@K
Generation — ROUGE-1, ROUGE-2, ROUGE-L, Semantic Similarity (SBERT cosine)
LLM-Judge  — Faithfulness (0-1), Answer Relevance (0-1) via DeepSeek
System     — Latency (mean / median / P95 / P99), Confidence distribution,
             Retry rate, Citation rate, Off-topic detection accuracy
Intent     — Codex routing Precision / Recall / F1, Mean keyword count
Graph      — Graph-enrichment ratio, graph-sourced article fraction

Usage
-----
    cd diploma_code
    python evaluation_metrics.py                         # full run
    python evaluation_metrics.py --no-llm-judge          # skip LLM judge (faster)
    python evaluation_metrics.py --k 5 10                # custom K values
    python evaluation_metrics.py --out results/eval.json # custom output path
    python evaluation_metrics.py --cases labor_01 tax_01 # run specific cases only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Rich UI ──────────────────────────────────────────────────────────────────
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

# ── Project path setup ───────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from multi_agent_rag import Session, close_driver, run  # noqa: E402
from multi_agent_rag.config import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LEGAL_CODEXES,
)
from multi_agent_rag.database import ping  # noqa: E402
from multi_agent_rag.pipeline import PipelineResult  # noqa: E402

# ── Optional sentence-transformers for semantic similarity ────────────────────
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

# ── Optional NLTK for ROUGE / BLEU ───────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# =============================================================================
# 1.  GROUND-TRUTH TEST DATASET
# =============================================================================

@dataclass
class EvalCase:
    """Single evaluation test case with ground-truth annotations."""
    id: str
    question: str
    language: str                        # "ru" | "kz"
    query_type: str                      # factual | procedural | definitional | cross_codex | off_topic
    expected_codexes: list[str]          # codex slugs the system should route to
    expected_article_numbers: list[str]  # article numbers that should appear in retrieved context
    reference_answer: str                # gold-standard short answer (for ROUGE / similarity)
    is_legal: bool = True                # False → off-topic, pipeline must detect this


GROUND_TRUTH: list[EvalCase] = [
    # ── Labor Law ────────────────────────────────────────────────────────────
    EvalCase(
        id="labor_01",
        question="Можно ли уволить беременную женщину по инициативе работодателя?",
        language="ru",
        query_type="factual",
        expected_codexes=["labor"],
        expected_article_numbers=["44", "52", "53", "54"],  # art.44 = временный перевод беременных женщин
        reference_answer=(
            "Расторжение трудового договора по инициативе работодателя с беременными "
            "женщинами не допускается, за исключением случаев ликвидации работодателя."
        ),
    ),
    EvalCase(
        id="labor_02",
        question="Сколько календарных дней оплачиваемого отпуска предоставляется работнику в год?",
        language="ru",
        query_type="factual",
        expected_codexes=["labor"],
        expected_article_numbers=["88", "89"],
        reference_answer=(
            "Работникам предоставляется оплачиваемый ежегодный трудовой отпуск "
            "продолжительностью не менее 24 календарных дней."
        ),
    ),
    EvalCase(
        id="labor_03",
        question="Каков максимальный срок испытания при приёме на работу по трудовому договору?",
        language="ru",
        query_type="factual",
        expected_codexes=["labor"],
        expected_article_numbers=["36", "37"],
        reference_answer=(
            "Испытательный срок при заключении трудового договора не может "
            "превышать трёх месяцев."
        ),
    ),
    EvalCase(
        id="labor_04",
        question="Ереуілге қатысушы жұмыскерлердің ереуіл уақыты ішіндегі жалақысы сақтала ма?",
        language="kz",
        query_type="procedural",
        expected_codexes=["labor"],
        expected_article_numbers=["175", "176"],
        reference_answer=(
            "Ереуіл жалақының төленбеуіне немесе уақтылы төленбеуіне байланысты өткізілгеннен басқа жағдайларда, "
            "жұмыскерлердің ереуіл уақыты ішіндегі жалақысы сақталмайды."
        ),
    ),
    # ── Criminal Law ─────────────────────────────────────────────────────────
    EvalCase(
        id="criminal_01",
        question="Что такое кража и какое наказание предусмотрено за нее?",
        language="ru",
        query_type="definitional",
        expected_codexes=["criminal"],  
        expected_article_numbers=["188"],
        reference_answer=(
            "Кража — тайное хищение чужого имущества. По статье 188 УК РК "
            "наказывается штрафом либо лишением свободы в зависимости от "
            "размера ущерба и квалифицирующих признаков."
        ),
    ),
    EvalCase(
        id="criminal_02",
        question="Каковы основания для освобождения от уголовной ответственности в связи с примирением сторон?",
        language="ru",
        query_type="procedural",
        expected_codexes=["criminal"],
        expected_article_numbers=["68"],
        reference_answer=(
            "Лицо освобождается от уголовной ответственности, если загладило "
            "причинённый вред и примирилось с потерпевшим по делам небольшой "
            "или средней тяжести."
        ),
    ),
    # ── Civil Law ────────────────────────────────────────────────────────────
    EvalCase(
        id="civil_01",
        question="Каков общий срок исковой давности по гражданским делам в Казахстане?",
        language="ru",
        query_type="factual",
        expected_codexes=["civil_general"],
        expected_article_numbers=["177", "178"],
        reference_answer=(
            "Общий срок исковой давности составляет три года со дня, "
            "когда лицо узнало или должно было узнать о нарушении своего права."
        ),
    ),
    EvalCase(
        id="civil_02",
        question="Что такое юридическое лицо и с какого момента оно считается созданным?",
        language="ru",
        query_type="definitional",
        expected_codexes=["civil_general"],
        expected_article_numbers=["33", "35", "42"],  # art.35 = правоспособность юрлица
        reference_answer=(
            "Юридическое лицо — организация с обособленным имуществом, "
            "способная от своего имени приобретать права и нести обязанности. "
            "Считается созданным с момента государственной регистрации."
        ),
    ),
    # ── Family Law ───────────────────────────────────────────────────────────
    EvalCase(
        id="family_01",
        question="Каков минимальный брачный возраст для заключения брака в Казахстане?",
        language="ru",
        query_type="factual",
        expected_codexes=["family"],
        expected_article_numbers=["10"],
        reference_answer=(
            "Брачный возраст устанавливается в 18 лет. При исключительных "
            "обстоятельствах возможно снижение до 16 лет по решению суда."
        ),
    ),
    EvalCase(
        id="family_02",
        question="Неке жасы қанша және оны қалай кемейтуге болады?",
        language="kz",
        query_type="factual",
        expected_codexes=["family"],
        expected_article_numbers=["10"],
        reference_answer=(
            "Неке жасы 18 жасқа белгіленген. Ерекше жағдайларда сот шешімімен "
            "16 жасқа дейін төмендетілуі мүмкін."
        ),
    ),
    EvalCase(
        id="family_03",
        question="Как происходит раздел совместно нажитого имущества супругов при разводе?",
        language="ru",
        query_type="procedural",
        expected_codexes=["family"],
        expected_article_numbers=["36", "37", "38"],
        reference_answer=(
            "Имущество, нажитое в браке, является совместной собственностью "
            "и при разделе делится поровну, если брачным договором не "
            "предусмотрено иное."
        ),
    ),
    # ── Tax Law ──────────────────────────────────────────────────────────────
    EvalCase(
        id="tax_01",
        question="Какова ставка налога на добавленную стоимость (НДС) в Казахстане?",
        language="ru",
        query_type="factual",
        expected_codexes=["tax_code", "tax_payments"],
        expected_article_numbers=["422"],
        reference_answer=(
            "Ставка налога на добавленную стоимость составляет 12 процентов."
        ),
    ),
    EvalCase(
        id="tax_02",
        question="Кто является плательщиком индивидуального подоходного налога?",
        language="ru",
        query_type="definitional",
        expected_codexes=["tax_code", "tax_payments"],
        expected_article_numbers=["316"],
        reference_answer=(
            "Плательщики ИПН — физические лица, получающие доходы, облагаемые "
            "у источника выплаты, а также доходы, не облагаемые у источника."
        ),
    ),
    # ── Administrative Offenses ──────────────────────────────────────────────
    EvalCase(
        id="admin_01",
        question="Что такое административное задержание и каков его максимальный срок?",
        language="ru",
        query_type="definitional",
        expected_codexes=["admin_offenses"],
        expected_article_numbers=["787", "788", "789"],
        reference_answer=(
            "Административное задержание — кратковременное ограничение свободы "
            "физического лица. Общий срок не должен превышать трёх часов."
        ),
    ),
    # ── Environmental Law ────────────────────────────────────────────────────
    EvalCase(
        id="env_01",
        question="Какие права имеют граждане в области охраны окружающей среды?",
        language="ru",
        query_type="definitional",
        expected_codexes=["environmental"],
        expected_article_numbers=["13", "14"],
        reference_answer=(
            "Граждане имеют право на благоприятную окружающую среду, "
            "достоверную экологическую информацию и возмещение ущерба, "
            "причинённого здоровью экологическим правонарушением."
        ),
    ),
    # ── Land Law ─────────────────────────────────────────────────────────────
    EvalCase(
        id="land_01",
        question="Кто может быть собственником земли в Казахстане?",
        language="ru",
        query_type="factual",
        expected_codexes=["land"],
        expected_article_numbers=["22", "23", "24"],  # art.22 = возникновение права собственности
        reference_answer=(
            "Земля может находиться в частной собственности граждан и "
            "негосударственных юридических лиц РК. Иностранные граждане "
            "вправе обладать участками лишь на праве землепользования."
        ),
    ),
    # ── Cross-codex ──────────────────────────────────────────────────────────
    EvalCase(
        id="cross_01",
        question=(
            "Какие последствия наступают для работодателя при незаконном "
            "увольнении работника — трудовые и административные?"
        ),
        language="ru",
        query_type="cross_codex",
        expected_codexes=["labor", "admin_offenses"],
        # art.160=сроки обращения по трудовым спорам, art.161=восстановление на работе
        # (was wrongly set to 175,176 which are about strikes, not illegal dismissal)
        expected_article_numbers=["160", "161"],
        reference_answer=(
            "Работодатель обязан восстановить незаконно уволенного работника "
            "и выплатить средний заработок за вынужденный прогул. "
            "Дополнительно грозит административная ответственность за нарушение "
            "трудового законодательства."
        ),
    ),
    EvalCase(
        id="cross_02",
        question=(
            "Как соотносятся нормы Гражданского и Предпринимательского кодексов "
            "при заключении хозяйственных договоров?"
        ),
        language="ru",
        query_type="cross_codex",
        expected_codexes=["civil_general", "entrepreneurial"],
        expected_article_numbers=["380", "381", "382"],  # art.381 = смешанный договор (between 380 and 382)
        reference_answer=(
            "При заключении хозяйственных договоров применяются нормы "
            "Предпринимательского кодекса, а в части, им не урегулированной, "
            "— нормы Гражданского кодекса."
        ),
    ),
    # ── Off-topic Russian (non-legal) ─────────────────────────────────────────
    EvalCase(
        id="offtopic_01",
        question="Какая погода сейчас в Алматы?",
        language="ru",
        query_type="off_topic",
        expected_codexes=[],
        expected_article_numbers=[],
        reference_answer="",
        is_legal=False,
    ),
    EvalCase(
        id="offtopic_02",
        question="Как приготовить бешбармак?",
        language="ru",
        query_type="off_topic",
        expected_codexes=[],
        expected_article_numbers=[],
        reference_answer="",
        is_legal=False,
    ),
    EvalCase(
        id="offtopic_03",
        question="Кто выиграл чемпионат мира по футболу в 2022 году?",
        language="ru",
        query_type="off_topic",
        expected_codexes=[],
        expected_article_numbers=[],
        reference_answer="",
        is_legal=False,
    ),
    EvalCase(
        id="offtopic_04",
        question="Какова численность населения Казахстана на сегодняшний день?",
        language="ru",
        query_type="off_topic",
        expected_codexes=[],
        expected_article_numbers=[],
        reference_answer="",
        is_legal=False,
    ),
    # ── Off-topic Kazakh (non-legal) ──────────────────────────────────────────
    EvalCase(
        id="offtopic_05",
        question="Астана қаласындағы ауа райы қандай?",
        language="kz",
        query_type="off_topic",
        expected_codexes=[],
        expected_article_numbers=[],
        reference_answer="",
        is_legal=False,
    ),
    EvalCase(
        id="offtopic_06",
        question="Қымыз қалай дайындалады?",
        language="kz",
        query_type="off_topic",
        expected_codexes=[],
        expected_article_numbers=[],
        reference_answer="",
        is_legal=False,
    ),
    EvalCase(
        id="offtopic_07",
        question="Қазақстанның ең биік тауы қайсы?",
        language="kz",
        query_type="off_topic",
        expected_codexes=[],
        expected_article_numbers=[],
        reference_answer="",
        is_legal=False,
    ),
    # ── Labor Law — Kazakh ────────────────────────────────────────────────────
    EvalCase(
        id="labor_05",
        question="Жұмыс аптасының ең ұзақ уақыты қанша сағат?",
        language="kz",
        query_type="factual",
        expected_codexes=["labor"],
        expected_article_numbers=["68", "69"],
        reference_answer=(
            "Жұмыс аптасының ұзақтығы 40 сағаттан аспауы тиіс. "
            "Кейбір санаттағы жұмыскерлер үшін қысқартылған жұмыс уақыты белгіленуі мүмкін."
        ),
    ),
    EvalCase(
        id="labor_06",
        question="Еңбек тәртібін бұзғаны үшін қандай тәртіптік жазалар қолданылады?",
        language="kz",
        query_type="definitional",
        expected_codexes=["labor"],
        expected_article_numbers=["64", "65"],
        reference_answer=(
            "Жұмыскерге ескерту, сөгіс немесе жұмыстан шығару түріндегі "
            "тәртіптік жазалар қолданылады. Бір тәртіптік теріс қылық үшін "
            "тек бір тәртіптік жаза қолданылады."
        ),
    ),
    # ── Criminal Law — Kazakh ─────────────────────────────────────────────────
    EvalCase(
        id="criminal_03",
        question="Адам өлтіргені үшін қылмыстық кодекс қандай жаза белгілейді?",
        language="kz",
        query_type="definitional",
        expected_codexes=["criminal"],
        expected_article_numbers=["99"],
        reference_answer=(
            "Адам өлтіру, яғни екінші адамды заңсыз өлтіру үшін бас бостандығынан "
            "айыру жазасы белгіленеді. Ауырлататын мән-жайлар болған кезде өмір бойы "
            "бас бостандығынан айыруға дейін жетуі мүмкін."
        ),
    ),
    EvalCase(
        id="criminal_04",
        question="Қажетті қорғаныс деген не және оның шегі қандай?",
        language="kz",
        query_type="definitional",
        expected_codexes=["criminal"],
        expected_article_numbers=["32", "33"],
        reference_answer=(
            "Қажетті қорғаныс — адамның өзіне немесе өзгеге бағытталған қоғамдық "
            "қауіпті шабуылды тоқтату үшін зиян келтіруі. Қорғаныс шектен шықпаса, "
            "қылмыстық жауапкершілік туындамайды."
        ),
    ),
    # ── Civil Law — Kazakh ────────────────────────────────────────────────────
    EvalCase(
        id="civil_03",
        question="Азаматтық кодекс бойынша меншік құқығы дегеніміз не?",
        language="kz",
        query_type="definitional",
        expected_codexes=["civil_general"],
        expected_article_numbers=["188", "189"],
        reference_answer=(
            "Меншік құқығы — заттарды заңда белгіленген шектерде өз қалауынша "
            "иелену, пайдалану және билік ету құқықтарының жиынтығы."
        ),
    ),
    EvalCase(
        id="civil_04",
        question="Шарт міндеттемесі бұзылған кезде залалды өтеу тәртібі қандай?",
        language="kz",
        query_type="procedural",
        expected_codexes=["civil_general"],
        expected_article_numbers=["350", "351"],
        reference_answer=(
            "Міндеттемені бұзған кінәлі тарап нақты залалды да, табыс жоғалтуды да "
            "өтеуге міндетті. Тараптар шартта залалдың мөлшерін алдын ала белгілей алады."
        ),
    ),
    # ── Family Law — Kazakh ───────────────────────────────────────────────────
    EvalCase(
        id="family_04",
        question="Балаға алимент қалай есептеледі және оның мөлшері қандай?",
        language="kz",
        query_type="factual",
        expected_codexes=["family"],
        expected_article_numbers=["138", "139"],
        reference_answer=(
            "Бір балаға ата-ананың табысынан төрттен бір бөлігі, екі балаға үштен "
            "бір бөлігі, үш және одан да көп балаға жартысы өндіріледі."
        ),
    ),
    EvalCase(
        id="family_05",
        question="Баланы асырап алу үшін қандай шарттар орындалуы керек?",
        language="kz",
        query_type="procedural",
        expected_codexes=["family"],
        expected_article_numbers=["75", "76"],
        reference_answer=(
            "Асырап алу тек сот шешімімен жүзеге асырылады. Асырап алушы кәмелетке "
            "толған іс-әрекет қабілетті адам болуы, асырап алынатын баладан кемінде "
            "16 жас үлкен болуы тиіс."
        ),
    ),
    # ── Tax Law — Kazakh ──────────────────────────────────────────────────────
    EvalCase(
        id="tax_03",
        question="Жеке тұлғалардың мүліктеріне салынатын салықты кімдер төлейді?",
        language="kz",
        query_type="definitional",
        expected_codexes=["tax_code", "tax_payments"],
        expected_article_numbers=["521", "522"],
        reference_answer=(
            "Жеке тұлғалардың мүліктеріне салынатын салықты Қазақстан Республикасының "
            "аумағында салық салу объектілері бар жеке тұлғалар төлейді."
        ),
    ),
    EvalCase(
        id="tax_04",
        question="Корпоративтік табыс салығының негізгі ставкасы қандай?",
        language="kz",
        query_type="factual",
        expected_codexes=["tax_code", "tax_payments"],
        expected_article_numbers=["313", "314"],
        reference_answer=(
            "Корпоративтік табыс салығының негізгі ставкасы 20 пайызды құрайды."
        ),
    ),
    # ── Administrative Offenses — Kazakh ─────────────────────────────────────
    EvalCase(
        id="admin_02",
        question="Жол жүру ережелерін бұзғаны үшін қандай айыппұл салынады?",
        language="kz",
        query_type="factual",
        expected_codexes=["admin_offenses"],
        expected_article_numbers=["590", "591"],
        reference_answer=(
            "Жол жүру ережелерін бұзғаны үшін айыппұл мөлшері бұзушылықтың сипатына "
            "байланысты анықталады. Нақты айыппұл сомасы бұзушылық санатына қарай белгіленеді."
        ),
    ),
    EvalCase(
        id="admin_03",
        question="Әкімшілік қамауға алудың ең ұзақ мерзімі қандай?",
        language="kz",
        query_type="factual",
        expected_codexes=["admin_offenses"],
        expected_article_numbers=["50", "51"],
        reference_answer=(
            "Әкімшілік қамауға алу жаза ретінде 30 тәулікке дейін белгіленуі мүмкін, "
            "ал ерекше жағдайларда — 45 тәулікке дейін."
        ),
    ),
    # ── Environmental Law — Kazakh ────────────────────────────────────────────
    EvalCase(
        id="env_02",
        question="Қоршаған ортаны ластағаны үшін қандай жауаптылық көзделген?",
        language="kz",
        query_type="definitional",
        expected_codexes=["environmental"],
        expected_article_numbers=["321", "322"],
        reference_answer=(
            "Қоршаған ортаны ластағаны үшін азаматтық-құқықтық, әкімшілік немесе "
            "қылмыстық жауаптылық туындауы мүмкін. Кінәлі тұлғалар келтірген "
            "залалды толық өтеуге міндетті."
        ),
    ),
    # ── Health Law — Kazakh ───────────────────────────────────────────────────
    EvalCase(
        id="health_01",
        question="Пациенттің медициналық жәрдем алу кезіндегі негізгі құқықтары қандай?",
        language="kz",
        query_type="definitional",
        expected_codexes=["health"],
        expected_article_numbers=["84", "85"],
        reference_answer=(
            "Пациенттің тіршілікке ыңғайлы жағдайда медициналық жәрдем алуға, "
            "денсаулық жағдайы туралы ақпарат алуға, жеке өміріне қол сұқпаушылыққа "
            "және медициналық құпияның сақталуына құқығы бар."
        ),
    ),
    # ── Cross-codex — Kazakh ─────────────────────────────────────────────────
    EvalCase(
        id="cross_03",
        question="Жер учаскесін сатып алу-сату шартының ерекшелігі қандай?",
        language="kz",
        query_type="cross_codex",
        expected_codexes=["land", "civil_general"],
        expected_article_numbers=["155", "158"],
        reference_answer=(
            "Жер учаскесін сатып алу-сату шарты жазбаша нысанда жасалады және "
            "мемлекеттік тіркеуден өтуі тиіс. Жер кодексінің ерекше нормалары "
            "Азаматтық кодекстің жалпы нормаларына қатысты басымдыққа ие."
        ),
    ),
]


# =============================================================================
# 2.  RETRIEVAL METRICS
# =============================================================================

def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer (no NLTK dependency)."""
    import re
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def precision_at_k(retrieved_numbers: list[str], relevant: set[str], k: int) -> float:
    top = retrieved_numbers[:k]
    if not top:
        return 0.0
    hits = sum(1 for n in top if n in relevant)
    return hits / k


def recall_at_k(retrieved_numbers: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0  # nothing expected → trivially recalled
    top = retrieved_numbers[:k]
    hits = sum(1 for n in top if n in relevant)
    return hits / len(relevant)


def f1_at_k(prec: float, rec: float) -> float:
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mrr(retrieved_numbers: list[str], relevant: set[str]) -> float:
    for rank, num in enumerate(retrieved_numbers, start=1):
        if num in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_numbers: list[str], relevant: set[str], k: int) -> float:
    top = retrieved_numbers[:k]
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, num in enumerate(top, start=1)
        if num in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved_numbers: list[str], relevant: set[str], k: int) -> float:
    return float(any(n in relevant for n in retrieved_numbers[:k]))


def compute_retrieval_metrics(
    retrieved_numbers: list[str],
    expected_numbers: list[str],
    k_values: list[int],
) -> dict[str, float]:
    """Compute all retrieval metrics for a single case."""
    if not expected_numbers:
        return {}  # off-topic or no ground truth → skip

    relevant = set(expected_numbers)
    results: dict[str, float] = {}

    for k in k_values:
        p = precision_at_k(retrieved_numbers, relevant, k)
        r = recall_at_k(retrieved_numbers, relevant, k)
        results[f"precision@{k}"] = p
        results[f"recall@{k}"]    = r
        results[f"f1@{k}"]        = f1_at_k(p, r)
        results[f"ndcg@{k}"]      = ndcg_at_k(retrieved_numbers, relevant, k)
        results[f"hit@{k}"]       = hit_at_k(retrieved_numbers, relevant, k)

    results["mrr"] = mrr(retrieved_numbers, relevant)
    return results


# =============================================================================
# 3.  GENERATION / TEXT QUALITY METRICS
# =============================================================================

def rouge_n_score(hypothesis: str, reference: str, n: int) -> dict[str, float]:
    """Compute ROUGE-N precision, recall, and F1."""
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)
    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)

    if not hyp_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_count: dict[tuple, int] = defaultdict(int)
    for ng in ref_ngrams:
        ref_count[ng] += 1

    overlap = 0
    for ng in hyp_ngrams:
        if ref_count.get(ng, 0) > 0:
            overlap += 1
            ref_count[ng] -= 1

    precision = overlap / len(hyp_ngrams)
    recall    = overlap / len(ref_ngrams)
    f1        = f1_at_k(precision, recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_score(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L using Longest Common Subsequence."""
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)

    if not hyp or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs     = dp[m][n]
    precision = lcs / n if n else 0.0
    recall    = lcs / m if m else 0.0
    f1        = f1_at_k(precision, recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    """Return ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not hypothesis or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return {
        "rouge1": rouge_n_score(hypothesis, reference, 1)["f1"],
        "rouge2": rouge_n_score(hypothesis, reference, 2)["f1"],
        "rougeL": rouge_l_score(hypothesis, reference)["f1"],
    }


# ── Semantic Similarity (SBERT cosine) ───────────────────────────────────────

_sbert_model: Any = None


def _get_sbert() -> Any | None:
    global _sbert_model
    if not _SBERT_AVAILABLE:
        return None
    if _sbert_model is None:
        console.print(
            "  [dim]Loading sentence-transformer model "
            "(paraphrase-multilingual-MiniLM-L12-v2)…[/dim]"
        )
        _sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _sbert_model


def semantic_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between SBERT embeddings; returns 0.0 if unavailable."""
    if not text1 or not text2:
        return 0.0
    model = _get_sbert()
    if model is None:
        return float("nan")
    emb = model.encode([text1, text2], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


# =============================================================================
# 4.  LLM-AS-JUDGE
# =============================================================================

_JUDGE_FAITHFULNESS_PROMPT = """\
You are an impartial evaluator for a legal RAG system that answers questions about \
Kazakhstan legislation.

Question: {question}

Retrieved context (law articles used by the system):
{context}

System answer:
{answer}

Task: Score the FAITHFULNESS of the answer (0.0 – 1.0).
Faithfulness means the answer's legal claims are grounded in the retrieved context.
Important rules:
- Paraphrasing or summarising statutory text counts as grounded (score high).
- Minor stylistic differences from the exact article wording are acceptable.
- Only penalise if the answer makes specific legal claims (article numbers, rates, \
deadlines, prohibitions) that are absent from or contradicted by the context.
- A score of 1.0 = fully grounded; 0.0 = key claims completely absent from context.

Return ONLY valid JSON: {{"faithfulness": <float 0.0-1.0>, "reason": "<one sentence>"}}
"""

_JUDGE_RELEVANCE_PROMPT = """\
You are an impartial evaluator for a legal RAG system.

Question: {question}

System answer:
{answer}

Task: Score the ANSWER RELEVANCE (0.0 – 1.0).
Relevance means the answer actually addresses what was asked.
A score of 1.0 = perfectly on-topic; 0.0 = completely off-topic.

Return ONLY valid JSON: {{"relevance": <float 0.0-1.0>, "reason": "<one sentence>"}}
"""


async def _llm_judge_call(prompt: str) -> dict:
    """Single DeepSeek call for LLM-as-judge evaluation."""
    from openai import AsyncOpenAI  # noqa: PLC0415

    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=30,
        max_retries=1,
    )
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        ),
        timeout=30,
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    return json.loads(raw)


async def judge_faithfulness(
    question: str, answer: str, context: str
) -> tuple[float, str]:
    """Return (score, reason). Falls back to NaN on error."""
    try:
        data = await _llm_judge_call(
            _JUDGE_FAITHFULNESS_PROMPT.format(
                question=question, context=context[:6000], answer=answer[:2000]
            )
        )
        score  = float(data.get("faithfulness", float("nan")))
        reason = str(data.get("reason", ""))
        return max(0.0, min(1.0, score)), reason
    except Exception as exc:
        return float("nan"), f"error: {exc}"


async def judge_relevance(question: str, answer: str) -> tuple[float, str]:
    """Return (score, reason). Falls back to NaN on error."""
    try:
        data = await _llm_judge_call(
            _JUDGE_RELEVANCE_PROMPT.format(question=question, answer=answer[:2000])
        )
        score  = float(data.get("relevance", float("nan")))
        reason = str(data.get("reason", ""))
        return max(0.0, min(1.0, score)), reason
    except Exception as exc:
        return float("nan"), f"error: {exc}"


# =============================================================================
# 5.  RESULT DATA CLASS
# =============================================================================

@dataclass
class CaseResult:
    """All evaluation outputs for one test case."""
    case_id:        str
    question:       str
    query_type:     str
    language:       str
    is_legal_gt:    bool   # ground-truth: is it a legal question?

    # Pipeline outputs
    predicted_legal:  bool  = False
    predicted_codexes: list[str] = field(default_factory=list)
    retrieved_numbers: list[str] = field(default_factory=list)  # article numbers in order
    bm25_count:       int   = 0
    graph_count:      int   = 0
    answer_text:      str   = ""
    confidence:       float = 0.0
    latency_ms:       int   = 0
    retried:          bool  = False
    num_citations:    int   = 0
    num_keywords:     int   = 0
    context_text:     str   = ""
    error:            str   = ""

    # Retrieval metrics (filled after pipeline run)
    retrieval: dict[str, float] = field(default_factory=dict)

    # Generation metrics
    rouge1:    float = float("nan")
    rouge2:    float = float("nan")
    rougeL:    float = float("nan")
    sem_sim:   float = float("nan")

    # LLM judge
    faithfulness: float = float("nan")
    relevance:    float = float("nan")

    # Intent
    codex_routing_precision: float = float("nan")
    codex_routing_recall:    float = float("nan")
    codex_routing_f1:        float = float("nan")


# =============================================================================
# 6.  MAIN EVALUATION LOOP
# =============================================================================

async def evaluate_case(
    case: EvalCase,
    k_values: list[int],
    run_llm_judge: bool,
) -> CaseResult:
    """Run the full pipeline on one test case and compute all metrics."""
    result = CaseResult(
        case_id=case.id,
        question=case.question,
        query_type=case.query_type,
        language=case.language,
        is_legal_gt=case.is_legal,
    )

    # ── Run pipeline ─────────────────────────────────────────────────────────
    session = Session()
    try:
        pr: PipelineResult = await run(case.question, session)
    except Exception as exc:
        result.error = str(exc)
        return result

    # ── Extract raw outputs ───────────────────────────────────────────────────
    result.predicted_legal   = pr.intent.is_legal_question
    result.predicted_codexes = list(pr.intent.codex_slugs)
    result.answer_text       = pr.answer.answer
    result.confidence        = pr.answer.confidence
    result.latency_ms        = pr.elapsed_ms
    result.retried           = pr.retried
    result.num_citations     = len(pr.answer.citations)
    result.num_keywords      = len(pr.intent.keywords)
    result.context_text      = pr.retrieval.context_text

    stats = pr.retrieval.stats
    result.bm25_count  = stats.get("bm25_hits", 0)
    result.graph_count = stats.get("graph_hits", 0)

    # Article numbers in ranked retrieval order
    result.retrieved_numbers = [a.number for a in pr.retrieval.articles]

    # ── Retrieval metrics (only for legal queries with expected articles) ─────
    if case.is_legal and case.expected_article_numbers:
        result.retrieval = compute_retrieval_metrics(
            result.retrieved_numbers, case.expected_article_numbers, k_values
        )

    # ── Intent / Codex routing ────────────────────────────────────────────────
    if case.expected_codexes:
        expected_set  = set(case.expected_codexes)
        predicted_set = set(result.predicted_codexes)
        tp = len(expected_set & predicted_set)
        fp = len(predicted_set - expected_set)
        fn = len(expected_set - predicted_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result.codex_routing_precision = prec
        result.codex_routing_recall    = rec
        result.codex_routing_f1        = f1_at_k(prec, rec)

    # ── Generation metrics (only for legal queries with reference answers) ────
    if case.is_legal and case.reference_answer and result.answer_text:
        rouge = compute_rouge(result.answer_text, case.reference_answer)
        result.rouge1  = rouge["rouge1"]
        result.rouge2  = rouge["rouge2"]
        result.rougeL  = rouge["rougeL"]
        result.sem_sim = semantic_similarity(result.answer_text, case.reference_answer)

    # ── LLM-as-Judge ─────────────────────────────────────────────────────────
    if run_llm_judge and case.is_legal and result.answer_text:
        result.faithfulness, _ = await judge_faithfulness(
            case.question, result.answer_text, result.context_text
        )
        result.relevance, _ = await judge_relevance(case.question, result.answer_text)

    return result


# =============================================================================
# 6b. BM25-ONLY BASELINE (ablation: no graph enrichment)
# =============================================================================

async def _retrieve_bm25_only(intent: Any) -> list[str]:
    """Run BM25-only retrieval (skip graph_neighbours); return article numbers in ranked order."""
    from multi_agent_rag.database import bm25_search  # noqa: PLC0415

    keywords    = intent.keywords
    codex_slugs = intent.codex_slugs

    score_map: dict[str, tuple[str, float]] = {}  # id -> (number, score)

    for kw in keywords:
        limit = max(5, round(20 * (0.5 + kw.weight)))
        rows: list[dict] = []
        if codex_slugs:
            rows = await bm25_search(kw.text, limit=limit, codex_slugs=codex_slugs)
        if not rows:
            rows = await bm25_search(kw.text, limit=limit)

        for row in rows:
            aid    = (row.get("id") or "").strip()
            number = str(row.get("number") or "")
            score  = (row.get("score") or 0.0) * kw.weight
            if not aid:
                continue
            if aid not in score_map:
                score_map[aid] = (number, score)
            else:
                prev_num, prev_score = score_map[aid]
                score_map[aid] = (prev_num, prev_score + score * 0.5)

    ranked = sorted(score_map.values(), key=lambda x: x[1], reverse=True)
    return [num for num, _ in ranked]


async def evaluate_case_baseline(
    case: "EvalCase",
    k_values: list[int],
) -> dict[str, float]:
    """Run intent analysis + BM25-only retrieval; return retrieval metrics for ablation."""
    from multi_agent_rag.llm import analyse_intent  # noqa: PLC0415

    if not case.is_legal or not case.expected_article_numbers:
        return {}

    session = Session()
    try:
        intent = await analyse_intent(case.question, history_ctx="")
    except Exception:
        return {}

    try:
        retrieved_numbers = await _retrieve_bm25_only(intent)
    except Exception:
        return {}

    return compute_retrieval_metrics(retrieved_numbers, case.expected_article_numbers, k_values)


# =============================================================================
# 7.  AGGREGATE METRICS
# =============================================================================

def _safe_mean(vals: list[float]) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def _safe_std(vals: list[float]) -> float:
    """Population standard deviation over non-NaN values."""
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.std(valid, ddof=0)) if len(valid) > 1 else float("nan")


def _safe_percentile(vals: list[float], p: float) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.percentile(valid, p)) if valid else float("nan")


def aggregate(results: list[CaseResult], k_values: list[int]) -> dict:
    """Compute all aggregate metrics from per-case results."""
    legal_results    = [r for r in results if r.is_legal_gt and not r.error]
    offtopic_results = [r for r in results if not r.is_legal_gt and not r.error]
    retrieval_results = [r for r in legal_results if r.retrieval]

    # ── System metrics ────────────────────────────────────────────────────────
    latencies   = [r.latency_ms for r in legal_results]
    confidences = [r.confidence for r in legal_results]
    retry_flags = [r.retried    for r in legal_results]
    cit_rates   = [1 if r.num_citations > 0 else 0 for r in legal_results]

    conf_low  = sum(1 for c in confidences if c < 0.40)
    conf_med  = sum(1 for c in confidences if 0.40 <= c < 0.75)
    conf_high = sum(1 for c in confidences if c >= 0.75)

    # ── Off-topic detection ───────────────────────────────────────────────────
    offtopic_correct = sum(
        1 for r in offtopic_results if not r.predicted_legal
    )
    offtopic_acc = (
        offtopic_correct / len(offtopic_results)
        if offtopic_results else float("nan")
    )

    # ── Retrieval metrics (macro-averaged ± std) ─────────────────────────────
    retrieval_agg: dict[str, float] = {}
    for k in k_values:
        for metric in (f"precision@{k}", f"recall@{k}", f"f1@{k}",
                       f"ndcg@{k}", f"hit@{k}"):
            vals = [r.retrieval.get(metric, float("nan")) for r in retrieval_results]
            retrieval_agg[metric]            = _safe_mean(vals)
            retrieval_agg[f"{metric}_std"]   = _safe_std(vals)
    mrr_vals = [r.retrieval.get("mrr", float("nan")) for r in retrieval_results]
    retrieval_agg["mrr"]     = _safe_mean(mrr_vals)
    retrieval_agg["mrr_std"] = _safe_std(mrr_vals)

    # ── Generation metrics (mean ± std) ───────────────────────────────────────
    gen_results = [r for r in legal_results if not math.isnan(r.rouge1)]
    generation = {
        "rouge1":      _safe_mean([r.rouge1  for r in gen_results]),
        "rouge1_std":  _safe_std( [r.rouge1  for r in gen_results]),
        "rouge2":      _safe_mean([r.rouge2  for r in gen_results]),
        "rouge2_std":  _safe_std( [r.rouge2  for r in gen_results]),
        "rougeL":      _safe_mean([r.rougeL  for r in gen_results]),
        "rougeL_std":  _safe_std( [r.rougeL  for r in gen_results]),
        "sem_sim":     _safe_mean([r.sem_sim for r in gen_results]),
        "sem_sim_std": _safe_std( [r.sem_sim for r in gen_results]),
    }

    # ── LLM judge metrics (mean ± std) ────────────────────────────────────────
    judge_results = [r for r in legal_results if not math.isnan(r.faithfulness)]
    llm_judge = {
        "faithfulness":     _safe_mean([r.faithfulness for r in judge_results]),
        "faithfulness_std": _safe_std( [r.faithfulness for r in judge_results]),
        "relevance":        _safe_mean([r.relevance    for r in judge_results]),
        "relevance_std":    _safe_std( [r.relevance    for r in judge_results]),
    }

    # ── Intent / Codex routing ────────────────────────────────────────────────
    intent_results = [
        r for r in legal_results
        if not math.isnan(r.codex_routing_precision)
    ]
    intent = {
        "codex_precision": _safe_mean([r.codex_routing_precision for r in intent_results]),
        "codex_recall":    _safe_mean([r.codex_routing_recall    for r in intent_results]),
        "codex_f1":        _safe_mean([r.codex_routing_f1        for r in intent_results]),
        "mean_keywords":   _safe_mean([r.num_keywords            for r in legal_results]),
    }

    # ── Graph enrichment ──────────────────────────────────────────────────────
    total_bm25  = sum(r.bm25_count  for r in legal_results)
    total_graph = sum(r.graph_count for r in legal_results)
    total_cands = total_bm25 + total_graph
    graph_ratio = total_graph / total_cands if total_cands > 0 else 0.0

    graph_article_fractions = []
    for r in legal_results:
        n_retrieved = len(r.retrieved_numbers)
        if n_retrieved == 0:
            continue
        # We don't track per-article source here; use the graph_count / (bm25+graph) ratio
        # as a proxy for the enrichment fraction in this run
        total = r.bm25_count + r.graph_count
        if total > 0:
            graph_article_fractions.append(r.graph_count / total)

    return {
        "cases_total":           len(results),
        "cases_legal":           len(legal_results),
        "cases_offtopic":        len(offtopic_results),
        "cases_errored":         sum(1 for r in results if r.error),

        "system": {
            "latency_mean_ms":   _safe_mean(latencies),
            "latency_median_ms": _safe_percentile(latencies, 50),
            "latency_p95_ms":    _safe_percentile(latencies, 95),
            "latency_p99_ms":    _safe_percentile(latencies, 99),
            "latency_min_ms":    float(min(latencies)) if latencies else float("nan"),
            "latency_max_ms":    float(max(latencies)) if latencies else float("nan"),
            "confidence_mean":   _safe_mean(confidences),
            "confidence_std":    float(np.std(confidences)) if confidences else float("nan"),
            "confidence_low_pct":  100 * conf_low  / len(legal_results) if legal_results else float("nan"),
            "confidence_med_pct":  100 * conf_med  / len(legal_results) if legal_results else float("nan"),
            "confidence_high_pct": 100 * conf_high / len(legal_results) if legal_results else float("nan"),
            "retry_rate":          _safe_mean([float(f) for f in retry_flags]),
            "citation_rate":       _safe_mean([float(v) for v in cit_rates]),
            "offtopic_detection_accuracy": offtopic_acc,
        },

        "retrieval": retrieval_agg,
        "generation": generation,
        "llm_judge": llm_judge,
        "intent": intent,

        "graph": {
            "total_bm25_candidates":  total_bm25,
            "total_graph_candidates": total_graph,
            "graph_enrichment_ratio": graph_ratio,
            "mean_graph_fraction":    _safe_mean(graph_article_fractions),
        },
    }


# =============================================================================
# 8.  RICH REPORT
# =============================================================================

def _fmt(val: float, decimals: int = 3, pct: bool = False) -> str:
    if math.isnan(val):
        return "[dim]N/A[/dim]"
    if pct:
        return f"{val * 100:.1f}%"
    return f"{val:.{decimals}f}"


def _colour(val: float, low: float = 0.4, high: float = 0.7) -> str:
    if math.isnan(val):
        return "dim"
    if val >= high:
        return "bright_green"
    if val >= low:
        return "yellow"
    return "red"


def print_report(agg: dict, results: list[CaseResult], k_values: list[int]) -> None:
    """Print a formatted Rich report to console."""
    console.print()
    console.print(Rule("[bold white]EVALUATION RESULTS[/bold white]", style="blue"))
    console.print()

    # ── Overview ──────────────────────────────────────────────────────────────
    overview = Table(box=box.SIMPLE_HEAD, show_header=False, border_style="dim", padding=(0, 2))
    overview.add_column("Field", style="dim cyan")
    overview.add_column("Value", style="white")
    overview.add_row("Total cases",    str(agg["cases_total"]))
    overview.add_row("Legal cases",    str(agg["cases_legal"]))
    overview.add_row("Off-topic cases",str(agg["cases_offtopic"]))
    overview.add_row("Errored cases",  str(agg["cases_errored"]))
    overview.add_row("K values",       ", ".join(str(k) for k in k_values))
    console.print(Panel(overview, title="Overview", border_style="blue"))

    # ── System ────────────────────────────────────────────────────────────────
    sys_tbl = Table(box=box.SIMPLE_HEAD, show_header=False, border_style="dim", padding=(0, 2))
    sys_tbl.add_column("Metric", style="dim cyan")
    sys_tbl.add_column("Value",  style="white")
    s = agg["system"]
    sys_tbl.add_row("Latency — mean",    f"{s['latency_mean_ms']:.0f} ms")
    sys_tbl.add_row("Latency — median",  f"{s['latency_median_ms']:.0f} ms")
    sys_tbl.add_row("Latency — P95",     f"{s['latency_p95_ms']:.0f} ms")
    sys_tbl.add_row("Latency — P99",     f"{s['latency_p99_ms']:.0f} ms")
    sys_tbl.add_row("Latency — min/max", f"{s['latency_min_ms']:.0f} / {s['latency_max_ms']:.0f} ms")
    sys_tbl.add_row("Confidence — mean ± std",
                    f"{s['confidence_mean']:.3f} ± {s['confidence_std']:.3f}")
    sys_tbl.add_row("Confidence ≥ 0.75 (high)",
                    f"[bright_green]{s['confidence_high_pct']:.1f}%[/bright_green]")
    sys_tbl.add_row("Confidence 0.40–0.75 (medium)",
                    f"[yellow]{s['confidence_med_pct']:.1f}%[/yellow]")
    sys_tbl.add_row("Confidence < 0.40 (low)",
                    f"[red]{s['confidence_low_pct']:.1f}%[/red]")
    sys_tbl.add_row("Retry rate",          f"{s['retry_rate']*100:.1f}%")
    sys_tbl.add_row("Citation rate",       f"{s['citation_rate']*100:.1f}%")
    sys_tbl.add_row("Off-topic detection accuracy",
                    f"{s['offtopic_detection_accuracy']*100:.1f}%")
    console.print(Panel(sys_tbl, title="System Performance", border_style="blue"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    ret_tbl = Table(box=box.SIMPLE_HEAD, border_style="dim", padding=(0, 2))
    ret_tbl.add_column("Metric", style="dim cyan")
    for k in k_values:
        ret_tbl.add_column(f"@{k}", justify="right")
    if "mrr" in agg["retrieval"]:
        ret_tbl.add_column("MRR", justify="right")

    for metric_base in ("precision", "recall", "f1", "ndcg", "hit"):
        row = [metric_base.capitalize()]
        for k in k_values:
            key = f"{metric_base}@{k}"
            val = agg["retrieval"].get(key, float("nan"))
            std = agg["retrieval"].get(f"{key}_std", float("nan"))
            c   = _colour(val)
            std_str = f" ±{std:.3f}" if not math.isnan(std) else ""
            row.append(f"[{c}]{_fmt(val)}{std_str}[/{c}]")
        if "mrr" in agg["retrieval"] and metric_base == "f1":
            mrr_val = agg["retrieval"]["mrr"]
            mrr_std = agg["retrieval"].get("mrr_std", float("nan"))
            c = _colour(mrr_val)
            std_str = f" ±{mrr_std:.3f}" if not math.isnan(mrr_std) else ""
            row.append(f"[{c}]{_fmt(mrr_val)}{std_str}[/{c}]")
        elif "mrr" in agg["retrieval"]:
            row.append("")
        ret_tbl.add_row(*row)

    console.print(Panel(ret_tbl, title="Retrieval Metrics (macro-averaged over legal cases)", border_style="blue"))

    # ── Generation ────────────────────────────────────────────────────────────
    gen = agg["generation"]
    gen_tbl = Table(box=box.SIMPLE_HEAD, show_header=False, border_style="dim", padding=(0, 2))
    gen_tbl.add_column("Metric", style="dim cyan")
    gen_tbl.add_column("Score", justify="right")
    for label, key in (("ROUGE-1 F1", "rouge1"), ("ROUGE-2 F1", "rouge2"),
                       ("ROUGE-L F1", "rougeL"), ("Semantic Similarity (SBERT)", "sem_sim")):
        val = gen.get(key, float("nan"))
        std = gen.get(f"{key}_std", float("nan"))
        c   = _colour(val, low=0.2, high=0.5)
        std_str = f" ± {std:.3f}" if not math.isnan(std) else ""
        gen_tbl.add_row(label, f"[{c}]{_fmt(val)}{std_str}[/{c}]")
    console.print(Panel(gen_tbl, title="Generation Quality (vs reference answers)", border_style="blue"))

    # ── LLM Judge ─────────────────────────────────────────────────────────────
    jdg = agg["llm_judge"]
    jdg_tbl = Table(box=box.SIMPLE_HEAD, show_header=False, border_style="dim", padding=(0, 2))
    jdg_tbl.add_column("Metric", style="dim cyan")
    jdg_tbl.add_column("Score", justify="right")
    for label, key in (("Faithfulness", "faithfulness"), ("Answer Relevance", "relevance")):
        val = jdg.get(key, float("nan"))
        std = jdg.get(f"{key}_std", float("nan"))
        c   = _colour(val)
        std_str = f" ± {std:.3f}" if not math.isnan(std) else ""
        jdg_tbl.add_row(label, f"[{c}]{_fmt(val)}{std_str}[/{c}]")
    console.print(Panel(jdg_tbl, title="LLM-as-Judge (DeepSeek evaluator)", border_style="blue"))

    # ── Intent ────────────────────────────────────────────────────────────────
    intent = agg["intent"]
    int_tbl = Table(box=box.SIMPLE_HEAD, show_header=False, border_style="dim", padding=(0, 2))
    int_tbl.add_column("Metric", style="dim cyan")
    int_tbl.add_column("Score", justify="right")
    for label, key in (
        ("Codex Routing Precision", "codex_precision"),
        ("Codex Routing Recall",    "codex_recall"),
        ("Codex Routing F1",        "codex_f1"),
    ):
        val = intent.get(key, float("nan"))
        c   = _colour(val)
        int_tbl.add_row(label, f"[{c}]{_fmt(val)}[/{c}]")
    int_tbl.add_row("Mean keyword count per query",
                    f"{intent.get('mean_keywords', float('nan')):.1f}")
    console.print(Panel(int_tbl, title="Intent Analysis Quality", border_style="blue"))

    # ── Graph enrichment ──────────────────────────────────────────────────────
    g = agg["graph"]
    g_tbl = Table(box=box.SIMPLE_HEAD, show_header=False, border_style="dim", padding=(0, 2))
    g_tbl.add_column("Metric", style="dim cyan")
    g_tbl.add_column("Value", justify="right")
    g_tbl.add_row("Total BM25 candidates",  str(g["total_bm25_candidates"]))
    g_tbl.add_row("Total graph candidates", str(g["total_graph_candidates"]))
    ratio = g["graph_enrichment_ratio"]
    c     = _colour(ratio, low=0.1, high=0.3)
    g_tbl.add_row("Graph enrichment ratio",
                  f"[{c}]{_fmt(ratio, pct=True)}[/{c}]")
    g_tbl.add_row("Mean graph fraction per query",
                  _fmt(g["mean_graph_fraction"], pct=True))
    console.print(Panel(g_tbl, title="Knowledge Graph Contribution", border_style="blue"))

    # ── Language breakdown ────────────────────────────────────────────────────
    ru_legal = [r for r in results if r.language == "ru" and r.is_legal_gt and not r.error]
    kz_legal = [r for r in results if r.language == "kz" and r.is_legal_gt and not r.error]
    lang_tbl = Table(box=box.SIMPLE_HEAD, border_style="dim", padding=(0, 2))
    lang_tbl.add_column("Language", style="dim cyan")
    lang_tbl.add_column("Cases",    justify="right")
    lang_tbl.add_column("Conf mean", justify="right")
    lang_tbl.add_column("Sem Sim",   justify="right")
    lang_tbl.add_column("Cit rate",  justify="right")
    for label, group in (("Russian (ru)", ru_legal), ("Kazakh (kz)", kz_legal)):
        if not group:
            continue
        conf  = _safe_mean([r.confidence for r in group])
        sim   = _safe_mean([r.sem_sim for r in group if not math.isnan(r.sem_sim)])
        crate = _safe_mean([float(r.num_citations > 0) for r in group])
        lang_tbl.add_row(
            label,
            str(len(group)),
            f"[{_colour(conf)}]{conf:.3f}[/{_colour(conf)}]",
            f"[{_colour(sim, low=0.4, high=0.65)}]{_fmt(sim)}[/{_colour(sim, low=0.4, high=0.65)}]",
            f"{crate*100:.0f}%",
        )
    console.print(Panel(lang_tbl, title="Russian vs Kazakh — legal queries", border_style="blue"))

    # ── Per-case summary ──────────────────────────────────────────────────────
    case_tbl = Table(
        title="Per-Case Summary",
        box=box.SIMPLE_HEAD,
        border_style="dim",
        padding=(0, 1),
        show_lines=False,
    )
    case_tbl.add_column("ID",       style="cyan", no_wrap=True)
    case_tbl.add_column("Type",     style="dim")
    case_tbl.add_column("Lang",     style="dim")
    case_tbl.add_column("Conf",     justify="right")
    if k_values:
        k0 = k_values[0]
        case_tbl.add_column(f"P@{k0}", justify="right")
        case_tbl.add_column(f"R@{k0}", justify="right")
    case_tbl.add_column("ROUGE-L",  justify="right")
    case_tbl.add_column("SemSim",   justify="right")
    case_tbl.add_column("Ret (ms)", justify="right")
    case_tbl.add_column("Retry",    justify="center")
    case_tbl.add_column("Err",      style="red", no_wrap=True)

    for r in results:
        conf_c = _colour(r.confidence)
        k0 = k_values[0] if k_values else 5
        pk  = r.retrieval.get(f"precision@{k0}", float("nan"))
        rk  = r.retrieval.get(f"recall@{k0}",    float("nan"))

        case_tbl.add_row(
            r.case_id,
            r.query_type[:9],
            r.language,
            f"[{conf_c}]{r.confidence:.2f}[/{conf_c}]" if not r.error else "[red]ERR[/red]",
            _fmt(pk) if k_values else "",
            _fmt(rk) if k_values else "",
            _fmt(r.rougeL),
            _fmt(r.sem_sim),
            str(r.latency_ms),
            "↩" if r.retried else "·",
            r.error[:20] if r.error else "",
        )

    console.print(case_tbl)
    console.print()


# =============================================================================
# 9.  SERIALISATION
# =============================================================================

def _clean_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON serialisation."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


def save_results(
    agg: dict,
    results: list[CaseResult],
    out_path: Path,
) -> None:
    payload = {
        "aggregate": _clean_for_json(agg),
        "cases": [_clean_for_json(asdict(r)) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[dim]Results saved → [cyan]{out_path}[/cyan][/dim]\n")


# =============================================================================
# 10.  ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation suite for the Kazakhstan Legal RAG system."
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[5, 10],
        help="K values for retrieval metrics (default: 5 10)",
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="Skip LLM-as-judge evaluation (faster, no extra API cost)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(_ROOT) / "evaluation_results.json",
        help="Output path for JSON results (default: diploma_code/evaluation_results.json)",
    )
    parser.add_argument(
        "--cases", nargs="*",
        help="Run only specific case IDs (e.g. --cases labor_01 tax_01)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Also run BM25-only baseline (no graph enrichment) and print comparison table",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    console.print(Panel(
        Text.from_markup(
            "[bold cyan]Multi-Agent Graph-Based RAG — Evaluation Suite[/bold cyan]\n"
            "[dim]Legislation of the Republic of Kazakhstan[/dim]"
        ),
        border_style="blue",
        padding=(1, 4),
    ))

    # ── Neo4j connectivity check ───────────────────────────────────────────────
    console.print("[dim]Checking Neo4j connection…[/dim]", end=" ")
    if not await ping():
        console.print("[red]✗ Neo4j unreachable.[/red]")
        console.print(
            "[yellow]Make sure Neo4j is running and .env is configured, then retry.[/yellow]"
        )
        sys.exit(1)
    console.print("[bright_green]✓ connected[/bright_green]\n")

    # ── Select cases ──────────────────────────────────────────────────────────
    cases: list[EvalCase]
    if args.cases:
        id_set = set(args.cases)
        cases  = [c for c in GROUND_TRUTH if c.id in id_set]
        if not cases:
            console.print(f"[red]No matching case IDs: {args.cases}[/red]")
            sys.exit(1)
    else:
        cases = GROUND_TRUTH

    run_llm_judge = not args.no_llm_judge

    console.print(
        f"  Cases         : [cyan]{len(cases)}[/cyan]\n"
        f"  K values      : [cyan]{args.k}[/cyan]\n"
        f"  LLM judge     : [cyan]{'yes' if run_llm_judge else 'no (--no-llm-judge)'}[/cyan]\n"
        f"  SBERT sim     : [cyan]{'yes' if _SBERT_AVAILABLE else 'no (pip install sentence-transformers)'}[/cyan]\n"
        f"  Ablation      : [cyan]{'yes (--ablation)' if args.ablation else 'no'}[/cyan]"
    )
    console.print()

    # ── Run evaluation ────────────────────────────────────────────
    results: list[CaseResult] = []
    k_values: list[int] = args.k

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Evaluating…", total=len(cases))
        for case in cases:
            progress.update(task_id, description=f"[cyan]{case.id}[/cyan]")
            result = await evaluate_case(case, k_values, run_llm_judge=run_llm_judge)
            results.append(result)
            progress.advance(task_id)

    # ── Aggregate ─────────────────────────────────────────────────────
    agg = aggregate(results, k_values)

    # ── Report ────────────────────────────────────────────────────────
    print_report(agg, results, k_values)

    # ── Ablation: BM25-only baseline ───────────────────────────────────
    if args.ablation:
        legal_cases = [c for c in cases if c.is_legal and c.expected_article_numbers]
        console.print(Rule("[bold white]ABLATION: BM25-only baseline[/bold white]", style="yellow"))
        console.print(f"  [dim]Running BM25-only retrieval on {len(legal_cases)} legal cases "
                      f"(no graph enrichment, no answer synthesis)…[/dim]\n")

        baseline_results: list[tuple[str, dict[str, float]]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console, transient=True,
        ) as prog:
            t = prog.add_task("BM25 baseline…", total=len(legal_cases))
            for case in legal_cases:
                prog.update(t, description=f"[yellow]{case.id}[/yellow]")
                metrics = await evaluate_case_baseline(case, k_values)
                baseline_results.append((case.id, metrics))
                prog.advance(t)

        # Aggregate baseline retrieval metrics
        def _agg_baseline(key: str) -> tuple[float, float]:
            vals = [m.get(key, float("nan")) for _, m in baseline_results]
            return _safe_mean(vals), _safe_std(vals)

        # Build comparison table
        cmp_tbl = Table(
            title="Graph-enriched vs BM25-only — Retrieval Metric Comparison",
            box=box.SIMPLE_HEAD, border_style="yellow", padding=(0, 2),
        )
        cmp_tbl.add_column("Metric",           style="dim cyan")
        cmp_tbl.add_column("Full (mean ± std)", justify="right")
        cmp_tbl.add_column("BM25-only (mean ± std)", justify="right")
        cmp_tbl.add_column("Δ",                justify="right")

        compare_keys = (
            [(f"hit@{k}",  f"Hit@{k}")  for k in k_values] +
            [(f"ndcg@{k}", f"NDCG@{k}") for k in k_values] +
            [("mrr", "MRR")]
        )
        for key, label in compare_keys:
            full_mean = agg["retrieval"].get(key, float("nan"))
            full_std  = agg["retrieval"].get(f"{key}_std", float("nan"))
            base_mean, base_std = _agg_baseline(key)
            delta = full_mean - base_mean if not (math.isnan(full_mean) or math.isnan(base_mean)) else float("nan")

            def _fmt_pm(m: float, s: float) -> str:
                if math.isnan(m):
                    return "[dim]N/A[/dim]"
                s_str = f" ± {s:.3f}" if not math.isnan(s) else ""
                return f"{m:.3f}{s_str}"

            delta_str = (
                f"[bright_green]+{delta:.3f}[/bright_green]" if delta > 0.005
                else f"[red]{delta:.3f}[/red]" if delta < -0.005
                else f"[dim]{delta:.3f}[/dim]"
            ) if not math.isnan(delta) else "[dim]N/A[/dim]"

            cmp_tbl.add_row(label, _fmt_pm(full_mean, full_std),
                            _fmt_pm(base_mean, base_std), delta_str)

        console.print(cmp_tbl)
        console.print()

        # Save ablation results alongside main results
        ablation_payload = {
            "baseline_retrieval": {
                case_id: _clean_for_json(metrics)
                for case_id, metrics in baseline_results
            },
            "baseline_aggregate": {
                key: _clean_for_json({"mean": _agg_baseline(key)[0],
                                      "std":  _agg_baseline(key)[1]})
                for key, _ in compare_keys
            },
        }
        ablation_path = args.out.with_name(args.out.stem + "_ablation.json")
        ablation_path.write_text(
            json.dumps(ablation_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[dim]Ablation results saved → [cyan]{ablation_path}[/cyan][/dim]\n")

    # ── Save ────────────────────────────────────────────────────────────
    save_results(agg, results, args.out)


if __name__ == "__main__":
    asyncio.run(main())
