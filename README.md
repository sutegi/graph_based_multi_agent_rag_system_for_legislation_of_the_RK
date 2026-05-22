# Graph Based Multi-Agent RAG System for Legislation of the Republic of Kazakhstan

A retrieval-augmented generation system that answers legal questions about Kazakhstani legislation. The system combines BM25 fulltext search, knowledge graph traversal, and a large language model to produce precise, article-grounded answers in Russian and Kazakh.

---

## Overview

The knowledge base contains **9,587 articles** from **24 legal codes** of the Republic of Kazakhstan, ingested into a Neo4j graph database. Articles are connected by semantic horizontal relations (references, extensions, correlations, etc.), enabling the system to surface not only directly matching articles but also legally related ones through graph traversal.

Queries are processed through a three-phase pipeline:

```
Question → [Intent Analysis] → [Graph-Augmented Retrieval] → [Answer Synthesis] → Cited Answer
```

Each answer includes direct citations to the articles used, a confidence score, and full multi-turn conversation memory.

---

## Architecture

```
diploma_code/
├── main.py                    Entry point — Rich terminal chat interface
├── multi_agent_rag/
│   ├── config.py              Environment, constants, 24-codex registry
│   ├── database.py            Async Neo4j client (BM25, graph traversal, definitions)
│   ├── retriever.py           Weighted BM25 + one-hop graph enrichment
│   ├── llm.py                 LLM calls — intent analysis & answer synthesis
│   ├── session.py             Conversation history manager
│   ├── pipeline.py            3-phase pipeline orchestrator
│   └── __init__.py            Public API
├── ingestion/                 Scripts to build the Neo4j database from source
├── data/
│   ├── hierarchical_graph_base/   Article JSON with keywords and semantics
│   ├── horizontal_relations/      Cross-article relation graph
│   └── keyword_definitions/       Legal term definitions
└── requirements.txt
```

### Pipeline phases

**Phase 1 — Intent Analysis**  
The LLM classifies the question: detects the language (Russian/Kazakh), identifies the relevant legal code(s) from the 24-codex registry, and extracts 4–10 search keywords ranked by importance weight.

**Phase 2 — Graph-Augmented Retrieval**  
BM25 fulltext searches run in parallel for each keyword. The result limit per keyword is scaled by its weight (`limit = round(10 × (0.5 + weight))`), so the most important terms get broader searches. Results are scored by `bm25_score × keyword_weight` and merged across keywords. The top-8 scoring articles seed a one-hop graph traversal that pulls in related articles via six relation types: `CORRELATED_WITH`, `REFERENCES`, `EXTENDS`, `MANIFESTED_IN`, `BASED_ON`, `CAUSED_BY`. Legal term definitions are fetched in parallel for the top-3 keywords. Everything is packed into a single ranked context string (≤ 18,000 chars).

**Phase 3 — Answer Synthesis**  
The LLM produces a structured JSON response: `{ answer, citations, confidence }`. If confidence falls below the threshold (0.55), the pipeline automatically retries with no codex filter (broader corpus) and returns whichever attempt scored higher.

### Knowledge graph schema

```
(Article) -[:HAS_KEYWORD]->   (Keyword) -[:HAS_DEFINITION]-> (Definition)
(Article) -[:CORRELATED_WITH|REFERENCES|EXTENDS|
            MANIFESTED_IN|BASED_ON|CAUSED_BY]-       (Article)
(Section/Chapter/Paragraph) -[:CONTAINS]->           (Article)
```

Fulltext index: `article_fulltext_ru` on `Article.body_ru` and `Article.name_ru`.

---

## Requirements

- Python **3.10+**
- **Neo4j 5.x** — local or remote instance with the database already populated
- **DeepSeek API key** (or any OpenAI-compatible endpoint)

---

## Installation

```powershell
git clone <repo-url>
cd diploma_code
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the **project root** (the `diploma/` directory, one level above `diploma_code/`):

```env
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

---

## Usage

```powershell
cd diploma_code
python main.py
```

The system verifies the Neo4j connection on startup, then opens an interactive chat prompt.

```
╔══════════════════════════════════════════════════════════╗
║   Казахстанский Правовой ИИ-Ассистент                    ║
║   Kazakhstan Legal AI Assistant                          ║
╚══════════════════════════════════════════════════════════╝

Вы: Можно ли уволить беременную женщину?

  ↳ Кодекс: Трудовой кодекс  |  Термины: увольнение беременной, расторжение трудового договора

╭─ Ответ ────────────────────────────────────── уверенность 91% · 1843 мс ─╮
│  Согласно статье 55 Трудового кодекса РК, расторжение трудового           │
│  договора по инициативе работодателя с беременными женщинами              │
│  не допускается...                                                        │
╰───────────────────────────────────────────────────────────────────────────╯
```

### Commands

| Command  | Description                                 |
|----------|---------------------------------------------|
| `/help`  | Show available commands                     |
| `/clear` | Clear conversation history                  |
| `/stats` | Detailed retrieval statistics for last query |
| `/quit`  | Exit                                        |

---

## Legal Codes Covered

The system covers all 24 codes of the Republic of Kazakhstan, including:

| Slug | Code |
|------|------|
| `labor` | Трудовой кодекс |
| `criminal` | Уголовный кодекс |
| `civil_general` | Гражданский кодекс (Общая часть) |
| `civil_special` | Гражданский кодекс (Особенная часть) |
| `tax_code` | Налоговый кодекс |
| `admin_offenses` | Кодекс об административных правонарушениях |
| `family` | Кодекс о браке (супружестве) и семье |
| `environmental` | Экологический кодекс |
| `health` | Кодекс о здоровье народа и системе здравоохранения |
| `entrepreneurial` | Предпринимательский кодекс |
| `...` | + 14 more codes |

---

## Rebuilding the Database

If you need to repopulate Neo4j from the source JSON files in `./data/`:

```powershell
cd ingestion
python ingest.py
```

This requires all packages in `requirements.txt` including the NLP/embedding dependencies.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Graph database | Neo4j 5.x (async driver) |
| Fulltext search | Neo4j BM25 (`db.index.fulltext`) |
| LLM | DeepSeek via OpenAI-compatible API |
| Terminal UI | Rich |
| Language | Python 3.10+ (asyncio throughout) |
