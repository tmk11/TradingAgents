# Optional upgrade: news + macro archive (RAG over dataflow output)

This page documents the third opt-in RAG upgrade. The first two —
[semantic memory and parallel analysts](./rag_and_parallel.md) — gave
the Portfolio Manager smarter past-context retrieval and made the
analyst phase concurrent. This one closes a different gap: the
**dataflow tools that fetch news and macro data discard everything
they fetch** the moment the analysis run ends. There is no corpus,
no caching, no cross-run memory.

When `news_archive_enabled = True`, every fetch is mirrored into a
persistent Chroma store, and two new tools become available to the
analysts so they can query the accumulated corpus by embedding
similarity rather than paying a fresh fetch every run.

The upgrade is independent of `rag_enabled` (the decision-log RAG)
and off by default. Existing runs are unchanged.

---

## What gets indexed

| Source                          | Trigger                                    | `source` tag                  |
| ------------------------------- | ------------------------------------------ | ----------------------------- |
| Yahoo Finance ticker news       | every `get_news` call                      | `yfinance:ticker`             |
| Yahoo Finance global macro news | every `get_global_news` call               | `yfinance:global`             |
| Mining.com / Investing.com / Bloomberg RSS feeds | every `get_gold_news` call (commodity runs) | `rss:<host>` (e.g. `rss:mining.com`) |
| Macro data snapshot             | every `get_macro_data` call (commodity)    | (kind=`macro_snapshot`)       |

Every record carries metadata for `where` filtering: `ticker`,
`published_date`, `published_at_ts` (epoch), `kind` (`article` vs
`macro_snapshot`), `source`. Ticker filters are case-insensitive.

Indexing is **fire-and-forget**. The hooks live in
`tradingagents/dataflows/_archive_indexer.py` and never raise — a
broken Chroma install or a missing OpenAI key only disables future
indexing for the rest of the process; the trading run completes.

## Architecture

```
┌──────────────────────────┐                ┌─────────────────────────┐
│ get_news / global / gold │ ──── on ────►  │  Chroma collection      │
│ get_macro_data           │     fetch       │  news_archive           │
│ (dataflow tools)         │                 │  ~/.tradingagents/      │
└──────────────────────────┘                 │    news_archive/chroma  │
                                             └────────────┬────────────┘
                                                          │
                              search_news_archive  ◄──────┤
                              search_macro_archive ◄──────┘
                                          │
                                          ▼
                         News Analyst              Market Analyst
                         (any asset type)          (commodity only)
```

## Files of interest

| File                                                        | Role                                                               |
| ----------------------------------------------------------- | ------------------------------------------------------------------ |
| `tradingagents/retrieval/news_archive.py`                   | `ArchiveArticle` + `NewsArchive` (kind-scoped index/search/format) |
| `tradingagents/dataflows/_archive_indexer.py`               | Lazy facade with `record_news_articles` + `record_macro_snapshot`  |
| `tradingagents/dataflows/yfinance_news.py`                  | Hooks ticker + global news fetches                                 |
| `tradingagents/dataflows/gold_news.py`                      | Hooks per-RSS-feed fetches                                         |
| `tradingagents/dataflows/macro_data.py`                     | Hooks rendered-snapshot indexing                                   |
| `tradingagents/agents/utils/archive_search_tools.py`        | The two `@tool` callables                                          |
| `tradingagents/graph/trading_graph.py`                      | Adds tools to the four `ToolNode` instances                        |
| `tradingagents/agents/analysts/news_analyst.py`             | Binds `search_news_archive` when archive is enabled                |
| `tradingagents/agents/analysts/market_analyst.py`           | Binds `search_macro_archive` for commodity runs when enabled       |

## Enabling

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
config["news_archive_enabled"] = True
ta = TradingAgentsGraph(config=config)
ta.propagate("GLD", "2026-05-22", asset_type="commodity")
```

Or via env vars:

```bash
export TRADINGAGENTS_NEWS_ARCHIVE_ENABLED=true
export TRADINGAGENTS_NEWS_ARCHIVE_PATH=/some/path/chroma   # optional
export TRADINGAGENTS_RAG_EMBEDDING_PROVIDER=openai          # default; reused
export TRADINGAGENTS_RAG_EMBEDDING_MODEL=text-embedding-3-small
```

## Tunable config keys

| Key                        | Default                                          | Notes                                                   |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| `news_archive_enabled`     | `False`                                          | Master switch                                           |
| `news_archive_path`        | `~/.tradingagents/news_archive/chroma`           | Persistent on-disk store                                |
| `rag_embedding_provider`   | `"openai"`                                       | Shared with the decision-log RAG                        |
| `rag_embedding_model`      | `"text-embedding-3-small"`                       | Shared                                                  |

The provider/model are **shared** with the decision-log RAG by design:
running both upgrades on the same embedder keeps similarity scores
comparable across stores, and avoids running two embedding APIs in
parallel.

## Search semantics

`search_news_archive(query, ticker?, days_back?, curr_date?, limit=5)`

- `query` is free text. Embedding similarity is what ranks results.
- `ticker` (optional, case-insensitive) restricts to ticker-news for
  that symbol. Pass `None` for cross-ticker macro searches.
- `days_back` (default 90) restricts to recent articles only. Articles
  without a parsed `published_at` aren't filtered out, so the tool
  doesn't accidentally hide untimestamped sources.
- `curr_date` (optional, `yyyy-mm-dd`) is the look-ahead guard:
  back-tested runs must pass this so the tool never returns articles
  dated after the analysis date.
- Returns a markdown block in the same shape as `get_news` — the LLM
  can stitch archive results next to fresh fetches without
  special-casing the formatting.
- Returns a clearly-labelled placeholder string when the archive is
  disabled, empty, or has no matches.

`search_macro_archive(query, limit=3)`

- Searches the macro snapshot collection only.
- Returns one rendered macro block per match.

## When does this start being useful?

The archive bootstraps from zero — there's no migration of historical
JSON logs (the saved per-run state under `~/.tradingagents/logs/`
contains rendered markdown, not the structured article rows the
archive needs). The first analysis run after enabling the switch
populates the index with whatever it fetches; the second run is the
first one that can usefully retrieve from it.

In practice, after a few weeks of daily runs against the gold complex
the archive accumulates several hundred unique articles plus dozens
of macro snapshots — enough for queries like _"how did gold trade
last time real yields fell this fast?"_ to return useful comps.

## Failure semantics

The archive layer is held to the same contract as the decision-log
RAG: **a quality improvement, not a hard dependency**. Every error
path is caught and logged at WARNING. The most common failures and
their behaviour:

| Failure                                          | Behaviour                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------------ |
| `chromadb` not installed                         | Indexer never builds; record/search calls are no-ops                     |
| `OPENAI_API_KEY` missing (with `provider=openai`)| Init fails, archive disabled for the process                             |
| `news_archive_path` write-protected              | Init fails, archive disabled for the process                             |
| Embedding API down mid-run                       | Indexing fails for that batch, search returns `[news archive empty]`     |
| Bad article dict (missing title)                 | That row skipped; rest of the batch indexes normally                     |
| Re-fetch of an already-indexed article           | Upsert is idempotent on `(source, link or title)` — no duplicates        |

## Testing

```bash
pytest tests/test_news_archive.py
```

The 34 tests cover the dataclass adapter, archive index/search
semantics (incl. ticker/days_back/look-ahead filters), the lazy
facade (disabled no-op, enabled indexing, init failure), end-to-end
dataflow hooks (with mocked yfinance + RSS + macro_data), the two
tool wrappers, and the analyst tool-binding gating.

## Combining with the previous RAG layers

The three RAG knobs compose. A typical "fast + smart + persistent" config:

```bash
export TRADINGAGENTS_ANALYST_CONCURRENCY=4         # parallel analysts
export TRADINGAGENTS_RAG_ENABLED=true              # semantic past_context
export TRADINGAGENTS_NEWS_ARCHIVE_ENABLED=true     # news/macro corpus
```

In that mode:

- The analyst phase runs in parallel.
- The news/sentiment/market analysts can query the archive via tool
  calls during their loop.
- The `get_macro_data` and news fetches passively populate the archive
  for future runs.
- The Memory Retriever node (between Trader and risk debate) selects
  the Portfolio Manager's `past_context` by embedding similarity over
  the decision log.
