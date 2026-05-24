# Optional upgrades: RAG memory and parallel analysts

This page documents two opt-in upgrades that were added on top of the
core LangGraph pipeline:

1. **Semantic memory (RAG)** — replace the recency-based
   `past_context` selector with embedding similarity over the
   decision log, and feed the Portfolio Manager analyses that are
   actually relevant to the current market regime.
2. **Parallel analysts** — fan the four analyst nodes out from
   `START` instead of running them in sequence, cutting wall time
   roughly N-fold during the analyst phase.

Both features are off by default. Existing runs, tests, and saved
state are unaffected unless you set the flags below.

---

## 1. Semantic memory (RAG)

### What it does

The legacy memory log keeps every resolved decision in
`~/.tradingagents/memory/trading_memory.md` and surfaces the last 5
same-ticker entries plus the last 3 cross-ticker reflections to the
Portfolio Manager. With more than a few months of history, recency
quickly stops correlating with relevance — a hawkish-Fed decision
from January is included in a dovish-Fed analysis run today, while
the actually-comparable dovish-Fed analysis from six months ago gets
pruned.

When `rag_enabled = True`, every resolved entry is also indexed into a
local Chroma vector store. A new `Memory Retriever` graph node runs
between the Trader and the risk debate, builds a query from the
analyst reports + research plan + trader proposal, and selects past
context by **embedding similarity** rather than recency.

### Architecture

```
┌──────────────────────┐                  ┌────────────────────────┐
│ trading_memory.md    │ ──── on write ─► │ Chroma vector store    │
│ (markdown log,       │                  │ ~/.tradingagents/      │
│  always written)     │ ◄─ recency ──    │   memory/chroma/       │
└──────────────────────┘    fallback      └────────────────────────┘
                                                     │
                                                     ▼
            Trader ─► Memory Retriever ─► Aggressive Analyst ─► …
                          (semantic
                           past_context)
```

Files of interest:

| File | Role |
|---|---|
| `tradingagents/retrieval/embeddings.py` | `OpenAIEmbedder` and `FakeEmbedder` + `create_embedder` factory |
| `tradingagents/retrieval/vector_store.py` | Chroma adapter (`MemoryVectorStore`, cosine distance) |
| `tradingagents/retrieval/memory_retriever.py` | `SemanticMemoryRetriever` (index, search, format) |
| `tradingagents/agents/utils/memory.py` | `TradingMemoryLog.get_past_context_semantic` + indexing hooks |
| `tradingagents/graph/memory_retriever_node.py` | LangGraph node that calls the retriever from inside the pipeline |
| `tradingagents/graph/setup.py` | Inserts the retriever node between `Trader` and `Aggressive Analyst` |

### Enabling

Either flip the config key in code:

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
config["rag_enabled"] = True
ta = TradingAgentsGraph(config=config)
ta.propagate("GLD", "2026-05-22", asset_type="commodity")
```

Or set env vars (picked up by `_apply_env_overrides`):

```bash
export TRADINGAGENTS_RAG_ENABLED=true
export TRADINGAGENTS_RAG_EMBEDDING_PROVIDER=openai      # default
export TRADINGAGENTS_RAG_EMBEDDING_MODEL=text-embedding-3-small
```

### Tunable config keys

| Key | Default | Notes |
|---|---|---|
| `rag_enabled` | `False` | Master switch |
| `rag_vector_store_path` | `~/.tradingagents/memory/chroma` | Persistent on-disk store |
| `rag_embedding_provider` | `"openai"` | `"openai"` or `"fake"` |
| `rag_embedding_model` | `"text-embedding-3-small"` | OpenAI model name |
| `rag_n_same_ticker` | `5` | Same-ticker hits in `past_context` |
| `rag_n_cross_ticker` | `3` | Cross-ticker reflections in `past_context` |

The `"fake"` provider is a deterministic hash-based bag-of-words
embedder. It needs no network or API key and is what the test
suite uses, but its retrieval quality is far below a real embedding
model — use it only for offline smoke runs.

### Bootstrap behaviour

The first time RAG is enabled on an existing log, the memory log
scans `trading_memory.md` and indexes every resolved entry that
isn't already in the vector store. There's nothing to migrate — just
turn the flag on and the next run is RAG-enabled.

### Failure semantics

RAG is a **quality improvement, not a hard dependency**. Every error
path inside the retriever is caught and logged at WARNING; the
memory log silently falls back to the recency-based selector. A
trading run never crashes because chromadb is missing, the OpenAI
key isn't set, or an embedding API is unreachable.

### Testing

```bash
pytest tests/test_retrieval.py
```

The fake embedder makes these tests fully offline.

---

## 2. Parallel analysts

### What it does

The four analyst types — Market, Sentiment, News, Fundamentals — run
in selection order in the legacy pipeline, with `Msg Clear` nodes
between them so each one starts with a fresh `messages` channel.
Most of an analyst's wall time is spent waiting on LLM completions
and HTTP tool calls — pure I/O — so running them concurrently scales
roughly linearly with N.

When `analyst_concurrency_limit > 1` and at least two analysts are
selected, each analyst is wrapped in a **compiled subgraph** that
owns its own `messages` channel, and the parent graph fans out from
`START` to all of them in parallel. They join on `Bull Researcher`
(LangGraph waits for every parallel branch before firing a
common-successor node), so the bull/bear debate sees a fully
populated state.

### Why subgraphs

Sharing `messages` between parallel analysts breaks two ways:

1. Analyst A's `tool_calls` and Analyst B's `tool_calls` interleave
   on the parent channel. The subsequent `ToolMessage` from Analyst
   A's tool node would land in Analyst B's prompt context, polluting
   reasoning and (with strict providers) producing tool-call-id
   mismatches.
2. The legacy `Msg Clear *` nodes can't safely drain a shared
   channel mid-fan-out — they would erase a sibling analyst's
   in-flight tool calls.

A subgraph encapsulates the analyst+ToolNode loop with its own
`messages` channel and returns only the analyst's report key to the
parent. Cross-pollution is structurally impossible.

```
                  ┌─ subgraph(market) ──┐
                  ├─ subgraph(social) ──┤
START ──┬────────►├─ subgraph(news) ────┤────► Bull Researcher
        │         └─ subgraph(funds) ───┘
        │
        │ (LangGraph schedules siblings concurrently;
        │  joins on Bull Researcher, which has a
        │  multi-edge predecessor.)
```

### Enabling

```python
config = DEFAULT_CONFIG.copy()
config["analyst_concurrency_limit"] = 4
```

Or:

```bash
export TRADINGAGENTS_ANALYST_CONCURRENCY=4
```

The downstream pipeline (researchers → research manager → trader →
risk debate → portfolio manager) is identical regardless of mode.

### Backward compatibility

- `analyst_concurrency_limit = 1` (default) keeps the legacy
  sequential wiring exactly as before, including the `Msg Clear *`
  and `tools_*` nodes the CLI tracks for progress.
- With concurrency > 1 but only one analyst selected, the setup
  also stays sequential — there's nothing to parallelise.

### Files of interest

| File | Role |
|---|---|
| `tradingagents/graph/analyst_subgraph.py` | `build_analyst_subgraph`, `create_parallel_analyst_node` |
| `tradingagents/graph/setup.py` | `_wire_sequential` vs `_wire_parallel` branching |

### Testing

```bash
pytest tests/test_parallel_analysts.py
```

---

## Combining them

The two upgrades compose. A typical "fast + smart" config:

```bash
export TRADINGAGENTS_ANALYST_CONCURRENCY=4
export TRADINGAGENTS_RAG_ENABLED=true
```

In that mode the analyst phase runs in parallel, the Memory
Retriever node runs once between the trader and the risk debate
using the now-populated reports as the embedding query, and the
Portfolio Manager receives semantically relevant past context.
