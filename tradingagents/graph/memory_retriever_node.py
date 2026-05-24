"""Graph node: build a ticker-specific past_context for the Portfolio Manager.

In the legacy pipeline, ``past_context`` was injected into the initial
state once, before any analyst had run, using a recency-based filter on
the markdown decision log.  That timing is wrong twice over:

1. The query has no signal — only the ticker is known — so the
   selector can only fall back to recency, which often surfaces
   irrelevant prior runs.
2. By the time the Portfolio Manager actually reads ``past_context``,
   we have analyst reports, an investment plan, and a trader proposal
   — every one of those is rich enough to drive a useful retrieval.

This node closes that gap.  Wired between ``Trader`` and
``Aggressive Analyst`` (so it runs once per analysis, before the risk
debate begins), it builds a context-rich query from the now-available
state slice and calls :meth:`TradingMemoryLog.get_past_context_semantic`.

When ``rag_enabled`` is off the memory log silently falls back to its
recency-based selector, so wiring this node in unconditionally is safe.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


# Keys we sample to build the retrieval query. Order matters — we
# concatenate in this order so the most diagnostic fields lead.
_QUERY_FIELDS = (
    ("market_report", "Market report", 1500),
    ("sentiment_report", "Sentiment report", 1000),
    ("news_report", "News report", 1500),
    ("fundamentals_report", "Fundamentals report", 1000),
    ("investment_plan", "Research plan", 1500),
    ("trader_investment_plan", "Trader plan", 1000),
)


def build_retrieval_query(state: Dict[str, Any]) -> str:
    """Render an embedding-friendly query from the current pipeline state.

    Truncate each section to keep the query within typical embedding
    token budgets (~8k tokens for OpenAI's small model).  The
    truncation is per-section rather than global so a verbose market
    report can't crowd out the news section.
    """
    ticker = state.get("company_of_interest", "")
    asset_type = state.get("asset_type", "stock")
    trade_date = state.get("trade_date", "")

    parts = [
        f"Ticker: {ticker}",
        f"Asset type: {asset_type}",
        f"Date: {trade_date}",
    ]
    for key, label, limit in _QUERY_FIELDS:
        value = state.get(key)
        if not value:
            continue
        snippet = value[:limit]
        parts.append(f"{label}:\n{snippet}")
    return "\n\n".join(parts)


def create_memory_retriever_node(
    memory_log,
    n_same: int = 5,
    n_cross: int = 3,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a LangGraph node callable.

    The returned function reads ``state`` and returns a dict that
    overwrites ``past_context``.  All other state keys are left
    untouched, so this node composes cleanly with the existing
    ``MessagesState`` reducer behaviour.

    Args:
        memory_log: a :class:`TradingMemoryLog` instance.  Must expose
            ``get_past_context_semantic``; the legacy log already does.
        n_same: number of same-ticker hits to include.
        n_cross: number of cross-ticker reflections to include.
    """

    def memory_retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        if not ticker:
            return {"past_context": state.get("past_context", "")}
        query = build_retrieval_query(state)
        past_context = memory_log.get_past_context_semantic(
            query=query,
            ticker=ticker,
            n_same=n_same,
            n_cross=n_cross,
        )
        return {"past_context": past_context}

    return memory_retriever_node
