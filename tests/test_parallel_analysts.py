"""Tests for parallel-analyst graph wiring and the memory-retriever node.

These tests focus on graph **topology** (which nodes exist, which
edges connect them) rather than execution semantics. Running the
full pipeline would require a live LLM client; topology assertions
are enough to lock in the wiring contract that
:class:`GraphSetup` provides.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.memory_retriever_node import (
    build_retrieval_query,
    create_memory_retriever_node,
)
from tradingagents.graph.setup import MEMORY_RETRIEVER_NODE, GraphSetup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_setup(concurrency_limit: int, with_retriever: bool = False):
    """Build a GraphSetup with pure mocks so the wiring code runs without LLMs.

    The analyst factories and downstream node creators are patched
    inside :class:`GraphSetup` itself, so we don't need to spin up a
    real LangChain client. We rely on the ``ToolNode`` mock having
    the same shape the wiring expects (it's just bound to nodes; the
    graph isn't executed).
    """
    quick = MagicMock(name="quick_llm")
    deep = MagicMock(name="deep_llm")
    tool_nodes = {
        "market": MagicMock(name="tools_market"),
        "social": MagicMock(name="tools_social"),
        "news": MagicMock(name="tools_news"),
        "fundamentals": MagicMock(name="tools_fundamentals"),
    }
    cl = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    retriever_node = None
    if with_retriever:
        # The node is just a callable that returns a dict; we don't
        # need a real memory log here.
        retriever_node = lambda state: {"past_context": ""}

    return GraphSetup(
        quick_thinking_llm=quick,
        deep_thinking_llm=deep,
        tool_nodes=tool_nodes,
        conditional_logic=cl,
        analyst_concurrency_limit=concurrency_limit,
        memory_retriever_node=retriever_node,
    )


def _node_names(workflow):
    """Pull the StateGraph's registered node names without compiling.

    LangGraph stores nodes on the internal ``nodes`` dict; reading it
    is safe because we never mutate it.
    """
    return set(workflow.nodes.keys())


# ---------------------------------------------------------------------------
# Sequential mode (default — concurrency_limit == 1)
# ---------------------------------------------------------------------------


class TestSequentialWiring:
    def test_default_concurrency_uses_sequential(self):
        setup = _make_graph_setup(concurrency_limit=1)
        wf = setup.setup_graph(["market", "social", "news"])
        names = _node_names(wf)
        # Sequential mode keeps the legacy per-analyst Msg Clear and
        # tools_* siblings.
        for label in (
            "Market Analyst", "Sentiment Analyst", "News Analyst",
            "Msg Clear Market", "Msg Clear Sentiment", "Msg Clear News",
            "tools_market", "tools_social", "tools_news",
            "Bull Researcher", "Bear Researcher", "Research Manager",
            "Trader", "Aggressive Analyst", "Conservative Analyst",
            "Neutral Analyst", "Portfolio Manager",
        ):
            assert label in names, f"missing node: {label}"

    def test_sequential_no_memory_retriever_by_default(self):
        setup = _make_graph_setup(concurrency_limit=1, with_retriever=False)
        wf = setup.setup_graph(["market", "social", "news"])
        assert MEMORY_RETRIEVER_NODE not in _node_names(wf)

    def test_sequential_with_memory_retriever_inserts_node(self):
        setup = _make_graph_setup(concurrency_limit=1, with_retriever=True)
        wf = setup.setup_graph(["market", "social", "news"])
        assert MEMORY_RETRIEVER_NODE in _node_names(wf)


# ---------------------------------------------------------------------------
# Parallel mode (concurrency_limit > 1)
# ---------------------------------------------------------------------------


class TestParallelWiring:
    def test_parallel_drops_msg_clear_and_tool_nodes(self):
        setup = _make_graph_setup(concurrency_limit=4)
        wf = setup.setup_graph(["market", "social", "news", "fundamentals"])
        names = _node_names(wf)
        # The four analysts are present as parent-graph wrappers.
        for label in (
            "Market Analyst", "Sentiment Analyst",
            "News Analyst", "Fundamentals Analyst",
        ):
            assert label in names, f"missing analyst wrapper: {label}"
        # Per-analyst tool / clear siblings live inside the subgraphs
        # now, so the parent graph should NOT carry them.
        for legacy_label in (
            "Msg Clear Market", "Msg Clear Sentiment", "Msg Clear News",
            "Msg Clear Fundamentals",
            "tools_market", "tools_social", "tools_news", "tools_fundamentals",
        ):
            assert legacy_label not in names, (
                f"parallel mode must not register legacy node {legacy_label!r} "
                "in the parent graph"
            )

    def test_parallel_keeps_downstream_section(self):
        setup = _make_graph_setup(concurrency_limit=2)
        wf = setup.setup_graph(["market", "news"])
        names = _node_names(wf)
        # The post-research section is identical in both modes.
        for label in (
            "Bull Researcher", "Bear Researcher", "Research Manager",
            "Trader", "Aggressive Analyst", "Conservative Analyst",
            "Neutral Analyst", "Portfolio Manager",
        ):
            assert label in names, f"missing downstream node: {label}"

    def test_parallel_single_analyst_falls_back_to_sequential(self):
        # With only one analyst there's nothing to parallelise; the
        # setup must keep the sequential wiring (still has tools_market
        # and Msg Clear Market).
        setup = _make_graph_setup(concurrency_limit=4)
        wf = setup.setup_graph(["market"])
        names = _node_names(wf)
        assert "tools_market" in names
        assert "Msg Clear Market" in names

    def test_parallel_with_retriever_inserts_node(self):
        setup = _make_graph_setup(concurrency_limit=2, with_retriever=True)
        wf = setup.setup_graph(["market", "news"])
        assert MEMORY_RETRIEVER_NODE in _node_names(wf)


# ---------------------------------------------------------------------------
# Memory retriever node — query construction and state output
# ---------------------------------------------------------------------------


class TestMemoryRetrieverNode:
    def test_build_query_includes_ticker_and_reports(self):
        state = {
            "company_of_interest": "GLD",
            "asset_type": "commodity",
            "trade_date": "2026-05-22",
            "market_report": "DXY soft, real yields falling, gold breakout.",
            "news_report": "Fed signals dovish pivot.",
            "investment_plan": "Recommendation: Overweight.",
        }
        q = build_retrieval_query(state)
        assert "Ticker: GLD" in q
        assert "Asset type: commodity" in q
        assert "DXY soft" in q
        assert "Fed signals dovish" in q
        assert "Overweight" in q

    def test_build_query_skips_empty_sections(self):
        # A run with only the market report present must still build
        # a non-empty query, but must not contain stub headers for
        # missing sections.
        state = {
            "company_of_interest": "NVDA",
            "asset_type": "stock",
            "trade_date": "2026-05-22",
            "market_report": "Trend up, momentum strong.",
        }
        q = build_retrieval_query(state)
        assert "Market report" in q
        assert "Sentiment report" not in q  # not provided
        assert "News report" not in q

    def test_build_query_truncates_per_section(self):
        # 5000 chars → must be capped at 1500 for market_report.
        long_text = "x" * 5000
        state = {
            "company_of_interest": "GLD",
            "asset_type": "commodity",
            "trade_date": "2026-05-22",
            "market_report": long_text,
        }
        q = build_retrieval_query(state)
        # Total query body for market section ≤ 1500 + headers.
        market_idx = q.index("Market report:")
        market_section = q[market_idx:]
        assert len(market_section) < 2000

    def test_node_writes_past_context(self, tmp_path):
        # Wire a real (recency-based) memory log through the node and
        # verify the node returns the right state delta.
        log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
        log.store_decision("GLD", "2026-01-10", "Rating: Buy\nBuy GLD on dovish Fed.")
        log.batch_update_with_outcomes([{
            "ticker": "GLD", "trade_date": "2026-01-10",
            "raw_return": 0.03, "alpha_return": 0.01, "holding_days": 5,
            "reflection": "Correct.",
        }])

        node = create_memory_retriever_node(log, n_same=5, n_cross=3)
        out = node({
            "company_of_interest": "GLD",
            "asset_type": "commodity",
            "trade_date": "2026-05-22",
            "market_report": "DXY soft, real yields falling.",
        })
        assert "past_context" in out
        assert "Past analyses of GLD" in out["past_context"]

    def test_node_handles_missing_ticker_gracefully(self):
        log = MagicMock()
        log.get_past_context_semantic.return_value = "should not be called"
        node = create_memory_retriever_node(log)
        out = node({"past_context": "preexisting"})
        # Without a ticker we passthrough whatever was already there
        # rather than calling the retriever with garbage.
        assert out == {"past_context": "preexisting"}
        log.get_past_context_semantic.assert_not_called()
