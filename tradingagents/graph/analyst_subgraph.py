"""Per-analyst subgraphs that isolate ``messages`` for parallel execution.

Why subgraphs
-------------

In the sequential pipeline every analyst shares the parent's
``messages`` channel: they each emit tool calls that route to a
shared ``ToolNode`` and then loop back to the same analyst, with a
``Msg Clear`` node draining the channel between analysts so the next
one starts on a clean slate.

That model breaks under parallel fan-out.  If two analysts run at
the same time and both append ``AIMessage(tool_calls=[...])`` to the
parent ``messages`` channel, then a ``ToolMessage`` from the news
analyst's tool would land in the market analyst's prompt context
(and vice versa) — confusing the LLM and, in some providers,
producing tool_call_id mismatches that raise.

The fix is to package each analyst (analyst node + its ToolNode +
its self-loop) into a **compiled subgraph** with its own
``MessagesState``.  The parent graph treats every analyst as an
atomic node that consumes the read-only parent state slice
(ticker / date / asset_type) and returns only its
``*_report`` field.  Messages stay private to the subgraph; the
parent's ``messages`` channel is never written to from analysts.

Because LangGraph schedules nodes with edges from the same
predecessor concurrently, this delivers true wall-time speedup for
the I/O-bound analyst phase (LLM + HTTP tool calls release the GIL).
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AnalystSubState(TypedDict, total=False):
    """Subgraph-local state.

    Mirrors the parent's :class:`AgentState` only for the keys an
    analyst genuinely reads or writes.  The ``messages`` field uses
    LangGraph's built-in ``add_messages`` reducer so tool-call
    chains accumulate the same way they always did — except scoped
    to this subgraph.
    """

    messages: Annotated[list, add_messages]
    company_of_interest: str
    asset_type: str
    trade_date: str
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str


def _has_tool_calls(state: Dict[str, Any]) -> str:
    """Conditional edge: keep looping while the last message has tool calls.

    Identical semantics to the legacy
    :class:`tradingagents.graph.conditional_logic.ConditionalLogic`
    routers, but inlined here because subgraph node names are local
    (``analyst`` / ``tools``) — the parent's ``tools_market`` /
    ``Msg Clear Market`` labels don't apply.
    """
    messages = state.get("messages") or []
    if not messages:
        return END
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END


def build_analyst_subgraph(analyst_node, tool_node):
    """Compile a self-contained subgraph: analyst ↔ tools loop → END.

    Args:
        analyst_node: the closure returned by
            ``create_market_analyst`` / ``create_sentiment_analyst`` /
            etc.  Reads ``state['messages']`` and returns a dict that
            includes ``messages`` and (when finished) the analyst's
            report key.
        tool_node: the matching ``ToolNode`` (e.g.
            ``tool_nodes['market']``) — just the data tools, no
            LLM logic.

    Returns the compiled :class:`langgraph.graph.state.CompiledStateGraph`.
    """
    sub = StateGraph(AnalystSubState)
    sub.add_node("analyst", analyst_node)
    sub.add_node("tools", tool_node)
    sub.add_edge(START, "analyst")
    sub.add_conditional_edges(
        "analyst",
        _has_tool_calls,
        {"tools": "tools", END: END},
    )
    sub.add_edge("tools", "analyst")
    return sub.compile()


def create_parallel_analyst_node(
    subgraph,
    report_key: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Wrap a compiled analyst subgraph as a single parent-graph node.

    The wrapper does three things:

    1. Project the parent state into the subgraph's smaller schema,
       seeding ``messages`` with a fresh ``("human", ticker)`` so the
       subgraph starts clean (matches the legacy
       :class:`Propagator.create_initial_state` shape).
    2. Invoke the compiled subgraph synchronously — LangGraph's
       parallel scheduler runs *this* wrapper concurrently with its
       siblings, so the inner ``invoke`` blocking is fine.
    3. Return *only* ``{report_key: ...}`` to the parent so we
       don't leak the subgraph's local ``messages`` upward (which
       would re-introduce the cross-analyst pollution the subgraph
       exists to prevent).
    """

    def analyst_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        sub_state: Dict[str, Any] = {
            "messages": [("human", ticker)],
            "company_of_interest": ticker,
            "asset_type": state.get("asset_type", "stock"),
            "trade_date": state.get("trade_date", ""),
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
        }
        result = subgraph.invoke(sub_state)
        return {report_key: result.get(report_key, "")}

    return analyst_wrapper
