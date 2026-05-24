# TradingAgents/graph/setup.py

from typing import Any, Callable, Dict, Optional
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .analyst_execution import build_analyst_execution_plan
from .analyst_subgraph import build_analyst_subgraph, create_parallel_analyst_node
from .conditional_logic import ConditionalLogic


# Node label for the optional semantic past_context retriever. Inserted
# between Trader and Aggressive Analyst so it runs once per analysis
# with full analyst + plan context available to drive the embedding
# query. Kept as a module-level constant so tests can assert topology.
MEMORY_RETRIEVER_NODE = "Memory Retriever"


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
        memory_retriever_node: Optional[Callable] = None,
    ):
        """Initialize with required components.

        Args:
            quick_thinking_llm / deep_thinking_llm: bound LangChain LLM
                clients for the two reasoning tiers.
            tool_nodes: mapping from analyst key (``market`` /
                ``social`` / ``news`` / ``fundamentals``) to the
                pre-built :class:`ToolNode` exposing that analyst's
                data tools.
            conditional_logic: provides the ``should_continue_*`` and
                ``should_continue_debate`` / ``_risk_analysis``
                routers.
            analyst_concurrency_limit: when ``> 1``, analysts are wired
                as parallel subgraphs (see
                :mod:`tradingagents.graph.analyst_subgraph`). Default
                ``1`` preserves the legacy sequential chain.
            memory_retriever_node: optional callable inserted between
                ``Trader`` and ``Aggressive Analyst`` that overwrites
                ``past_context`` based on now-available reports. When
                ``None`` the legacy edge ``Trader → Aggressive
                Analyst`` is preserved.
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit
        self.memory_retriever_node = memory_retriever_node

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def setup_graph(
        self, selected_analysts=["market", "social", "news"]
    ):
        """Set up and compile the agent workflow graph.

        Picks sequential or parallel analyst wiring based on
        ``analyst_concurrency_limit``. The downstream sections
        (researchers → research manager → trader → optional memory
        retriever → risk debate → portfolio manager) are identical in
        both modes — only the analyst phase changes.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Sentiment analyst (news + StockTwits + Reddit)
                - "news": News analyst
                - "fundamentals": Fundamentals analyst (equities only — the
                  Gold Edition default omits this because commodities have
                  no company-style fundamentals)
        """
        plan = build_analyst_execution_plan(
            selected_analysts,
            concurrency_limit=self.analyst_concurrency_limit,
        )

        analyst_factories = self._build_analyst_factories()
        downstream = self._build_downstream_nodes()

        if self.analyst_concurrency_limit > 1 and len(plan.specs) > 1:
            return self._wire_parallel(plan, analyst_factories, downstream)
        return self._wire_sequential(plan, analyst_factories, downstream)

    # ------------------------------------------------------------------
    # Shared scaffolding
    # ------------------------------------------------------------------

    def _build_analyst_factories(self) -> Dict[str, Callable]:
        """Map analyst key → zero-arg factory returning the node callable."""
        return {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

    def _build_downstream_nodes(self) -> Dict[str, Any]:
        """Pre-build all non-analyst nodes (researchers, manager, trader, risk, PM)."""
        return {
            "Bull Researcher": create_bull_researcher(self.quick_thinking_llm),
            "Bear Researcher": create_bear_researcher(self.quick_thinking_llm),
            "Research Manager": create_research_manager(self.deep_thinking_llm),
            "Trader": create_trader(self.quick_thinking_llm),
            "Aggressive Analyst": create_aggressive_debator(self.quick_thinking_llm),
            "Neutral Analyst": create_neutral_debator(self.quick_thinking_llm),
            "Conservative Analyst": create_conservative_debator(self.quick_thinking_llm),
            "Portfolio Manager": create_portfolio_manager(self.deep_thinking_llm),
        }

    def _add_downstream_nodes(self, workflow: StateGraph, downstream: Dict[str, Any]) -> None:
        for label, node in downstream.items():
            workflow.add_node(label, node)

    def _wire_post_research_section(self, workflow: StateGraph) -> None:
        """Edges from Bull/Bear → Research Manager → Trader → (retriever?) → risk debate → PM.

        Identical regardless of how the analyst phase is wired.
        """
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")

        # Optional memory retriever between Trader and the risk debate.
        # Inserted unconditionally when configured; the node itself
        # falls back to recency-based context when RAG is disabled.
        if self.memory_retriever_node is not None:
            workflow.add_node(MEMORY_RETRIEVER_NODE, self.memory_retriever_node)
            workflow.add_edge("Trader", MEMORY_RETRIEVER_NODE)
            workflow.add_edge(MEMORY_RETRIEVER_NODE, "Aggressive Analyst")
        else:
            workflow.add_edge("Trader", "Aggressive Analyst")

        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

    # ------------------------------------------------------------------
    # Sequential wiring (legacy — analyst_concurrency_limit == 1)
    # ------------------------------------------------------------------

    def _wire_sequential(
        self,
        plan,
        analyst_factories: Dict[str, Callable],
        downstream: Dict[str, Any],
    ):
        workflow = StateGraph(AgentState)

        # Analyst nodes + per-analyst Msg Clear and ToolNode siblings.
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        self._add_downstream_nodes(workflow, downstream)

        workflow.add_edge(START, plan.specs[0].agent_node)

        # Chain analysts in selection order.
        for i, spec in enumerate(plan.specs):
            workflow.add_conditional_edges(
                spec.agent_node,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [spec.tool_node, spec.clear_node],
            )
            workflow.add_edge(spec.tool_node, spec.agent_node)

            if i < len(plan.specs) - 1:
                workflow.add_edge(spec.clear_node, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(spec.clear_node, "Bull Researcher")

        self._wire_post_research_section(workflow)
        return workflow

    # ------------------------------------------------------------------
    # Parallel wiring (analyst_concurrency_limit > 1)
    # ------------------------------------------------------------------

    def _wire_parallel(
        self,
        plan,
        analyst_factories: Dict[str, Callable],
        downstream: Dict[str, Any],
    ):
        """Fan analysts out from START, join on Bull Researcher.

        Each analyst is wrapped in a compiled subgraph that owns its
        own ``messages`` channel (see
        :mod:`tradingagents.graph.analyst_subgraph`). The parent
        graph never sees inter-analyst tool calls, so concurrency is
        safe even with three or four analysts running at once.

        We do **not** add per-analyst ``Msg Clear`` nodes in this
        mode — there is no shared messages channel to clean. The
        legacy wire keys (``tools_market`` etc.) also disappear from
        the parent topology.
        """
        workflow = StateGraph(AgentState)

        # Build one subgraph wrapper node per selected analyst.
        for spec in plan.specs:
            analyst_node = analyst_factories[spec.key]()
            tool_node = self.tool_nodes[spec.key]
            sub = build_analyst_subgraph(analyst_node, tool_node)
            wrapper = create_parallel_analyst_node(sub, spec.report_key)
            workflow.add_node(spec.agent_node, wrapper)

        self._add_downstream_nodes(workflow, downstream)

        # Parallel fan-out from START. LangGraph schedules these
        # branches concurrently because they share a predecessor.
        for spec in plan.specs:
            workflow.add_edge(START, spec.agent_node)

        # Join on Bull Researcher: LangGraph waits until every parallel
        # branch has reached the same downstream node before firing
        # it, so the bull/bear debate gets a fully populated state.
        for spec in plan.specs:
            workflow.add_edge(spec.agent_node, "Bull Researcher")

        self._wire_post_research_section(workflow)
        return workflow
