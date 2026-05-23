from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str, asset_type: str = "stock") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    if asset_type == "crypto":
        instrument_label = "asset"
        extra_hint = (
            " Treat it as a crypto asset rather than a company, and do not assume"
            " company fundamentals are available."
        )
    elif asset_type == "commodity":
        # Gold Edition: when the ticker maps to the gold complex (futures,
        # spot pair, ETF, or miner index), explicitly steer the agents
        # away from company-style fundamentals — those tools either return
        # empty strings or, worse, misleading ETF-trust accounting that
        # the LLM may over-interpret. The drivers listed below are the
        # canonical macro inputs that move bullion.
        instrument_label = "commodity"
        extra_hint = (
            " Treat it as a precious-metal commodity rather than a company."
            " Company-style fundamentals (earnings, balance sheet, P/E,"
            " dividend yield, profit margin) do not apply and should not"
            " be invented. Anchor your analysis in the drivers that move"
            " gold prices: USD strength (DXY), real yields (10Y TIPS,"
            " 5y5y forwards), Federal Reserve and major central-bank policy"
            " stance, central-bank gold purchases / reserve diversification,"
            " geopolitical risk premia, inflation expectations, ETF flows"
            " (GLD/IAU holdings), futures positioning, and physical demand"
            " from China and India."
            " If the ticker is a futures contract (e.g. ``GC=F``), an ETF"
            " (e.g. ``GLD``, ``IAU``), a spot pair (``XAUUSD=X``), or a"
            " miner index (``GDX``, ``^XAU``), preserve that exact symbol"
            " and note the wrapper's specific quirks (roll yield for"
            " futures, expense ratio for ETFs, operational leverage for"
            " miners) where relevant."
        )
    else:
        instrument_label = "instrument"
        extra_hint = ""
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )


def build_risk_fundamentals_block(fundamentals_report: str, asset_type: str = "stock") -> str:
    """Render the fundamentals/macro reference block for risk-debate prompts.

    For equities the section heading stays "Company Fundamentals Report"
    so existing behaviour is preserved. For crypto and commodity the
    equity-style fundamentals report is typically empty (the
    Fundamentals Analyst is auto-disabled), so the label is rephrased
    to "Macro & Monetary Context" / "Asset Fundamentals Report" and the
    section is omitted entirely if the upstream string is empty —
    keeping misleading ``Company Fundamentals Report: `` stubs out of
    the prompt.
    """
    if asset_type == "commodity":
        if not fundamentals_report:
            return ""
        return f"Macro & Monetary Context: {fundamentals_report}\n"
    if asset_type == "crypto":
        if not fundamentals_report:
            return ""
        return f"Asset Fundamentals Report: {fundamentals_report}\n"
    return f"Company Fundamentals Report: {fundamentals_report}\n"

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
