"""LangChain tool wrapper for the Gold-Edition macro data fetcher.

Lives in its own module so the import surface stays narrow — the
fetcher pulls in yfinance + urllib + csv, which we don't want to load
on equity / crypto runs that never touch commodity mode.
"""

from typing import Annotated, Optional

from langchain_core.tools import tool


@tool
def get_macro_data(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_days: Annotated[
        Optional[int],
        "Days of history to fetch; defaults to 90 (about a quarter)",
    ] = None,
) -> str:
    """
    Retrieve gold-driver macro time-series data.

    Use this when analysing gold-complex instruments. Returns a single
    markdown block summarising the canonical gold drivers — real
    yields, USD strength, breakeven inflation, VIX, Fed balance sheet
    — with latest value, 1d/1w/1m percentage change, and min/max in
    the lookback window for each series.

    Two providers feed the block:

      - yfinance (always loads): ^TNX 10Y nominal yield, DX-Y.NYB DXY,
        ^VIX, TIP TIPS ETF, GC=F gold futures.
      - FRED public CSV (best-effort): DFII10 10Y real yield, T10YIE
        10Y breakeven inflation, WALCL Fed total assets, DTWEXBGS
        broad trade-weighted USD.

    Sits alongside get_news / get_global_news / get_gold_news — call
    those for narrative coverage, and this one for hard numbers.

    Args:
        curr_date (str): Current trading date in yyyy-mm-dd format.
        lookback_days (int): How many days back to compute the window
            stats. Defaults to 90 (~1 quarter).

    Returns:
        str: Formatted markdown block with one section per series.
    """
    # Late import keeps yfinance / langchain users who never touch
    # commodity mode from paying the import cost of the macro module.
    from tradingagents.dataflows.macro_data import fetch_gold_macro_data

    if lookback_days is None:
        lookback_days = 90
    return fetch_gold_macro_data(curr_date, lookback_days=lookback_days)
