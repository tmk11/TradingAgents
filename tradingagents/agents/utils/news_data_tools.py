from langchain_core.tools import tool
from typing import Annotated, Optional
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[Optional[int], "Days to look back; omit to use the configured default"] = None,
    limit: Annotated[Optional[int], "Max articles to return; omit to use the configured default"] = None,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config

    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)



@tool
def get_gold_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[
        Optional[int], "Days to look back; defaults to global_news_lookback_days config"
    ] = None,
    limit: Annotated[
        Optional[int], "Max articles per source (Kitco + Mining.com); defaults to 10"
    ] = None,
) -> str:
    """
    Retrieve gold-specific news from Kitco and Mining.com.

    Use this when analysing gold-complex instruments (GLD, IAU, GC=F,
    XAUUSD=X, GDX, GDXJ, ^XAU, etc.). Returns a combined block with two
    clearly-labeled sections:

      - Kitco News: gold-focused desk covering spot price drivers,
        central-bank flows, sector commentary.
      - Mining.com: industry trade press, most relevant for miner ETFs
        and supply-side headlines (production, M&A, jurisdiction risk).

    Sits alongside the existing get_news / get_global_news tools — call
    those for ticker-specific or generic macro coverage, and this one
    for gold-native sources.

    Args:
        curr_date (str): Current trading date in yyyy-mm-dd format.
        look_back_days (int): How many days back to include.
        limit (int): Max items per source. Combined output may reach 2*limit.

    Returns:
        str: A formatted markdown block of recent gold-complex news.
    """
    # Late import keeps yfinance / langchain users who never touch
    # commodity mode from paying the import cost of urllib + email.
    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.gold_news import fetch_gold_macro_news

    cfg = get_config()
    if look_back_days is None:
        look_back_days = cfg.get("global_news_lookback_days", 7)
    if limit is None:
        # Default to 10 per source — matches global_news_article_limit
        # and keeps the combined block within a sensible token budget.
        limit = cfg.get("global_news_article_limit", 10)

    return fetch_gold_macro_news(curr_date, look_back_days=look_back_days, limit=limit)
