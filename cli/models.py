from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    # Wire value stays "social" for saved-config and string-keyed-caller
    # back-compat; the user-facing label is "Sentiment Analyst".
    SOCIAL = "social"
    NEWS = "news"
    # Kept in the enum for back-compat with tests / imports, but the
    # Gold Edition fork no longer surfaces this analyst in the CLI:
    # gold and other commodities have no company-style fundamentals.
    FUNDAMENTALS = "fundamentals"


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    # New asset class introduced by the Gold Edition fork. Triggered when
    # the user enters a ticker that maps to gold (futures, spot pair, ETF,
    # or miner index). Routing logic auto-removes the Fundamentals Analyst
    # for this asset type and rewires prompts / data sources to gold-
    # specific macro drivers (USD, real yields, central-bank flows, etc.).
    COMMODITY = "commodity"
