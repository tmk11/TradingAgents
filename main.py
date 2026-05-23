from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Gold Edition example: analyse the SPDR Gold Shares ETF (GLD).
# - ``selected_analysts`` omits ``"fundamentals"`` because gold has no
#   company-style fundamentals (this is also the new default in
#   TradingAgentsGraph, but spelling it out keeps the example explicit).
# - ``asset_type="commodity"`` rewires prompts (bull/bear research, risk
#   debators, instrument context) to gold-specific macro drivers.
# - To analyse gold futures or spot pairs instead, replace ``"GLD"``
#   with ``"GC=F"`` (front-month COMEX) or ``"XAUUSD=X"`` (spot).
ta = TradingAgentsGraph(
    selected_analysts=["market", "social", "news"],
    debug=True,
    config=config,
)

# forward propagate
_, decision = ta.propagate("GLD", "2026-05-22", asset_type="commodity")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
