from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        asset_type = state.get("asset_type", "stock")

        # Asset-class branching: bear case framed in the drivers that
        # actually apply. For gold the dominant headwind is rising real
        # yields and a strong USD; for equities it's company-level
        # weakness and competitive risk; for crypto it's regulatory and
        # liquidity risk.
        if asset_type == "commodity":
            target_label = "gold position"
            fundamentals_label = (
                "Macro & monetary context (no company fundamentals apply for commodities)"
            )
            key_points_block = (
                "Key points to focus on:\n"
                "- Monetary headwinds: Fed staying restrictive, rising real yields"
                " (TIPS, 5y5y), strengthening DXY, hawkish guidance from major"
                " central banks — all raise the opportunity cost of holding"
                " a non-yielding asset.\n"
                "- Demand fragility: ETF outflows (GLD/IAU), softening jewelry"
                " demand if China / India growth stalls or import duties rise,"
                " central-bank purchase pace decelerating.\n"
                "- Positioning risk: stretched speculative longs in COMEX gold"
                " futures, crowded trade, potential for a sharp unwind.\n"
                "- Cooling inflation: falling CPI prints and breakevens reduce"
                " the inflation-hedge bid.\n"
                "- Risk-on regimes: equity rallies and tight credit spreads"
                " typically pull capital away from safe-haven gold.\n"
                "- Bull Counterpoints: critically refute the bull case using"
                " specific macro data and positioning evidence.\n"
                "- Engagement: argue conversationally, engage the bull directly."
            )
        elif asset_type == "crypto":
            target_label = "crypto asset"
            fundamentals_label = (
                "Asset fundamentals report (may be unavailable for crypto)"
            )
            key_points_block = (
                "Key points to focus on:\n"
                "- Regulatory and policy risk: enforcement, tax treatment, banking"
                " access.\n"
                "- Liquidity and counterparty risk: exchange solvency, stablecoin"
                " peg risk, on-chain liquidity fragmentation.\n"
                "- Tokenomics headwinds: supply unlocks, dilution from staking"
                " issuance, fee compression.\n"
                "- Macro headwinds: rising real yields, strong USD, risk-off"
                " regimes that compress speculative-asset valuations.\n"
                "- Bull Counterpoints: critically analyze and rebut the bull case.\n"
                "- Engagement: argue conversationally and refute the bull directly."
            )
        else:
            target_label = "stock"
            fundamentals_label = "Company fundamentals report"
            key_points_block = (
                "Key points to focus on:\n"
                "- Risks and Challenges: highlight factors like market saturation,"
                " financial instability, or macroeconomic threats that could hinder"
                " the stock's performance.\n"
                "- Competitive Weaknesses: emphasize vulnerabilities such as weaker"
                " market positioning, declining innovation, or threats from"
                " competitors.\n"
                "- Negative Indicators: use evidence from financial data, market"
                " trends, or recent adverse news to support your position.\n"
                "- Bull Counterpoints: critically analyze the bull argument with"
                " specific data and sound reasoning, exposing weaknesses or"
                " over-optimistic assumptions.\n"
                "- Engagement: present your argument in a conversational style,"
                " directly engaging with the bull analyst's points and debating"
                " effectively rather than simply listing facts."
            )

        prompt = f"""You are a Bear Analyst making the case against the {target_label}. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators relevant to this asset class. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

{key_points_block}

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the {target_label}.
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
