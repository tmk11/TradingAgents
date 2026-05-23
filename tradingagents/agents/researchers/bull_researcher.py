from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        asset_type = state.get("asset_type", "stock")

        # Asset-class branching: each branch reframes the bull case in
        # the language and drivers that actually apply to that
        # instrument. Equity-style "growth potential / competitive
        # moat" arguments don't move bullion; gold-style "real yields
        # falling / central-bank buying" doesn't move equities.
        if asset_type == "commodity":
            target_label = "gold position"
            fundamentals_label = (
                "Macro & monetary context (no company fundamentals apply for commodities)"
            )
            key_points_block = (
                "Key points to focus on:\n"
                "- Monetary tailwinds: Fed pivoting toward cuts, falling real yields"
                " (10Y TIPS, 5y5y forwards), softer DXY, dovish guidance from major"
                " central banks.\n"
                "- Structural demand: net central-bank gold purchases, EM reserve"
                " diversification away from USD, sustained physical demand from"
                " China & India (jewelry, official-sector, household savings).\n"
                "- Risk-on-risk-off: geopolitical escalation, sanctions risk,"
                " sovereign credit concerns — all of which historically drive"
                " safe-haven flows into gold.\n"
                "- Flow signals: positive ETF holdings trend (GLD/IAU), futures"
                " positioning skew, premia on physical bars/coins.\n"
                "- Inflation hedge: rising or sticky inflation expectations and"
                " breakevens that erode real returns on cash and bonds.\n"
                "- Bear Counterpoints: critically refute the bear case using"
                " specific macro data, not generalities.\n"
                "- Engagement: present your argument conversationally, engaging"
                " directly with the bear analyst's points."
            )
        elif asset_type == "crypto":
            target_label = "crypto asset"
            fundamentals_label = (
                "Asset fundamentals report (may be unavailable for crypto)"
            )
            key_points_block = (
                "Key points to focus on:\n"
                "- Adoption and network effects: active addresses, on-chain volume,"
                " developer activity, integration into payment / DeFi / institutional"
                " rails.\n"
                "- Tokenomics: supply schedule, halving / burn dynamics, staking"
                " yields where relevant.\n"
                "- Liquidity and flows: ETF / spot inflows, exchange balances,"
                " stablecoin float.\n"
                "- Macro tailwinds: dovish policy, weak USD, risk-on regimes.\n"
                "- Bear Counterpoints: critically analyze the bear argument with"
                " specific data and sound reasoning.\n"
                "- Engagement: argue conversationally and refute the bear directly."
            )
        else:
            target_label = "stock"
            fundamentals_label = "Company fundamentals report"
            key_points_block = (
                "Key points to focus on:\n"
                "- Growth Potential: highlight the company's market opportunities,"
                " revenue projections, and scalability.\n"
                "- Competitive Advantages: emphasize factors like unique products,"
                " strong branding, or dominant market positioning.\n"
                "- Positive Indicators: use financial health, industry trends, and"
                " recent positive news as evidence.\n"
                "- Bear Counterpoints: critically analyze the bear argument with"
                " specific data and sound reasoning, addressing concerns thoroughly"
                " and showing why the bull perspective holds stronger merit.\n"
                "- Engagement: present your argument in a conversational style,"
                " engaging directly with the bear analyst's points and debating"
                " effectively rather than just listing data."
            )

        prompt = f"""You are a Bull Analyst advocating for the {target_label}. Your task is to build a strong, evidence-based case emphasizing the supportive drivers and positive market indicators relevant to this asset class. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

{key_points_block}

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
