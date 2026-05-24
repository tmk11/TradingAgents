from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_gold_news,
    get_language_instruction,
    get_news,
    search_news_archive,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        # Wording adapts to asset class: stocks have company-specific news,
        # commodities (gold) lean on macro headlines, crypto sits in
        # between.
        if asset_type == "stock":
            asset_label = "company"
        elif asset_type == "commodity":
            asset_label = "commodity / gold-complex"
        else:
            asset_label = "asset"
        instrument_context = build_instrument_context(
            state["company_of_interest"], asset_type
        )

        # Tool selection is asset-aware. Commodity (gold) runs additionally
        # get ``get_gold_news`` — Kitco + Mining.com RSS — because the
        # default yfinance/Alpha Vantage pipelines undercover bullion and
        # precious-metals industry coverage. Equity and crypto runs see
        # the original tool set, so behaviour is unchanged for them.
        tools = [
            get_news,
            get_global_news,
        ]
        if asset_type == "commodity":
            tools.append(get_gold_news)
        # When the news/macro archive is enabled, expose the semantic
        # search tool so the analyst can query historical context
        # accumulated from prior runs. Off by default; gated on the
        # active config rather than asset_type because the archive is
        # useful for every asset class.
        if get_config().get("news_archive_enabled"):
            tools.append(search_news_archive)

        gold_focus_addendum = (
            ""
            if asset_type != "commodity"
            else (
                " Because the instrument is part of the gold complex, weight"
                " global macro news heavily: Federal Reserve and major"
                " central-bank policy moves, real-yield direction (10Y TIPS,"
                " 5y5y forwards), DXY / US dollar momentum, central-bank"
                " gold purchases and reserve diversification, geopolitical"
                " risk premia (war, sanctions, trade conflict), inflation"
                " surprises, and physical demand from China and India."
                " Ticker-specific news is typically thin for futures and"
                " spot pairs — supplement with global news so the report"
                " reflects the actual price drivers."
                " Always call get_gold_news at least once for"
                " bullion-native coverage from Kitco and Mining.com;"
                " these sources surface central-bank flows, ETF holdings,"
                " and miner / supply-side headlines that Yahoo Finance"
                " alone tends to miss."
            )
        )

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + gold_focus_addendum
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
