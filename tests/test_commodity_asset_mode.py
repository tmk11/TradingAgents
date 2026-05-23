"""Gold Edition: tests for the ``commodity`` asset_type pipeline.

Mirrors the structure of ``test_crypto_asset_mode.py`` but covers the
new gold-complex routing: detection of common gold tickers, automatic
removal of the Fundamentals Analyst, asset-aware instrument context,
and conditional fundamentals-block rendering in the risk debate.
"""

import unittest

from cli.models import AnalystType, AssetType
from cli.utils import GOLD_TICKERS, detect_asset_type, filter_analysts_for_asset_type
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_risk_fundamentals_block,
)
from tradingagents.graph.propagation import Propagator


class CommodityAssetDetectionTests(unittest.TestCase):
    """``detect_asset_type`` should classify the gold complex as commodity."""

    def test_detects_etf_tickers(self):
        for ticker in ("GLD", "IAU", "SGOL", "GDX"):
            self.assertEqual(
                detect_asset_type(ticker),
                AssetType.COMMODITY,
                msg=f"{ticker} should map to commodity",
            )

    def test_detects_futures_and_spot_pairs(self):
        # Futures and spot pairs use ``=`` notation, which the equity
        # path would otherwise leave alone.
        self.assertEqual(detect_asset_type("GC=F"), AssetType.COMMODITY)
        self.assertEqual(detect_asset_type("MGC=F"), AssetType.COMMODITY)
        self.assertEqual(detect_asset_type("XAUUSD=X"), AssetType.COMMODITY)
        self.assertEqual(detect_asset_type("XAU=X"), AssetType.COMMODITY)

    def test_detection_is_case_insensitive(self):
        self.assertEqual(detect_asset_type("gld"), AssetType.COMMODITY)
        self.assertEqual(detect_asset_type("gc=f"), AssetType.COMMODITY)

    def test_non_gold_tickers_unchanged(self):
        self.assertEqual(detect_asset_type("AAPL"), AssetType.STOCK)
        self.assertEqual(detect_asset_type("SPY"), AssetType.STOCK)
        self.assertEqual(detect_asset_type("BTC-USD"), AssetType.CRYPTO)

    def test_gold_takes_precedence_over_crypto_suffix(self):
        # A hypothetical ``XAU-USD`` ticker would match the crypto suffix
        # heuristic by accident; making sure we don't add anything like
        # that to GOLD_TICKERS without intending to. Today none of the
        # entries in GOLD_TICKERS end with crypto suffixes, so the
        # invariant holds; this test guards against regression.
        crypto_suffixes = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")
        offenders = [t for t in GOLD_TICKERS if t.endswith(crypto_suffixes)]
        self.assertEqual(
            offenders,
            [],
            msg=f"GOLD_TICKERS entries collide with crypto suffixes: {offenders}",
        )


class CommodityAnalystFilterTests(unittest.TestCase):
    """Fundamentals Analyst is auto-removed for commodity, like crypto."""

    def _all_analysts(self):
        return [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]

    def test_filters_out_fundamentals_for_commodity(self):
        self.assertEqual(
            filter_analysts_for_asset_type(self._all_analysts(), AssetType.COMMODITY),
            [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
        )

    def test_keeps_all_analysts_for_stock(self):
        self.assertEqual(
            filter_analysts_for_asset_type(self._all_analysts(), AssetType.STOCK),
            self._all_analysts(),
        )


class InstrumentContextTests(unittest.TestCase):
    """``build_instrument_context`` must steer LLMs away from equity-style
    fundamentals when the asset is a commodity."""

    def test_commodity_context_mentions_gold_drivers(self):
        ctx = build_instrument_context("GC=F", asset_type="commodity")
        # Spot-check that the canonical gold drivers are listed.
        for driver in (
            "USD strength",
            "real yields",
            "central-bank",
            "geopolitical risk",
            "inflation expectations",
            "ETF flows",
        ):
            self.assertIn(driver, ctx, msg=f"missing driver: {driver}")

    def test_commodity_context_warns_off_company_fundamentals(self):
        ctx = build_instrument_context("XAUUSD=X", asset_type="commodity")
        self.assertIn("do not apply", ctx)
        self.assertIn("commodity", ctx)

    def test_stock_context_unchanged(self):
        # Backwards-compat: the equity path has no extra hint and uses
        # the generic "instrument" label.
        ctx = build_instrument_context("AAPL", asset_type="stock")
        self.assertIn("`AAPL`", ctx)
        self.assertNotIn("commodity", ctx.lower())
        self.assertNotIn("crypto asset", ctx)


class RiskFundamentalsBlockTests(unittest.TestCase):
    """The risk debators' fundamentals reference block must adapt by
    asset type and gracefully omit empty sections."""

    def test_stock_keeps_company_label(self):
        out = build_risk_fundamentals_block("Q1 EPS beat 5%", "stock")
        self.assertIn("Company Fundamentals Report", out)
        self.assertIn("Q1 EPS beat 5%", out)

    def test_commodity_uses_macro_label_when_present(self):
        out = build_risk_fundamentals_block("Real yields falling", "commodity")
        self.assertIn("Macro & Monetary Context", out)
        self.assertNotIn("Company Fundamentals Report", out)

    def test_commodity_omits_empty_block(self):
        # When fundamentals_report is empty (typical for gold runs since
        # the analyst is disabled), the block must collapse to "" so the
        # LLM doesn't see a misleading dangling heading.
        self.assertEqual(build_risk_fundamentals_block("", "commodity"), "")

    def test_crypto_omits_empty_block(self):
        self.assertEqual(build_risk_fundamentals_block("", "crypto"), "")


class CommodityPropagatorTests(unittest.TestCase):
    """Initial graph state must round-trip the new asset_type value."""

    def test_initial_state_records_commodity(self):
        state = Propagator().create_initial_state(
            "GLD", "2026-05-22", asset_type=AssetType.COMMODITY.value
        )
        self.assertEqual(state["asset_type"], "commodity")
        self.assertEqual(state["company_of_interest"], "GLD")


if __name__ == "__main__":
    unittest.main()
