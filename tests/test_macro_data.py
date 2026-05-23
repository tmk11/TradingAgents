"""Tests for the Gold-Edition macro-data fetcher.

Live providers (yfinance + FRED) are mocked end-to-end so these tests
are deterministic and don't depend on network reachability.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tradingagents.dataflows import macro_data
from tradingagents.dataflows.macro_data import (
    MACRO_SERIES_GOLD,
    MacroSeries,
    _summarise,
    fetch_gold_macro_data,
    fetch_one_series,
    format_series_block,
)


def _obs_pair(date_str: str, value: float) -> tuple:
    """Helper: build a single (datetime_utc, float) observation tuple."""
    return (
        datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        float(value),
    )


class SummariseTests(unittest.TestCase):
    """The summary stats are the single load-bearing piece of math."""

    def _series_with_change(self) -> list[tuple]:
        # Build a 30-day daily series with values 100, 101, …, 129.
        # Shape lets us assert exact 1d / 1w / 1m percentages below.
        anchor = datetime(2026, 5, 22, tzinfo=timezone.utc)
        return [(anchor - timedelta(days=29 - i), 100.0 + i) for i in range(30)]

    def test_returns_none_for_empty_input(self):
        self.assertIsNone(_summarise([]))

    def test_computes_latest_and_window_extents(self):
        s = self._series_with_change()
        out = _summarise(s)
        self.assertEqual(out["last"], 129.0)
        self.assertEqual(out["last_date"], "2026-05-22")
        self.assertEqual(out["window_min"], 100.0)
        self.assertEqual(out["window_max"], 129.0)
        self.assertEqual(out["n_obs"], 30)

    def test_one_day_change_is_relative_to_prior_observation(self):
        s = self._series_with_change()
        out = _summarise(s)
        # 129 vs 128 => +0.78125%
        self.assertAlmostEqual(out["change_1d_pct"], (129 - 128) / 128 * 100, places=4)

    def test_one_week_change_uses_closest_prior_obs(self):
        s = self._series_with_change()
        out = _summarise(s)
        # 7 days back from 2026-05-22 => 2026-05-15 => value 122
        self.assertAlmostEqual(
            out["change_1w_pct"], (129 - 122) / 122 * 100, places=4
        )

    def test_returns_none_change_when_window_predates_data(self):
        # Single observation: no prior data point for any of the change
        # windows; all should be ``None`` rather than 0%.
        single = [_obs_pair("2026-05-22", 100.0)]
        out = _summarise(single)
        self.assertEqual(out["last"], 100.0)
        self.assertIsNone(out["change_1d_pct"])
        self.assertIsNone(out["change_1w_pct"])
        self.assertIsNone(out["change_1m_pct"])


class FormatSeriesBlockTests(unittest.TestCase):
    def test_formats_summary_into_markdown(self):
        series = MacroSeries(
            provider="yfinance",
            series_id="^TEST",
            label="Test Series",
            description="A test series.",
        )
        summary = {
            "last": 4.55,
            "last_date": "2026-05-22",
            "change_1d_pct": -0.5,
            "change_1w_pct": 2.0,
            "change_1m_pct": None,
            "window_min": 4.0,
            "window_max": 5.0,
            "n_obs": 27,
        }
        out = format_series_block(series, summary)
        self.assertIn("### Test Series", out)
        self.assertIn("4.55", out)
        # Sign formatting: positive must include +, negative includes -.
        self.assertIn("+2.00%", out)
        self.assertIn("-0.50%", out)
        # Missing change renders as "--" (not "None" or "0%").
        self.assertIn("--", out)
        # Min/max + obs count surfaced.
        self.assertIn("4 / 5", out)
        self.assertIn("n=27", out)

    def test_renders_placeholder_when_summary_is_none(self):
        series = MacroSeries(
            provider="fred",
            series_id="UNREACHABLE",
            label="Unreachable Series",
            description="Used to test fallback.",
        )
        out = format_series_block(series, None)
        self.assertIn("no data", out)
        self.assertIn("provider fred unavailable", out)
        self.assertIn("UNREACHABLE", out)


class FetchYfinanceSeriesTests(unittest.TestCase):
    """yfinance fetcher must mock cleanly and degrade gracefully."""

    def _mock_history(self, dates_values):
        """Return a fake yf.Ticker.history() DataFrame stub."""
        import pandas as pd

        idx = pd.DatetimeIndex(
            [datetime.strptime(d, "%Y-%m-%d") for d, _ in dates_values],
            tz="UTC",
        )
        return pd.DataFrame({"Close": [v for _, v in dates_values]}, index=idx)

    def test_returns_chronological_observations(self):
        df = self._mock_history(
            [("2026-05-20", 4.5), ("2026-05-21", 4.6), ("2026-05-22", 4.55)]
        )

        class _FakeTicker:
            def history(self_inner, **_):
                return df

        with patch("yfinance.Ticker", return_value=_FakeTicker()):
            out = macro_data.fetch_yfinance_series(
                "^TNX", lookback_days=30, curr_date="2026-05-22"
            )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0][1], 4.5)
        self.assertEqual(out[-1][1], 4.55)
        # All datetimes are timezone-aware UTC.
        for dt, _ in out:
            self.assertEqual(dt.tzinfo, timezone.utc)

    def test_returns_none_on_empty_dataframe(self):
        import pandas as pd

        empty = pd.DataFrame()

        class _FakeTicker:
            def history(self_inner, **_):
                return empty

        with patch("yfinance.Ticker", return_value=_FakeTicker()):
            out = macro_data.fetch_yfinance_series(
                "^TNX", lookback_days=30, curr_date="2026-05-22"
            )
        self.assertIsNone(out)

    def test_returns_none_on_exception(self):
        class _Boom:
            def history(self_inner, **_):
                raise RuntimeError("network down")

        with patch("yfinance.Ticker", return_value=_Boom()):
            out = macro_data.fetch_yfinance_series(
                "^TNX", lookback_days=30, curr_date="2026-05-22"
            )
        self.assertIsNone(out)


class FetchFredSeriesTests(unittest.TestCase):
    """FRED fetcher must parse CSV correctly and fall back on errors."""

    def _patched_urlopen(self, body: str):
        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return body.encode("utf-8")

        return patch.object(macro_data, "urlopen", return_value=_FakeResp())

    def test_parses_csv_and_filters_to_window(self):
        # FRED's "." marks missing observations and must be skipped.
        body = (
            "DATE,DFII10\n"
            "2026-05-20,1.85\n"
            "2026-05-21,.\n"
            "2026-05-22,1.88\n"
            "2024-01-01,1.50\n"  # outside the 30-day window from anchor
        )
        with self._patched_urlopen(body):
            out = macro_data.fetch_fred_series(
                "DFII10", lookback_days=30, curr_date="2026-05-22"
            )
        # The "." row dropped, the out-of-window row dropped, leaving 2.
        self.assertEqual(len(out), 2)
        self.assertEqual(out[-1][1], 1.88)
        self.assertEqual(out[-1][0].strftime("%Y-%m-%d"), "2026-05-22")

    def test_returns_none_on_empty_body(self):
        with self._patched_urlopen(""):
            out = macro_data.fetch_fred_series(
                "DFII10", lookback_days=30, curr_date="2026-05-22"
            )
        self.assertIsNone(out)

    def test_returns_none_on_network_error(self):
        from urllib.error import URLError

        with patch.object(macro_data, "urlopen", side_effect=URLError("boom")):
            out = macro_data.fetch_fred_series(
                "DFII10", lookback_days=30, curr_date="2026-05-22"
            )
        self.assertIsNone(out)

    def test_returns_none_on_timeout_or_unexpected_exception(self):
        with patch.object(macro_data, "urlopen", side_effect=TimeoutError("slow")):
            out = macro_data.fetch_fred_series(
                "DFII10", lookback_days=30, curr_date="2026-05-22"
            )
        self.assertIsNone(out)


class FetchOneSeriesTests(unittest.TestCase):
    """Provider routing must dispatch on ``MacroSeries.provider``."""

    def test_dispatches_yfinance_provider(self):
        s = MacroSeries(
            provider="yfinance",
            series_id="^TNX",
            label="lbl",
            description="desc",
        )
        with patch.object(
            macro_data,
            "fetch_yfinance_series",
            return_value=[_obs_pair("2026-05-22", 4.5)],
        ) as yf_mock, patch.object(
            macro_data, "fetch_fred_series"
        ) as fred_mock:
            out = fetch_one_series(s, lookback_days=30, curr_date="2026-05-22")
        yf_mock.assert_called_once()
        fred_mock.assert_not_called()
        self.assertIn("4.5", out)

    def test_dispatches_fred_provider(self):
        s = MacroSeries(
            provider="fred", series_id="DFII10", label="lbl", description="desc"
        )
        with patch.object(macro_data, "fetch_yfinance_series") as yf_mock, patch.object(
            macro_data,
            "fetch_fred_series",
            return_value=[_obs_pair("2026-05-22", 1.88)],
        ) as fred_mock:
            out = fetch_one_series(s, lookback_days=30, curr_date="2026-05-22")
        yf_mock.assert_not_called()
        fred_mock.assert_called_once()
        self.assertIn("1.88", out)

    def test_unknown_provider_renders_placeholder(self):
        s = MacroSeries(
            provider="weird", series_id="X", label="lbl", description="desc"
        )
        out = fetch_one_series(s, lookback_days=30, curr_date="2026-05-22")
        self.assertIn("no data", out)


class FetchGoldMacroDataTests(unittest.TestCase):
    """End-to-end: registry expansion + provider routing + formatting."""

    def test_combines_all_registered_series_with_provider_sections(self):
        # Stub both providers so neither hits the network.
        with patch.object(
            macro_data,
            "fetch_yfinance_series",
            return_value=[_obs_pair("2026-05-22", 4.5)],
        ), patch.object(
            macro_data,
            "fetch_fred_series",
            return_value=[_obs_pair("2026-05-22", 1.88)],
        ):
            out = fetch_gold_macro_data("2026-05-22", lookback_days=30)

        # Provider section headings present.
        self.assertIn("## Market data (yfinance)", out)
        self.assertIn("## Macro time series (FRED)", out)

        # Every registered series shows up.
        for s in MACRO_SERIES_GOLD:
            self.assertIn(s.label, out, msg=f"missing series {s.label}")

    def test_fred_failure_falls_back_to_placeholder_only(self):
        # yfinance loads, FRED returns None → yfinance sections rich,
        # FRED sections show placeholders. The combined block stays
        # non-empty and parseable.
        with patch.object(
            macro_data,
            "fetch_yfinance_series",
            return_value=[_obs_pair("2026-05-22", 4.5)],
        ), patch.object(
            macro_data, "fetch_fred_series", return_value=None
        ):
            out = fetch_gold_macro_data("2026-05-22", lookback_days=30)

        # yfinance sections still have data.
        self.assertIn("4.5", out)
        # FRED sections degraded gracefully.
        self.assertIn("provider fred unavailable", out)

    def test_accepts_explicit_series_override(self):
        custom = [
            MacroSeries(
                provider="yfinance",
                series_id="ZZZZ",
                label="Custom Test Series",
                description="Only used in this test.",
            )
        ]
        with patch.object(
            macro_data,
            "fetch_yfinance_series",
            return_value=[_obs_pair("2026-05-22", 1.0)],
        ):
            out = fetch_gold_macro_data(
                "2026-05-22", lookback_days=30, series=custom
            )
        self.assertIn("Custom Test Series", out)
        # Default registry must NOT leak in when an override is given.
        for default in MACRO_SERIES_GOLD:
            if default.label != "Custom Test Series":
                self.assertNotIn(default.label, out)


class GoldNewsBloombergTests(unittest.TestCase):
    """Bloomberg Markets feed must be in the registry after the addition."""

    def test_bloomberg_is_registered(self):
        from tradingagents.dataflows.gold_news import GOLD_FEEDS

        labels = [f.label for f in GOLD_FEEDS]
        self.assertTrue(
            any("Bloomberg" in lbl for lbl in labels),
            msg=f"Bloomberg not in feed registry: {labels}",
        )


class RegistrySanityTests(unittest.TestCase):
    """Every registered series should have a known provider and a label."""

    def test_every_series_has_known_provider(self):
        for s in MACRO_SERIES_GOLD:
            self.assertIn(s.provider, ("yfinance", "fred"))
            self.assertGreater(len(s.label), 0)
            self.assertGreater(len(s.description), 0)
            self.assertGreater(len(s.series_id), 0)


class GetMacroDataToolTests(unittest.TestCase):
    """LangChain tool wrapper must be registered and routable."""

    def test_tool_is_registered(self):
        from tradingagents.agents.utils.macro_data_tools import get_macro_data

        self.assertEqual(get_macro_data.name, "get_macro_data")
        self.assertTrue(callable(get_macro_data.func))

    def test_tool_invokes_underlying_fetcher(self):
        from tradingagents.agents.utils.macro_data_tools import get_macro_data

        with patch(
            "tradingagents.dataflows.macro_data.fetch_gold_macro_data",
            return_value="STUBBED",
        ) as stub:
            out = get_macro_data.func("2026-05-22", lookback_days=45)
        self.assertEqual(out, "STUBBED")
        stub.assert_called_once()
        self.assertEqual(stub.call_args.kwargs["lookback_days"], 45)


if __name__ == "__main__":
    unittest.main()
