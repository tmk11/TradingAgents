"""Tests for the Gold-Edition multi-source RSS news fetchers.

Network access is mocked end-to-end so these tests are deterministic
and don't depend on the live feeds being reachable from CI.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tradingagents.dataflows import gold_news
from tradingagents.dataflows.gold_news import GOLD_FEEDS, GoldNewsFeed


def _rfc2822(dt: datetime) -> str:
    """Format a UTC datetime as RFC-2822 (the format RSS uses)."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


def _build_rss(items: list[dict]) -> bytes:
    """Build a minimal RSS 2.0 document from item dicts.

    Each item dict may contain ``title``, ``link``, ``description``,
    ``pub_date`` (datetime). Missing fields are omitted to exercise
    the parser's tolerance for irregular feeds.
    """
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0"><channel>',
        "<title>Test feed</title><link>https://example.test/</link>",
        "<description>fixture</description>",
    ]
    for it in items:
        parts.append("<item>")
        if "title" in it:
            parts.append(f"<title>{it['title']}</title>")
        if "link" in it:
            parts.append(f"<link>{it['link']}</link>")
        if "description" in it:
            parts.append(f"<description>{it['description']}</description>")
        if "pub_date" in it and it["pub_date"] is not None:
            parts.append(f"<pubDate>{_rfc2822(it['pub_date'])}</pubDate>")
        parts.append("</item>")
    parts.append("</channel></rss>")
    return "".join(parts).encode("utf-8")


class _FakeResp:
    """Minimal context-manager wrapper for ``urlopen``-style mocks."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _patch_urlopen(rss_payload: bytes):
    """Make ``urlopen`` return the given payload for every call."""
    return patch.object(gold_news, "urlopen", return_value=_FakeResp(rss_payload))


class StripHtmlTests(unittest.TestCase):
    def test_removes_tags_and_collapses_whitespace(self):
        out = gold_news._strip_html(
            "<p>Gold rallies as <b>Fed pivots</b>.</p>\n<p>Real yields fall.</p>"
        )
        self.assertIn("Gold rallies as Fed pivots", out)
        self.assertIn("Real yields fall", out)
        # Ensure no HTML tags survive.
        self.assertNotIn("<", out)
        self.assertNotIn(">", out)

    def test_decodes_common_entities(self):
        out = gold_news._strip_html("AT&amp;T &lt;buys&gt; mine")
        self.assertIn("AT&T", out)
        self.assertIn("<buys>", out)

    def test_truncates_with_ellipsis(self):
        out = gold_news._strip_html("a" * 400, max_len=50)
        self.assertEqual(len(out), 50)
        self.assertTrue(out.endswith("…"))


class ParsePubDateTests(unittest.TestCase):
    def test_parses_rfc2822_with_offset(self):
        dt = gold_news._parse_pub_date("Wed, 21 May 2026 14:30:00 +0000")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 5)
        self.assertEqual(dt.day, 21)

    def test_parses_investing_dot_com_iso_style(self):
        # Investing.com emits ``YYYY-MM-DD HH:MM:SS`` without a timezone.
        # The fallback parser treats those as UTC.
        dt = gold_news._parse_pub_date("2026-05-23 18:24:27")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.day, 23)

    def test_returns_none_on_garbage(self):
        self.assertIsNone(gold_news._parse_pub_date(""))
        self.assertIsNone(gold_news._parse_pub_date("not a date"))


class FetchFeedTests(unittest.TestCase):
    """End-to-end behaviour of the per-feed fetcher with mocked RSS."""

    def setUp(self):
        self.feed = GoldNewsFeed(label="Test Source", url="https://example.test/feed")

    def test_filters_to_lookback_window(self):
        anchor = datetime(2026, 5, 22, tzinfo=timezone.utc)
        items = [
            {
                "title": "In-window: Fed cuts 25bp",
                "description": "Real yields fall.",
                "link": "https://example.test/a",
                "pub_date": anchor - timedelta(days=1),
            },
            {
                "title": "Out-of-window: old central-bank story",
                "description": "Stale.",
                "link": "https://example.test/b",
                "pub_date": anchor - timedelta(days=30),
            },
            {
                "title": "Future-leak: tomorrow's headline",
                "pub_date": anchor + timedelta(days=5),
            },
        ]
        with _patch_urlopen(_build_rss(items)):
            out = gold_news.fetch_feed(
                self.feed,
                look_back_days=7,
                limit=10,
                curr_date="2026-05-22",
            )
        self.assertIn("Fed cuts 25bp", out)
        self.assertNotIn("old central-bank story", out)
        self.assertNotIn("Future-leak", out)

    def test_keeps_items_with_missing_pub_date(self):
        items = [
            {"title": "Undated but real"},
            {"title": "Also undated", "link": "https://example.test/x"},
        ]
        with _patch_urlopen(_build_rss(items)):
            out = gold_news.fetch_feed(self.feed, curr_date="2026-05-22")
        self.assertIn("Undated but real", out)
        self.assertIn("Also undated", out)

    def test_orders_newest_first(self):
        anchor = datetime(2026, 5, 22, tzinfo=timezone.utc)
        items = [
            {"title": "Older", "pub_date": anchor - timedelta(days=3)},
            {"title": "Newest", "pub_date": anchor - timedelta(hours=2)},
            {"title": "Middle", "pub_date": anchor - timedelta(days=1)},
        ]
        with _patch_urlopen(_build_rss(items)):
            out = gold_news.fetch_feed(
                self.feed, look_back_days=7, limit=10, curr_date="2026-05-22"
            )
        # Newest must appear before Middle, which appears before Older.
        self.assertLess(out.find("Newest"), out.find("Middle"))
        self.assertLess(out.find("Middle"), out.find("Older"))

    def test_returns_placeholder_when_feed_empty(self):
        with _patch_urlopen(_build_rss([])):
            out = gold_news.fetch_feed(self.feed, curr_date="2026-05-22")
        self.assertIn("no articles available", out)

    def test_handles_network_error_gracefully(self):
        from urllib.error import URLError

        with patch.object(gold_news, "urlopen", side_effect=URLError("boom")):
            out = gold_news.fetch_feed(self.feed, curr_date="2026-05-22")
        self.assertIn("no articles available", out)

    def test_emits_section_header_with_label(self):
        items = [
            {
                "title": "Headline 1",
                "pub_date": datetime(2026, 5, 22, tzinfo=timezone.utc),
            }
        ]
        with _patch_urlopen(_build_rss(items)):
            out = gold_news.fetch_feed(self.feed, curr_date="2026-05-22")
        self.assertIn("## Test Source", out)


class FetchGoldMacroNewsTests(unittest.TestCase):
    def test_combines_all_registered_feeds(self):
        # All feeds get the same canned RSS payload, but each renders
        # under its own label so the combined block has one section per
        # feed. That gives the LLM clean attribution.
        rss = _build_rss(
            [
                {
                    "title": "Macro: dollar slips on Fed minutes",
                    "pub_date": datetime(2026, 5, 21, tzinfo=timezone.utc),
                }
            ]
        )
        with _patch_urlopen(rss):
            out = gold_news.fetch_gold_macro_news(
                "2026-05-22", look_back_days=7, limit=5
            )

        # Header surfaces the feed count and date window.
        self.assertIn("Gold-complex news, last 7 day(s) ending 2026-05-22", out)
        self.assertIn(f"{len(GOLD_FEEDS)} gold-relevant RSS feed", out)

        # Every registered feed's label must show up at least once.
        for feed in GOLD_FEEDS:
            self.assertIn(feed.label, out, msg=f"missing section for {feed.label}")

    def test_accepts_explicit_feed_override(self):
        custom = [
            GoldNewsFeed(label="Custom Source A", url="https://a.test/feed"),
            GoldNewsFeed(label="Custom Source B", url="https://b.test/feed"),
        ]
        rss = _build_rss(
            [
                {
                    "title": "Some headline",
                    "pub_date": datetime(2026, 5, 21, tzinfo=timezone.utc),
                }
            ]
        )
        with _patch_urlopen(rss):
            out = gold_news.fetch_gold_macro_news(
                "2026-05-22", look_back_days=7, limit=5, feeds=custom
            )
        self.assertIn("Custom Source A", out)
        self.assertIn("Custom Source B", out)
        # Default feeds must NOT leak in when an override is given.
        for default in GOLD_FEEDS:
            self.assertNotIn(default.label, out)


class FeedRegistryTests(unittest.TestCase):
    """The registry itself should be sane and reachable."""

    def test_registry_is_populated(self):
        self.assertGreaterEqual(len(GOLD_FEEDS), 2)

    def test_every_feed_has_https_url_and_label(self):
        for feed in GOLD_FEEDS:
            self.assertTrue(feed.url.startswith("https://"))
            self.assertGreater(len(feed.label), 0)


class GetGoldNewsToolTests(unittest.TestCase):
    """The tool wrapper should be a LangChain tool, callable and routing."""

    def test_tool_is_registered(self):
        from tradingagents.agents.utils.news_data_tools import get_gold_news

        self.assertEqual(get_gold_news.name, "get_gold_news")
        self.assertTrue(callable(get_gold_news.func))

    def test_tool_invokes_underlying_fetcher(self):
        from tradingagents.agents.utils.news_data_tools import get_gold_news

        with patch(
            "tradingagents.dataflows.gold_news.fetch_gold_macro_news",
            return_value="STUBBED",
        ) as stub:
            out = get_gold_news.func("2026-05-22", look_back_days=3, limit=2)
        self.assertEqual(out, "STUBBED")
        stub.assert_called_once()
        kwargs = stub.call_args.kwargs
        self.assertEqual(kwargs["look_back_days"], 3)
        self.assertEqual(kwargs["limit"], 2)


class BackcompatShimTests(unittest.TestCase):
    """``fetch_kitco_news`` survives as a placeholder after the 2025 outage."""

    def test_kitco_shim_returns_clear_placeholder(self):
        out = gold_news.fetch_kitco_news(curr_date="2026-05-22")
        self.assertIn("Kitco", out)
        self.assertIn("no longer publishes", out.lower().replace("rss", "rss"))


if __name__ == "__main__":
    unittest.main()
