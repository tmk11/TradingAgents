"""Append-only markdown decision log for TradingAgents.

Two read paths are supported:

- :meth:`get_past_context` — legacy recency-based selector
  (last N same-ticker entries + last M cross-ticker reflections).
- :meth:`get_past_context_semantic` — embedding-similarity selector
  backed by a Chroma vector store, opt-in via ``rag_enabled`` in
  config. Falls back to recency on any retriever failure so the
  trading run never breaks because of RAG infrastructure issues.
"""

import logging
from typing import List, Optional
from pathlib import Path
import re

from tradingagents.agents.utils.rating import parse_rating

logger = logging.getLogger(__name__)


class TradingMemoryLog:
    """Append-only markdown log of trading decisions and reflections."""

    # HTML comment: cannot appear in LLM prose output, safe as a hard delimiter
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    # Precompiled patterns — avoids re-compilation on every load_entries() call
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._config = cfg
        self._log_path = None
        path = cfg.get("memory_log_path")
        if path:
            self._log_path = Path(path).expanduser()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Optional cap on resolved entries. None disables rotation.
        self._max_entries = cfg.get("memory_log_max_entries")
        # ---- Optional RAG retriever ------------------------------
        # Initialised lazily on first use so that
        # ``TradingMemoryLog(config=None)`` and ``rag_enabled=False``
        # paths do no chromadb / network work at all.
        self._retriever = None
        self._retriever_init_attempted = False
        self._rag_enabled = bool(cfg.get("rag_enabled"))

    # --- Write path (Phase A) ---

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
    ) -> None:
        """Append pending entry at end of propagate(). No LLM call."""
        if not self._log_path:
            return
        # Idempotency guard: fast raw-text scan instead of full parse
        if self._log_path.exists():
            raw = self._log_path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                if line.startswith(f"[{trade_date} | {ticker} |") and line.endswith("| pending]"):
                    return
        rating = parse_rating(final_trade_decision)
        tag = f"[{trade_date} | {ticker} | {rating} | pending]"
        entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(entry)

        # Mirror into the vector store when RAG is enabled. We index
        # pending entries too — same-ticker hits are scoped by
        # ``pending == False`` at search time so this stays inert
        # until the entry is resolved, but it lets cross-run debug
        # tooling inspect what's in flight.
        retriever = self._get_retriever()
        if retriever is not None:
            retriever.index_entry({
                "ticker": ticker,
                "date": str(trade_date),
                "rating": rating,
                "decision": final_trade_decision,
                "reflection": "",
                "pending": True,
                "raw": "pending",
                "alpha": None,
            })

    # --- Read path (Phase A) ---

    def load_entries(self) -> List[dict]:
        """Parse all entries from log. Returns list of dicts."""
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        entries = []
        for raw in raw_entries:
            parsed = self._parse_entry(raw)
            if parsed:
                entries.append(parsed)
        return entries

    def get_pending_entries(self) -> List[dict]:
        """Return entries with outcome:pending (for Phase B)."""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_past_context(self, ticker: str, n_same: int = 5, n_cross: int = 3) -> str:
        """Return formatted past context string for agent prompt injection."""
        entries = [e for e in self.load_entries() if not e.get("pending")]
        if not entries:
            return ""

        same, cross = [], []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker and len(same) < n_same:
                same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        if not same and not cross:
            return ""

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    # --- Update path (Phase B) ---

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str,
    ) -> None:
        """Replace pending tag and append REFLECTION section using atomic write.

        Finds the first pending entry matching (trade_date, ticker), updates
        its tag with return figures, and appends a REFLECTION section.  Uses
        a temp-file + os.replace() so a crash mid-write never corrupts the log.
        """
        if not self._log_path or not self._log_path.exists():
            return

        text = self._log_path.read_text(encoding="utf-8")
        blocks = text.split(self._SEPARATOR)

        pending_prefix = f"[{trade_date} | {ticker} |"
        raw_pct = f"{raw_return:+.1%}"
        alpha_pct = f"{alpha_return:+.1%}"

        updated = False
        new_blocks = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                new_blocks.append(block)
                continue

            lines = stripped.splitlines()
            tag_line = lines[0].strip()

            if (
                not updated
                and tag_line.startswith(pending_prefix)
                and tag_line.endswith("| pending]")
            ):
                # Parse rating from the existing pending tag
                fields = [f.strip() for f in tag_line[1:-1].split("|")]
                rating = fields[2]
                new_tag = (
                    f"[{trade_date} | {ticker} | {rating}"
                    f" | {raw_pct} | {alpha_pct} | {holding_days}d]"
                )
                rest = "\n".join(lines[1:])
                new_blocks.append(
                    f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{reflection}"
                )
                updated = True
            else:
                new_blocks.append(block)

        if not updated:
            return

        new_blocks = self._apply_rotation(new_blocks)
        new_text = self._SEPARATOR.join(new_blocks)
        tmp_path = self._log_path.with_suffix(".tmp")
        tmp_path.write_text(new_text, encoding="utf-8")
        tmp_path.replace(self._log_path)

        # Re-index the resolved entry so semantic search now sees a
        # ``pending == False`` document with the reflection text.
        retriever = self._get_retriever()
        if retriever is not None:
            retriever.index_entry({
                "ticker": ticker,
                "date": str(trade_date),
                "rating": rating,
                "decision": rest,
                "reflection": reflection,
                "pending": False,
                "raw": raw_pct,
                "alpha": alpha_pct,
                "holding": f"{holding_days}d",
            })

    def batch_update_with_outcomes(self, updates: List[dict]) -> None:
        """Apply multiple outcome updates in a single read + atomic write.

        Each element of updates must have keys: ticker, trade_date,
        raw_return, alpha_return, holding_days, reflection.
        """
        if not self._log_path or not self._log_path.exists() or not updates:
            return

        text = self._log_path.read_text(encoding="utf-8")
        blocks = text.split(self._SEPARATOR)

        # Build lookup keyed by (trade_date, ticker) for O(1) dispatch
        update_map = {(u["trade_date"], u["ticker"]): u for u in updates}

        new_blocks = []
        resolved_for_reindex: List[dict] = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                new_blocks.append(block)
                continue

            lines = stripped.splitlines()
            tag_line = lines[0].strip()

            matched = False
            for (trade_date, ticker), upd in list(update_map.items()):
                pending_prefix = f"[{trade_date} | {ticker} |"
                if tag_line.startswith(pending_prefix) and tag_line.endswith("| pending]"):
                    fields = [f.strip() for f in tag_line[1:-1].split("|")]
                    rating = fields[2]
                    raw_pct = f"{upd['raw_return']:+.1%}"
                    alpha_pct = f"{upd['alpha_return']:+.1%}"
                    new_tag = (
                        f"[{trade_date} | {ticker} | {rating}"
                        f" | {raw_pct} | {alpha_pct} | {upd['holding_days']}d]"
                    )
                    rest = "\n".join(lines[1:])
                    new_blocks.append(
                        f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{upd['reflection']}"
                    )
                    resolved_for_reindex.append({
                        "ticker": ticker,
                        "date": str(trade_date),
                        "rating": rating,
                        "decision": rest,
                        "reflection": upd["reflection"],
                        "pending": False,
                        "raw": raw_pct,
                        "alpha": alpha_pct,
                        "holding": f"{upd['holding_days']}d",
                    })
                    del update_map[(trade_date, ticker)]
                    matched = True
                    break

            if not matched:
                new_blocks.append(block)

        new_blocks = self._apply_rotation(new_blocks)
        new_text = self._SEPARATOR.join(new_blocks)
        tmp_path = self._log_path.with_suffix(".tmp")
        tmp_path.write_text(new_text, encoding="utf-8")
        tmp_path.replace(self._log_path)

        # Re-index every resolved entry. Done after the atomic write
        # so a vector-store failure can't corrupt the markdown.
        retriever = self._get_retriever()
        if retriever is not None and resolved_for_reindex:
            retriever.reindex(resolved_for_reindex)

    # --- Helpers ---

    def _apply_rotation(self, blocks: List[str]) -> List[str]:
        """Drop oldest resolved blocks when their count exceeds max_entries.

        Pending blocks are always kept (they represent unprocessed work).
        Returns ``blocks`` unchanged when rotation is disabled or under cap.
        """
        if not self._max_entries or self._max_entries <= 0:
            return blocks

        # Tag each block with (kept, is_resolved) by parsing tag-line markers.
        decisions = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                decisions.append((block, False))
                continue
            tag_line = stripped.splitlines()[0].strip()
            is_resolved = (
                tag_line.startswith("[")
                and tag_line.endswith("]")
                and not tag_line.endswith("| pending]")
            )
            decisions.append((block, is_resolved))

        resolved_count = sum(1 for _, r in decisions if r)
        if resolved_count <= self._max_entries:
            return blocks

        to_drop = resolved_count - self._max_entries
        kept: List[str] = []
        for block, is_resolved in decisions:
            if is_resolved and to_drop > 0:
                to_drop -= 1
                continue
            kept.append(block)
        return kept

    def _parse_entry(self, raw: str) -> Optional[dict]:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None
        fields = [f.strip() for f in tag_line[1:-1].split("|")]
        if len(fields) < 4:
            return None
        entry = {
            "date": fields[0],
            "ticker": fields[1],
            "rating": fields[2],
            "pending": fields[3] == "pending",
            "raw": fields[3] if fields[3] != "pending" else None,
            "alpha": fields[4] if len(fields) > 4 else None,
            "holding": fields[5] if len(fields) > 5 else None,
        }
        body = "\n".join(lines[1:]).strip()
        decision_match = self._DECISION_RE.search(body)
        reflection_match = self._REFLECTION_RE.search(body)
        entry["decision"] = decision_match.group(1).strip() if decision_match else ""
        entry["reflection"] = reflection_match.group(1).strip() if reflection_match else ""
        return entry

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"

    # --- RAG / semantic retrieval ------------------------------------

    def _get_retriever(self):
        """Lazy-init the semantic retriever. Returns ``None`` when off.

        Init failures (missing chromadb, missing OPENAI_API_KEY,
        embedder import error) are logged once and the log silently
        falls back to recency-based retrieval — RAG is a quality
        improvement, never a hard dependency for a trading run.
        """
        if not self._rag_enabled:
            return None
        if self._retriever is not None or self._retriever_init_attempted:
            return self._retriever
        self._retriever_init_attempted = True
        cfg = self._config
        try:
            from tradingagents.retrieval import (
                MemoryVectorStore,
                SemanticMemoryRetriever,
                create_embedder,
            )

            provider = cfg.get("rag_embedding_provider", "openai")
            model = cfg.get("rag_embedding_model", "text-embedding-3-small")
            embedder = create_embedder(provider=provider, model=model)
            store = MemoryVectorStore(
                path=cfg.get("rag_vector_store_path", ":memory:"),
                embedder=embedder,
                embedder_name=f"{provider}:{model}",
            )
            self._retriever = SemanticMemoryRetriever(store)
            # Bootstrap: if the markdown log already has entries that
            # aren't in the vector store yet (first run after enabling
            # RAG, or after wiping the chroma directory), back-fill
            # them so the very first semantic query is useful.
            self._bootstrap_index(store)
        except Exception as exc:
            logger.warning(
                "RAG retriever init failed (%s) — falling back to recency-based memory.",
                exc,
            )
            self._retriever = None
        return self._retriever

    def _bootstrap_index(self, store) -> None:
        """Index any markdown entries not yet in the vector store."""
        try:
            existing_ids = set(store.all_ids())
        except Exception:  # pragma: no cover - defensive
            existing_ids = set()
        to_index = []
        for entry in self.load_entries():
            entry_id = f"{entry.get('ticker', '')}:{entry.get('date', '')}"
            if entry_id in existing_ids:
                continue
            to_index.append(entry)
        if to_index and self._retriever is not None:
            self._retriever.reindex(to_index)

    def get_past_context_semantic(
        self,
        query: str,
        ticker: str,
        n_same: int = 5,
        n_cross: int = 3,
    ) -> str:
        """Embedding-similarity past_context for the Portfolio Manager.

        Falls back to :meth:`get_past_context` whenever the retriever
        isn't available, so callers can switch to this method
        unconditionally without first checking ``rag_enabled``.
        """
        retriever = self._get_retriever()
        if retriever is None:
            return self.get_past_context(ticker, n_same=n_same, n_cross=n_cross)
        try:
            same, cross = retriever.search(
                query=query,
                ticker=ticker,
                n_same=n_same,
                n_cross=n_cross,
            )
        except Exception as exc:
            logger.warning("RAG search failed (%s) — falling back to recency.", exc)
            return self.get_past_context(ticker, n_same=n_same, n_cross=n_cross)
        formatted = retriever.format_context(ticker, same, cross)
        if formatted:
            return formatted
        # No semantic hits yet (e.g., empty store) — recency fallback
        # is still the best we can do.
        return self.get_past_context(ticker, n_same=n_same, n_cross=n_cross)

    @property
    def rag_enabled(self) -> bool:
        return self._rag_enabled
