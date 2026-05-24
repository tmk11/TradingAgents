"""Chroma-backed persistent vector store for the trading memory log.

The store is intentionally tiny — one collection per memory log,
keyed by ``ticker:date`` — so it can live alongside the existing
``trading_memory.md`` markdown file without contention.  Chroma is
the right default here:

- pure-Python install, no separate service
- on-disk persistence with predictable file layout
- supports ``where`` filters needed for ticker scoping

The :class:`MemoryVectorStore` class wraps the chroma client and
exposes ``upsert`` / ``search`` calls in the shape the memory log
needs, so the log itself never has to know about chromadb's API.

Chroma is imported lazily inside ``__init__`` so that simply
importing this module — for type hints, dataclass dispatch, or
unit tests that stub the store — does not require chromadb to be
installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# Chroma's metadata values must be primitive (str | int | float | bool).
# We coerce in :meth:`MemoryVectorStore.upsert` so callers can pass plain
# dicts straight through from the markdown parser.
_PRIMITIVE = (str, int, float, bool)


def _coerce_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata values are Chroma-acceptable primitives.

    None is dropped (Chroma rejects it), other non-primitives are
    str()'d so we never silently lose information.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, _PRIMITIVE):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


class _ChromaEmbeddingAdapter:
    """Adapt our embedder callable to chromadb's ``EmbeddingFunction`` protocol.

    Chroma 1.x dispatches ``embed_query(input=...)`` for query routing
    (so trained models can produce different vectors for queries vs.
    documents). Our embedders treat the two identically, so we just
    forward to ``__call__`` with the same shape contract: ``input``
    is a list of strings, returns a list of vectors.
    """

    def __init__(self, embedder, name: str) -> None:
        self._embedder = embedder
        self._name = name

    def __call__(self, input):  # noqa: A002 - chromadb dictates the kw name
        return self._embedder(list(input))

    # Chroma uses this to fingerprint the collection so a different
    # embedder on reopen raises rather than silently producing
    # incompatible vectors.
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name

    # ---- Query embedding ------------------------------------------
    # Chroma 1.x's protocol passes ``input`` here too — same shape as
    # ``__call__``. Our embedders don't differentiate between query
    # and document encoding, so we route both through one path.
    def embed_query(self, input):  # noqa: A002
        return self._embedder(list(input))


class MemoryVectorStore:
    """Persistent Chroma vector store wrapping a single collection.

    Args:
        path: directory where chromadb persists its sqlite + parquet
            files. Created if missing. Pass ``":memory:"`` for an
            in-memory store (tests, ephemeral runs).
        embedder: callable that takes a list of strings and returns a
            list of vectors. Adapted to chromadb's protocol internally.
        collection_name: collection identifier inside the Chroma
            client. One memory log → one collection.
        embedder_name: optional fingerprint baked into collection
            metadata so that reopening with a different embedder
            raises rather than silently corrupting recall.
    """

    def __init__(
        self,
        path: str | Path,
        embedder: Callable[[Sequence[str]], List[List[float]]],
        *,
        collection_name: str = "trading_memory",
        embedder_name: str = "tradingagents-default",
    ) -> None:
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chromadb is required for MemoryVectorStore. "
                "Install it or disable rag_enabled in config."
            ) from exc

        self.path = str(path)
        self.collection_name = collection_name
        self._embedding_function = _ChromaEmbeddingAdapter(embedder, embedder_name)

        # ``:memory:`` is a sentinel for an ephemeral client — useful
        # in tests where we don't want the chroma sqlite file
        # leaking into the working tree.
        if self.path == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.path)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            # Cosine distance matches the L2-normalised vectors our
            # embedders produce. Lower distance == more similar.
            metadata={"hnsw:space": "cosine"},
        )

    # ---- Write path ----------------------------------------------------

    def upsert(self, entry_id: str, document: str, metadata: Dict[str, Any]) -> None:
        """Insert or replace a single document.

        Chroma's ``upsert`` is idempotent on ``id``, which is exactly
        what we want when the markdown log is re-indexed (e.g., on
        update_with_outcome — the entry's reflection appears).
        """
        self._collection.upsert(
            ids=[entry_id],
            documents=[document],
            metadatas=[_coerce_metadata(metadata)],
        )

    def upsert_many(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        """Batch upsert. Same idempotency as :meth:`upsert`."""
        if not ids:
            return
        self._collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=[_coerce_metadata(m) for m in metadatas],
        )

    def delete(self, entry_id: str) -> None:
        """Remove a single document by id (no-op if missing)."""
        try:
            self._collection.delete(ids=[entry_id])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to delete %s from vector store: %s", entry_id, exc)

    # ---- Read path -----------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search. Returns ``[]`` when the collection is empty.

        Each hit is a flat dict ``{id, document, metadata, distance}``.
        Distance is cosine distance from chroma; smaller is better.
        """
        # chromadb raises on n_results > collection size in some
        # versions; clamp defensively.
        size = self._collection.count()
        if size == 0:
            return []
        n = max(1, min(n_results, size))

        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n}
        if where:
            kwargs["where"] = where

        try:
            raw = self._collection.query(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Vector store query failed: %s", exc)
            return []

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        hits: List[Dict[str, Any]] = []
        for i, doc, meta, dist in zip(ids, documents, metadatas, distances):
            hits.append(
                {
                    "id": i,
                    "document": doc,
                    "metadata": meta or {},
                    "distance": dist,
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    # ---- Maintenance ---------------------------------------------------

    def all_ids(self) -> List[str]:
        """Return every document id in the collection.

        Used by the memory log to figure out which markdown entries
        still need indexing on first init.
        """
        try:
            res = self._collection.get(include=[])
        except Exception:  # pragma: no cover - defensive
            return []
        return list(res.get("ids") or [])
