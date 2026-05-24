"""Embedding callables used by the RAG memory retriever.

Two providers are shipped:

- :class:`OpenAIEmbedder` — wraps ``langchain_openai.OpenAIEmbeddings``.
  Requires ``OPENAI_API_KEY``; default model ``text-embedding-3-small``.
- :class:`FakeEmbedder` — deterministic hash-based bag-of-words embedder
  for offline tests and smoke runs. No network, no API key, fixed
  dimension. Vectors are L2-normalised so cosine similarity is in
  ``[-1, 1]`` and shared tokens drive non-trivial similarity.

Both implement the same callable shape so the Chroma adapter doesn't
have to special-case them: ``embedder(list[str]) -> list[list[float]]``.

The :func:`create_embedder` factory is what the rest of the codebase
should use; it accepts a ``provider`` string straight from config.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence


class FakeEmbedder:
    """Deterministic bag-of-words embedder. Tests and offline smoke runs.

    Token hashing is stable across processes because we use MD5; the
    resulting vector dimension defaults to 256 — small enough to keep
    Chroma fast in tests, large enough to keep collisions rare.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("FakeEmbedder dim must be > 0")
        self.dim = dim

    # The Chroma EmbeddingFunction protocol expects ``__call__(input)``
    # where ``input`` is a list of strings.
    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    # Convenience alias matching langchain's embeddings interface.
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        # Tokenise on whitespace and punctuation. Lower-case so
        # "Buy" / "buy" hash to the same bucket. Reasonable for the
        # short markdown decision blocks the memory log stores.
        tokens = (
            text.lower()
            .replace("\n", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("(", " ")
            .replace(")", " ")
            .split()
        )
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self.dim
            # Sign bit from the next nibble so two tokens hashing to
            # the same bucket can still cancel — lifts effective rank.
            sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedder:
    """Wraps ``langchain_openai.OpenAIEmbeddings`` for Chroma.

    Imported lazily so the module can still be imported on machines
    without the OpenAI SDK or a key — instantiation will raise, and
    the caller (``TradingMemoryLog._init_retriever``) is responsible
    for falling back to the recency-based path.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "OpenAIEmbedder requires `langchain-openai`. Install it "
                "or set rag_embedding_provider='fake'."
            ) from exc
        self.model = model
        self._client = OpenAIEmbeddings(model=model)

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        return self._client.embed_documents(list(texts))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


def create_embedder(
    provider: str = "openai",
    model: str = "text-embedding-3-small",
) -> object:
    """Build an embedder by provider name.

    ``provider`` is normalised to lower-case so config can pass any
    casing. Unknown providers raise ``ValueError`` so a typo surfaces
    immediately rather than silently disabling RAG.
    """
    p = (provider or "openai").strip().lower()
    if p == "fake":
        return FakeEmbedder()
    if p == "openai":
        return OpenAIEmbedder(model=model)
    raise ValueError(
        f"Unknown rag_embedding_provider: {provider!r} "
        "(supported: 'openai', 'fake')"
    )


def _coerce_iterable(input_: Iterable[str]) -> List[str]:
    """Helper for adapters: tolerate any iterable of strings."""
    return [str(t) for t in input_]
