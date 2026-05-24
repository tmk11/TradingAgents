"""Optional RAG / semantic-memory layer for TradingAgents.

This package is imported lazily from :class:`tradingagents.agents.utils.memory.TradingMemoryLog`
when ``rag_enabled`` is set in config, so installations without
``chromadb`` (or without an OpenAI key for embeddings) keep working
with the legacy recency-based memory log.

Public surface:

- :func:`create_embedder` — build an embedding callable (OpenAI or fake)
- :class:`MemoryVectorStore` — Chroma-backed persistent vector store
- :class:`SemanticMemoryRetriever` — index/search wrapper used by the log

The graph-side wiring lives in
:mod:`tradingagents.graph.memory_retriever_node` to keep import
boundaries clean: this package has no LangGraph dependency.
"""

from .embeddings import FakeEmbedder, OpenAIEmbedder, create_embedder
from .memory_retriever import SemanticMemoryRetriever
from .news_archive import ArchiveArticle, NewsArchive
from .vector_store import MemoryVectorStore

__all__ = [
    "ArchiveArticle",
    "FakeEmbedder",
    "MemoryVectorStore",
    "NewsArchive",
    "OpenAIEmbedder",
    "SemanticMemoryRetriever",
    "create_embedder",
]
