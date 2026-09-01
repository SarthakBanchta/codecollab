"""
TrustRAG — Retrieval with Confidence Gate
Retrieves top-k chunks from ChromaDB and applies a similarity threshold
to flag low-confidence queries before they reach generation.
"""

import time
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions

import config


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its content, metadata, and similarity score."""
    content: str
    metadata: dict
    score: float  # normalized similarity score (0-1, higher = more similar)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self.metadata.get("chunk_id", "unknown")


@dataclass
class RetrievalResult:
    """Result of a retrieval query, including confidence assessment."""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    is_low_confidence: bool = False
    retrieval_time_ms: float = 0.0
    top_score: float = 0.0
    query: str = ""


def _get_collection() -> chromadb.Collection:
    """
    Load the persistent ChromaDB collection.
    Uses OpenAI embeddings to match what was used during ingestion.
    """
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    # Use the same embedding function as ingestion
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.EMBEDDING_MODEL,
    )

    collection = client.get_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=openai_ef,
    )
    return collection


def retrieve(query: str) -> RetrievalResult:
    """
    Retrieve top-k chunks from ChromaDB for the given query.

    Uses ChromaDB's native query API (not LangChain's retriever wrapper) to get
    raw distance scores needed for the confidence gate.

    ChromaDB returns L2 distances by default. We convert to similarity scores:
        score = 1 / (1 + distance)
    This normalizes to (0, 1] where 1 = identical, 0 = infinitely distant.

    If the best score is below SIMILARITY_THRESHOLD, the query is flagged as
    low-confidence so downstream components can refuse or hedge.
    """
    start = time.perf_counter()

    try:
        collection = _get_collection()
    except Exception as e:
        # Collection doesn't exist — need to run ingestion first
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            chunks=[],
            is_low_confidence=True,
            retrieval_time_ms=elapsed,
            query=query,
        )

    # Query ChromaDB for top-k results with distances
    results = collection.query(
        query_texts=[query],
        n_results=config.TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    # Parse results into RetrievedChunk objects
    chunks = []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert L2 distance to similarity score (0-1, higher = better)
        score = 1.0 / (1.0 + dist)
        chunks.append(RetrievedChunk(
            content=doc,
            metadata=meta,
            score=round(score, 4),
        ))

    # Sort by score descending (most similar first)
    chunks.sort(key=lambda c: c.score, reverse=True)

    # Apply confidence gate
    top_score = chunks[0].score if chunks else 0.0
    is_low_confidence = top_score < config.SIMILARITY_THRESHOLD

    elapsed = (time.perf_counter() - start) * 1000

    return RetrievalResult(
        chunks=chunks,
        is_low_confidence=is_low_confidence,
        retrieval_time_ms=round(elapsed, 2),
        top_score=round(top_score, 4),
        query=query,
    )
