"""
TrustRAG — RAG Answer Generation
Builds a grounded RAG prompt with retrieved context and generates an answer
with explicit source citations. Supports OpenAI and Anthropic backends.
"""

import time
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from retriever import RetrievalResult


@dataclass
class GenerationResult:
    """Output of the RAG generation step."""
    answer: str
    is_low_confidence: bool
    generation_time_ms: float
    source_labels: list[str]  # ["Source 1", "Source 2", ...] for citation verification


# ── RAG System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided source chunks. Follow these rules strictly:

1. Base every claim on information from the sources below. Do NOT use any prior knowledge.
2. Cite sources inline using [Source N] notation (e.g., [Source 1], [Source 2]).
3. If the sources don't contain enough information to fully answer the question, say so explicitly. Do NOT guess or fill in gaps.
4. Never invent facts, statistics, dates, or sources that aren't in the provided chunks.
5. Keep your answer focused and concise.

{confidence_warning}

--- SOURCES ---
{formatted_sources}
"""

USER_PROMPT = """{query}"""

LOW_CONFIDENCE_WARNING = (
    "⚠️ IMPORTANT: Retrieval confidence is LOW for this query. "
    "Be especially conservative — only state what the sources directly support. "
    "If the sources don't adequately address the question, say so."
)


def _format_sources(retrieval_result: RetrievalResult) -> tuple[str, list[str]]:
    """
    Format retrieved chunks into numbered source blocks for the prompt.
    Returns (formatted_string, list_of_source_labels).
    """
    if not retrieval_result.chunks:
        return "(No sources retrieved)", []

    lines = []
    labels = []
    for i, chunk in enumerate(retrieval_result.chunks, start=1):
        source_file = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.chunk_id
        label = f"Source {i}"
        labels.append(label)

        lines.append(
            f"[{label}] (from: {source_file}, id: {chunk_id}, "
            f"relevance: {chunk.score:.2f}):\n{chunk.content}"
        )

    return "\n\n".join(lines), labels


def _get_llm():
    """
    Initialize the LLM based on the configured provider.
    Uses temperature=0 for deterministic, faithful outputs.
    """
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.GENERATION_MODEL,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY,
        )
    elif config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.GENERATION_MODEL,
            temperature=0,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")


def generate(query: str, retrieval_result: RetrievalResult) -> GenerationResult:
    """
    Generate a RAG answer grounded in retrieved context.

    The prompt instructs the model to:
    - Answer ONLY from the provided source chunks
    - Cite sources with [Source N] notation
    - Explicitly state when information is insufficient

    If retrieval confidence is low, an extra warning is injected into the prompt
    to make the model more conservative.
    """
    start = time.perf_counter()

    # Format sources for the prompt
    formatted_sources, source_labels = _format_sources(retrieval_result)

    # Add low-confidence warning if retrieval flagged it
    confidence_warning = ""
    if retrieval_result.is_low_confidence:
        confidence_warning = LOW_CONFIDENCE_WARNING

    # Build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    # Create the chain: prompt -> LLM -> parse to string
    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()

    # Run generation
    answer = chain.invoke({
        "query": query,
        "formatted_sources": formatted_sources,
        "confidence_warning": confidence_warning,
    })

    elapsed = (time.perf_counter() - start) * 1000

    return GenerationResult(
        answer=answer.strip(),
        is_low_confidence=retrieval_result.is_low_confidence,
        generation_time_ms=round(elapsed, 2),
        source_labels=source_labels,
    )
