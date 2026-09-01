"""
TrustRAG — Validation Layer (Core Differentiator)
Three-stage validation pipeline that grades generated answers for groundedness:

  Stage 1: Groundedness Check — LLM judge evaluates each claim against source chunks
  Stage 2: Citation Cross-Check — verifies cited sources actually exist in retrieval
  Stage 3: Decision Policy — determines show/warn/refuse based on scores

This is what separates TrustRAG from a naive RAG pipeline. Every answer is graded
before being shown to the user, catching hallucinations and unsupported claims.
"""

import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from retriever import RetrievedChunk


@dataclass
class ClaimVerdict:
    """Verdict for a single claim in the generated answer."""
    claim: str
    verdict: str  # "supported", "partial", "unsupported"
    source_chunk: str | None  # chunk_id that supports it, or None
    reasoning: str = ""


@dataclass
class ValidationResult:
    """Complete validation output for a generated answer."""
    overall_groundedness_score: int = 0
    claims: list[ClaimVerdict] = field(default_factory=list)
    fabricated_citations: list[str] = field(default_factory=list)
    recommendation: str = "refuse"  # "show" | "show_with_warning" | "refuse"
    validation_time_ms: float = 0.0
    judge_raw_response: str = ""  # raw judge output for debugging


# ── Judge Prompt ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a factual accuracy judge. Your job is to evaluate whether a generated answer is grounded in the provided source chunks.

You will receive:
1. A generated answer that claims to be based on source chunks
2. The actual source chunks that were retrieved

Your task: Break the answer into individual factual claims, and for each claim determine whether it is:
- "supported": directly and fully supported by the source chunks
- "partial": partially supported (some aspects are in sources, some are not)
- "unsupported": not supported by any source chunk (hallucinated or fabricated)

IMPORTANT RULES:
- Be strict. A claim is "supported" only if the source chunks contain clear evidence for it.
- Generic phrasing like "according to the sources" is not itself a claim — evaluate the factual content.
- If the answer says "the sources don't contain information about X", that's a valid truthful statement — mark it as "supported" if accurate.
- For the source_chunk field, use the chunk_id from the source metadata (e.g., "sample.txt::chunk_0").

Calculate the overall_groundedness_score as follows:
- Each "supported" claim = 1.0 points
- Each "partial" claim = 0.5 points
- Each "unsupported" claim = 0.0 points
- Score = (total_points / number_of_claims) * 100, rounded to nearest integer

Respond with ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "claims": [
    {
      "claim": "the specific factual claim",
      "verdict": "supported",
      "source_chunk": "filename.txt::chunk_0",
      "reasoning": "brief explanation"
    }
  ],
  "overall_groundedness_score": 85
}"""

JUDGE_USER_PROMPT = """--- GENERATED ANSWER ---
{answer}

--- SOURCE CHUNKS ---
{formatted_chunks}

Evaluate the groundedness of the generated answer above against the source chunks."""


def _get_judge_llm():
    """
    Initialize the judge LLM (uses the cheap/fast model configured for validation).
    """
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.JUDGE_MODEL,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY,
        )
    elif config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.JUDGE_MODEL,
            temperature=0,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")


def _format_chunks_for_judge(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for the judge prompt with IDs and content."""
    if not chunks:
        return "(No source chunks were retrieved)"

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.chunk_id
        source = chunk.metadata.get("source", "unknown")
        lines.append(
            f"[Chunk {i}] (id: {chunk_id}, from: {source}):\n{chunk.content}"
        )
    return "\n\n".join(lines)


def _parse_judge_response(raw_response: str) -> tuple[list[ClaimVerdict], int]:
    """
    Parse the judge's JSON response into structured ClaimVerdicts.
    Handles common JSON issues (markdown fences, trailing commas).
    Returns (claims_list, groundedness_score).
    """
    # Strip markdown code fences if present
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        # Remove closing fence
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: try to extract JSON object from the response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                # Complete parse failure — return worst-case
                return [ClaimVerdict(
                    claim="(Unable to parse judge response)",
                    verdict="unsupported",
                    source_chunk=None,
                    reasoning="Judge response was not valid JSON",
                )], 0
        else:
            return [ClaimVerdict(
                claim="(Unable to parse judge response)",
                verdict="unsupported",
                source_chunk=None,
                reasoning="Judge response was not valid JSON",
            )], 0

    # Extract claims
    claims = []
    for item in data.get("claims", []):
        claims.append(ClaimVerdict(
            claim=item.get("claim", ""),
            verdict=item.get("verdict", "unsupported"),
            source_chunk=item.get("source_chunk"),
            reasoning=item.get("reasoning", ""),
        ))

    score = int(data.get("overall_groundedness_score", 0))
    return claims, score


# ── Stage 1: Groundedness Check ──────────────────────────────────────────────

def _check_groundedness(
    answer: str,
    chunks: list[RetrievedChunk],
) -> tuple[list[ClaimVerdict], int, str]:
    """
    Make a second LLM call (the "judge") to evaluate whether each claim in the
    answer is supported by the retrieved source chunks.

    Returns (claims, score, raw_response).
    """
    formatted_chunks = _format_chunks_for_judge(chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", JUDGE_USER_PROMPT),
    ])

    llm = _get_judge_llm()
    chain = prompt | llm | StrOutputParser()

    # Run the judge
    raw_response = chain.invoke({
        "answer": answer,
        "formatted_chunks": formatted_chunks,
    })

    claims, score = _parse_judge_response(raw_response)
    return claims, score, raw_response


# ── Stage 2: Citation Cross-Check ────────────────────────────────────────────

def _check_citations(
    answer: str,
    chunks: list[RetrievedChunk],
    source_labels: list[str],
) -> list[str]:
    """
    Verify that every [Source N] citation in the answer references a real chunk
    that was actually retrieved. Catches cases where the model invents citations.

    Returns a list of fabricated citation labels (e.g., ["Source 5"]).
    """
    # Extract all [Source N] citations from the answer
    cited_sources = re.findall(r"\[Source (\d+)\]", answer)
    cited_numbers = set(int(n) for n in cited_sources)

    # Valid source numbers are 1..len(chunks)
    valid_numbers = set(range(1, len(chunks) + 1))

    # Find fabricated citations (referenced but not in retrieval)
    fabricated = []
    for num in cited_numbers:
        if num not in valid_numbers:
            fabricated.append(f"Source {num}")

    return fabricated


# ── Stage 3: Decision Policy ─────────────────────────────────────────────────

def _decide(score: int, is_low_confidence: bool, fabricated_count: int) -> str:
    """
    Apply the decision policy based on groundedness score, retrieval confidence,
    and citation integrity.

    Policy:
        score >= 80 and no issues → "show"
        score 50-79 or minor issues → "show_with_warning"
        score < 50 or major issues → "refuse"

    Low confidence retrieval makes the policy stricter:
        if low_confidence and score < 70 → "refuse"
    """
    # Fabricated citations are a serious red flag
    if fabricated_count > 0:
        score = max(0, score - (fabricated_count * 15))

    # Low confidence retrieval → stricter threshold
    if is_low_confidence and score < 70:
        return "refuse"

    if score >= config.SCORE_SHOW_THRESHOLD:
        return "show"
    elif score >= config.SCORE_WARN_THRESHOLD:
        return "show_with_warning"
    else:
        return "refuse"


# ── Main Validation Entry Point ──────────────────────────────────────────────

def validate(
    answer: str,
    chunks: list[RetrievedChunk],
    source_labels: list[str],
    is_low_confidence: bool = False,
) -> ValidationResult:
    """
    Run the full 3-stage validation pipeline on a generated answer.

    Args:
        answer: The generated RAG answer to validate
        chunks: The retrieved source chunks used for generation
        source_labels: Labels like ["Source 1", "Source 2"] for citation checking
        is_low_confidence: Whether retrieval flagged this as low confidence

    Returns:
        ValidationResult with groundedness score, per-claim verdicts,
        fabricated citations, and show/warn/refuse recommendation.
    """
    start = time.perf_counter()

    # Handle edge case: no chunks retrieved
    if not chunks:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            overall_groundedness_score=0,
            claims=[ClaimVerdict(
                claim="No source chunks were retrieved",
                verdict="unsupported",
                source_chunk=None,
                reasoning="Cannot validate without source material",
            )],
            fabricated_citations=[],
            recommendation="refuse",
            validation_time_ms=round(elapsed, 2),
        )

    # Stage 1: Groundedness check (LLM judge call)
    claims, score, raw_response = _check_groundedness(answer, chunks)

    # Stage 2: Citation cross-check
    fabricated = _check_citations(answer, chunks, source_labels)

    # Stage 3: Decision policy
    recommendation = _decide(score, is_low_confidence, len(fabricated))

    elapsed = (time.perf_counter() - start) * 1000

    return ValidationResult(
        overall_groundedness_score=score,
        claims=claims,
        fabricated_citations=fabricated,
        recommendation=recommendation,
        validation_time_ms=round(elapsed, 2),
        judge_raw_response=raw_response,
    )
