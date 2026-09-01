"""
TrustRAG — Streamlit UI
Chat interface with validation badges, source attribution,
and an observability dashboard showing pipeline telemetry.

Run with: streamlit run app.py
"""

import sys
import os
import streamlit as st

# Ensure trustrag/ is on the import path when running via `streamlit run app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import logger
from retriever import retrieve
from generator import generate
from validator import validate


# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrustRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── API Key Validation ───────────────────────────────────────────────────────

def check_setup() -> bool:
    """Verify API keys and vector store are ready. Returns True if OK."""
    is_valid, error_msg = config.validate_api_keys()
    if not is_valid:
        st.error(f"🔑 **API Key Missing**\n\n{error_msg}")
        st.info(
            "**Quick fix:** Copy `.env.example` to `.env` and add your API key:\n"
            "```\ncp .env.example .env\n# Edit .env and add your key\n```"
        )
        return False

    # Check if vector store exists
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        st.warning(
            "📦 **Vector store not found.** You need to ingest documents first.\n\n"
            "Run this command in the `trustrag/` directory:\n"
            "```bash\npython ingest.py\n```"
        )
        return False

    return True


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with Settings and Observability Dashboard tabs."""
    with st.sidebar:
        st.title("🛡️ TrustRAG")
        st.caption("RAG with built-in guardrails")

        tab_settings, tab_dashboard = st.tabs(["⚙️ Settings", "📊 Dashboard"])

        # ── Settings Tab ─────────────────────────────────────────────────
        with tab_settings:
            st.subheader("Configuration")
            st.markdown(f"**Provider:** `{config.LLM_PROVIDER}`")
            st.markdown(f"**Generation model:** `{config.GENERATION_MODEL}`")
            st.markdown(f"**Judge model:** `{config.JUDGE_MODEL}`")
            st.markdown(f"**Embedding model:** `{config.EMBEDDING_MODEL}`")

            st.divider()
            st.markdown(f"**Top-K chunks:** `{config.TOP_K}`")
            st.markdown(f"**Similarity threshold:** `{config.SIMILARITY_THRESHOLD}`")
            st.markdown(f"**Show threshold:** `≥ {config.SCORE_SHOW_THRESHOLD}`")
            st.markdown(f"**Warn threshold:** `≥ {config.SCORE_WARN_THRESHOLD}`")

            st.divider()
            st.caption("Edit `.env` to change settings, then restart the app.")

        # ── Observability Dashboard Tab ──────────────────────────────────
        with tab_dashboard:
            render_dashboard()


def render_dashboard():
    """Render the observability dashboard with metrics and flagged queries."""
    st.subheader("Observability")

    try:
        logger.init_db()
        stats = logger.get_stats()
    except Exception as e:
        st.error(f"Could not load stats: {e}")
        return

    # Metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", stats.get("total_queries", 0))
        st.metric(
            "Avg Groundedness",
            f"{stats.get('avg_groundedness', 0):.0f}%",
        )
    with col2:
        st.metric(
            "Avg Latency",
            f"{stats.get('avg_latency_ms', 0):.0f}ms",
        )
        refused = stats.get("refused_count", 0) or 0
        warned = stats.get("warned_count", 0) or 0
        total = stats.get("total_queries", 0) or 1
        flag_rate = ((refused + warned) / total) * 100 if total > 0 else 0
        st.metric("Flag Rate", f"{flag_rate:.0f}%")

    # Decision breakdown
    st.divider()
    shown = stats.get("shown_count", 0) or 0
    st.markdown(
        f"🟢 Shown: **{shown}** · "
        f"🟡 Warned: **{warned}** · "
        f"🔴 Refused: **{refused}**"
    )

    # Latency breakdown
    st.divider()
    st.caption("**Avg Latency Breakdown**")
    col_r, col_g, col_v = st.columns(3)
    with col_r:
        st.metric("Retrieval", f"{stats.get('avg_retrieval_ms', 0):.0f}ms")
    with col_g:
        st.metric("Generation", f"{stats.get('avg_generation_ms', 0):.0f}ms")
    with col_v:
        st.metric("Validation", f"{stats.get('avg_validation_ms', 0):.0f}ms")

    # Flagged queries table
    st.divider()
    st.caption("**Flagged / Refused Queries**")
    flagged = logger.get_flagged_queries(limit=20)
    if flagged:
        for q in flagged:
            decision = q.get("decision", "")
            icon = "🔴" if decision == "refuse" else "🟡"
            score = q.get("groundedness_score", "?")
            st.markdown(
                f"{icon} **Score: {score}%** | `{q.get('query_text', '')[:60]}...`"
            )
    else:
        st.caption("No flagged queries yet.")


# ── Chat Interface ───────────────────────────────────────────────────────────

def render_confidence_badge(recommendation: str, score: int):
    """Show a colored confidence badge based on the validation decision."""
    if recommendation == "show":
        st.success(f"🟢 **Grounded** — Confidence score: {score}%")
    elif recommendation == "show_with_warning":
        st.warning(f"🟡 **Partially Grounded** — Confidence score: {score}% — Some claims may be unverified")
    else:
        st.error(f"🔴 **Refused** — Confidence score: {score}% — Not enough grounded information")


def render_sources(chunks):
    """Show which sources were used in an expandable panel."""
    if not chunks:
        return

    with st.expander(f"📚 Sources Used ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            chunk_id = chunk.chunk_id
            score = chunk.score
            st.markdown(f"**[Source {i}]** `{source}` (score: {score:.2f})")
            st.caption(chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""))
            if i < len(chunks):
                st.divider()


def render_validation_details(validation_result, retrieval_time, generation_time):
    """Show the judge's per-claim verdicts in an expandable panel."""
    with st.expander("🔍 Validation Details", expanded=False):
        # Groundedness score bar
        st.markdown(f"**Overall Groundedness Score: {validation_result.overall_groundedness_score}%**")
        st.progress(validation_result.overall_groundedness_score / 100)

        # Per-claim verdicts
        st.markdown("**Per-Claim Verdicts:**")
        for claim in validation_result.claims:
            verdict = claim.verdict
            if verdict == "supported":
                icon = "✅"
            elif verdict == "partial":
                icon = "⚠️"
            else:
                icon = "❌"

            st.markdown(f"{icon} **{verdict.upper()}**: {claim.claim}")
            if claim.source_chunk:
                st.caption(f"  ↳ Source: `{claim.source_chunk}`")
            if claim.reasoning:
                st.caption(f"  ↳ Reasoning: {claim.reasoning}")

        # Fabricated citations
        if validation_result.fabricated_citations:
            st.divider()
            st.error(
                f"⚠️ **Fabricated Citations Detected:** "
                f"{', '.join(validation_result.fabricated_citations)}"
            )

        # Latency breakdown
        st.divider()
        st.markdown("**Latency Breakdown:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retrieval", f"{retrieval_time:.0f}ms")
        with col2:
            st.metric("Generation", f"{generation_time:.0f}ms")
        with col3:
            st.metric("Validation", f"{validation_result.validation_time_ms:.0f}ms")


REFUSAL_MESSAGE = (
    "I don't have enough grounded information to answer this confidently. "
    "The retrieved sources either don't cover this topic well enough, or the "
    "generated answer couldn't be adequately verified against the sources.\n\n"
    "**Suggestions:**\n"
    "- Try rephrasing your question\n"
    "- Ask about a topic covered in the ingested documents\n"
    "- Check that relevant documents have been ingested"
)


def process_query(query: str):
    """
    Run the full TrustRAG pipeline: retrieve → generate → validate → log → display.
    """
    # Step 1: Retrieve
    with st.spinner("🔍 Retrieving relevant chunks..."):
        retrieval_result = retrieve(query)

    # Step 2: Generate
    with st.spinner("✍️ Generating answer..."):
        generation_result = generate(query, retrieval_result)

    # Step 3: Validate
    with st.spinner("🧑‍⚖️ Validating groundedness..."):
        validation_result = validate(
            answer=generation_result.answer,
            chunks=retrieval_result.chunks,
            source_labels=generation_result.source_labels,
            is_low_confidence=retrieval_result.is_low_confidence,
        )

    # Step 4: Log everything
    try:
        logger.init_db()
        logger.log_query(
            query_text=query,
            retrieved_chunks=[
                {"chunk_id": c.chunk_id, "score": c.score}
                for c in retrieval_result.chunks
            ],
            generated_answer=generation_result.answer,
            groundedness_score=validation_result.overall_groundedness_score,
            claims_verdict=[
                {
                    "claim": c.claim,
                    "verdict": c.verdict,
                    "source_chunk": c.source_chunk,
                    "reasoning": c.reasoning,
                }
                for c in validation_result.claims
            ],
            fabricated_citations=validation_result.fabricated_citations,
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            generation_time_ms=generation_result.generation_time_ms,
            validation_time_ms=validation_result.validation_time_ms,
            decision=validation_result.recommendation,
            is_low_confidence=retrieval_result.is_low_confidence,
        )
    except Exception as e:
        st.toast(f"⚠️ Logging failed: {e}", icon="⚠️")

    # Step 5: Display results based on recommendation
    recommendation = validation_result.recommendation

    if recommendation == "refuse":
        # Show refusal message
        st.markdown(REFUSAL_MESSAGE)
        render_confidence_badge(recommendation, validation_result.overall_groundedness_score)
        render_sources(retrieval_result.chunks)
        render_validation_details(
            validation_result,
            retrieval_result.retrieval_time_ms,
            generation_result.generation_time_ms,
        )
        return REFUSAL_MESSAGE
    else:
        # Show the answer (with or without warning)
        st.markdown(generation_result.answer)
        render_confidence_badge(recommendation, validation_result.overall_groundedness_score)
        render_sources(retrieval_result.chunks)
        render_validation_details(
            validation_result,
            retrieval_result.retrieval_time_ms,
            generation_result.generation_time_ms,
        )
        return generation_result.answer


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render sidebar
    render_sidebar()

    # Main area header
    st.title("🛡️ TrustRAG")
    st.caption(
        "Ask questions about your documents. Every answer is validated for "
        "groundedness before being shown."
    )

    # Check setup
    if not check_setup():
        return

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant response
        with st.chat_message("assistant"):
            response_text = process_query(prompt)

        # Store assistant response in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
        })


if __name__ == "__main__":
    main()
