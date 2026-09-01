"""
TrustRAG Configuration
Centralized config loaded from .env with sensible defaults.
All tunable parameters live here — nothing is hardcoded in other modules.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ── LLM Provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Model names
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")  # cheap model for validation
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ── Retrieval Settings ───────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))


# ── Chunking Settings ────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


# ── Storage Paths ────────────────────────────────────────────────────────────
# Resolve paths relative to this file's directory (trustrag/)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", os.path.join(_BASE_DIR, "chroma_db")
)
SQLITE_DB_PATH = os.getenv(
    "SQLITE_DB_PATH", os.path.join(_BASE_DIR, "trustrag_logs.db")
)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_BASE_DIR, "data"))

# ChromaDB collection name
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "trustrag_docs")


# ── Validation Thresholds ────────────────────────────────────────────────────
# Groundedness score thresholds for decision policy
SCORE_SHOW_THRESHOLD = int(os.getenv("SCORE_SHOW_THRESHOLD", "80"))
SCORE_WARN_THRESHOLD = int(os.getenv("SCORE_WARN_THRESHOLD", "50"))


def validate_api_keys() -> tuple[bool, str]:
    """
    Check that the required API key is set for the configured provider.
    Returns (is_valid, error_message).
    """
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            return False, (
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set it as an environment variable.\n"
                "Get your key at: https://platform.openai.com/api-keys"
            )
    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            return False, (
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or set it as an environment variable.\n"
                "Get your key at: https://console.anthropic.com/settings/keys"
            )
    else:
        return False, (
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Supported values: 'openai', 'anthropic'"
        )
    return True, ""
