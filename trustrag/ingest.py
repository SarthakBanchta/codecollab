"""
TrustRAG — Document Ingestion Pipeline
Loads documents from data/, chunks them, embeds, and stores in ChromaDB.

Supported formats: .txt, .md, .pdf
Usage:
    python ingest.py           # ingest all files in data/
    python ingest.py --reset   # clear existing collection and re-ingest
"""

import os
import sys
import glob
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config


# ── Document Loaders ─────────────────────────────────────────────────────────

# Map file extensions to loader factories.
# .md files use TextLoader (fast, no extra dependencies needed).
LOADER_MAP = {
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md":  lambda path: TextLoader(path, encoding="utf-8"),
    ".pdf": lambda path: PyPDFLoader(path),
}


def load_documents(data_dir: str) -> list:
    """
    Walk the data directory and load all supported files.
    Returns a flat list of LangChain Document objects.
    """
    all_docs = []
    supported_extensions = set(LOADER_MAP.keys())

    if not os.path.isdir(data_dir):
        print(f"[Error] Data directory not found: {data_dir}")
        print("   Create it and add your .txt, .md, or .pdf files.")
        sys.exit(1)

    # Collect all files matching supported extensions
    files_found = []
    for ext in supported_extensions:
        pattern = os.path.join(data_dir, f"**/*{ext}")
        files_found.extend(glob.glob(pattern, recursive=True))

    if not files_found:
        print(f"[Warning] No supported files found in {data_dir}")
        print(f"   Supported formats: {', '.join(supported_extensions)}")
        return []

    for filepath in sorted(files_found):
        ext = os.path.splitext(filepath)[1].lower()
        loader_factory = LOADER_MAP.get(ext)
        if not loader_factory:
            continue

        try:
            loader = loader_factory(filepath)
            docs = loader.load()

            # Attach source filename to metadata
            filename = os.path.relpath(filepath, data_dir)
            for doc in docs:
                doc.metadata["source"] = filename

            all_docs.extend(docs)
            print(f"  [OK] Loaded: {filename} ({len(docs)} page(s))")
        except Exception as e:
            print(f"  [Skip] Skipped {filepath}: {e}")

    return all_docs


def chunk_documents(documents: list) -> list:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Adds chunk_index and chunk_id to each chunk's metadata for citation tracking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk indexing metadata per source file
    # Track chunk index per source so IDs are like "file.txt::chunk_0"
    source_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_chunk_counts.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{source}::chunk_{idx}"
        source_chunk_counts[source] = idx + 1

    return chunks


def create_vector_store(chunks: list, reset: bool = False) -> None:
    """
    Embed chunks and store in a persistent ChromaDB collection.
    If reset=True, deletes the existing collection first.
    """
    # Validate API key before attempting to embed
    is_valid, error_msg = config.validate_api_keys()
    if not is_valid:
        print(f"[Error] {error_msg}")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )

    if reset and os.path.exists(config.CHROMA_PERSIST_DIR):
        import shutil
        print("[Info] Clearing existing vector store...")
        shutil.rmtree(config.CHROMA_PERSIST_DIR)

    print(f"[Info] Embedding {len(chunks)} chunks...")

    # Use LangChain's Chroma integration for clean document ingestion
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    # Verify the collection was created
    collection_count = vector_store._collection.count()
    print(f"[OK] Vector store created: {collection_count} chunks indexed")
    print(f"     Persisted to: {config.CHROMA_PERSIST_DIR}")


def main():
    parser = argparse.ArgumentParser(description="TrustRAG Document Ingestion")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing vector store and re-ingest from scratch",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TrustRAG - Document Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load documents
    print(f"\n[Info] Loading documents from: {config.DATA_DIR}")
    documents = load_documents(config.DATA_DIR)
    if not documents:
        print("No documents to process. Exiting.")
        sys.exit(0)

    # Step 2: Chunk documents
    print(f"\n[Info] Chunking {len(documents)} document(s)...")
    print(f"       chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}")
    chunks = chunk_documents(documents)
    print(f"       -> {len(chunks)} chunks created")

    # Step 3: Embed and store
    print(f"\n[Info] Embedding and storing in ChromaDB...")
    create_vector_store(chunks, reset=args.reset)

    print("\n[OK] Ingestion complete! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
