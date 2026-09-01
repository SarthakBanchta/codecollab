# Retrieval-Augmented Generation (RAG): How It Works

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture pattern that combines 
information retrieval with large language model (LLM) generation. Instead of relying 
solely on the knowledge baked into a model's weights during training, RAG systems 
dynamically fetch relevant documents from an external knowledge base and use them as 
context for generating answers.

The core insight behind RAG is simple: LLMs are good at reasoning and generating natural 
language, but they hallucinate facts and their training data becomes stale. By grounding 
generation in retrieved documents, RAG systems can provide more accurate, up-to-date, and 
verifiable answers.

## The RAG Pipeline

A typical RAG pipeline consists of three stages:

### 1. Ingestion (Offline)
Documents are preprocessed and stored in a searchable format:
- **Chunking**: Documents are split into smaller segments (typically 200-1000 tokens) 
  because embedding models have limited context windows and smaller chunks allow more 
  precise retrieval.
- **Embedding**: Each chunk is converted into a dense vector representation using an 
  embedding model (e.g., OpenAI's text-embedding-3-small, or open-source models like 
  BGE or E5).
- **Storage**: Vectors are stored in a vector database (ChromaDB, Pinecone, Weaviate, 
  Qdrant) along with the original text and metadata.

### 2. Retrieval (Online)
When a user asks a question:
- The query is embedded using the same embedding model.
- A similarity search finds the top-k most relevant chunks from the vector database.
- Similarity is typically measured using cosine similarity or L2 distance.
- The retrieved chunks become the "context" for generation.

### 3. Generation (Online)
The LLM receives a prompt containing:
- The user's question
- The retrieved context chunks
- Instructions to answer based only on the provided context

The model generates an answer grounded in the retrieved information.

## Common Failure Modes

RAG systems are not perfect. Common issues include:

1. **Retrieval failures**: The relevant information exists in the knowledge base but 
   isn't retrieved because the query embedding doesn't match well with the chunk embedding. 
   This can happen with ambiguous queries or when the chunking strategy splits relevant 
   information across multiple chunks.

2. **Hallucination despite context**: Even when correct chunks are retrieved, the LLM 
   may still hallucinate — generating plausible-sounding information that isn't actually 
   in the retrieved context. This is especially common when the model's parametric 
   knowledge conflicts with the retrieved information.

3. **Missing context**: The knowledge base simply doesn't contain the information needed 
   to answer the question, but the system generates an answer anyway using its parametric 
   knowledge.

4. **Citation fabrication**: When asked to cite sources, LLMs sometimes invent plausible 
   but non-existent citations or attribute information to the wrong source chunk.

## Mitigation Strategies

Several techniques can improve RAG reliability:

- **Confidence gating**: Setting similarity score thresholds to filter out low-quality 
  retrievals before they reach the generator.
- **Answer validation**: Using a second LLM call to verify whether the generated answer 
  is actually grounded in the retrieved context (sometimes called "groundedness checking").
- **Structured citations**: Requiring the model to explicitly cite which source supports 
  each claim, making it easier to verify accuracy.
- **Hybrid search**: Combining dense vector search with keyword-based (BM25) search for 
  better recall.
- **Re-ranking**: Using a cross-encoder model to re-score retrieved chunks for better 
  precision before passing them to the generator.

## Key Metrics

When evaluating a RAG system, important metrics include:
- **Retrieval precision**: What fraction of retrieved chunks are actually relevant?
- **Retrieval recall**: What fraction of all relevant chunks were retrieved?
- **Answer faithfulness**: Is the generated answer faithful to the retrieved context?
- **Answer relevance**: Does the answer actually address the user's question?
- **Latency**: End-to-end time from query to answer, including retrieval and generation.
