# Customer Support RAG Pipeline Task

## Task Overview
You are tasked with improving the semantic search and retrieval accuracy of a customer support RAG system. The current pipeline struggles with poor retrieval due to oversized chunks and missing metadata during ingestion, and a basic top-k search using default settings. Your job is to:
- Reimplement the chunking and embedding initialization so that each document is split into 512-token chunks with 200-token overlap, and metadata (category, priority, date) is attached at the chunk level.
- Optimize retrieval logic so top-5 most relevant chunks are returned using cosine similarity, leveraging Chroma database features.
- Validate improvements: manually inspect results for sample queries (`sample_queries.txt`) and check recall@k against provided gold answers.

## Retrieval System Gaps
- The core retrieval issue is with chunk size/overlap and missing metadata on chunks, leading to context loss and low relevance.
- Retrieval logic is too basic—needs configuration for cosine similarity and strict k=5 returns.
- Metadata is not leveraged and should be implemented as part of ingestion for possible future advanced filtering.

## Your Focus
- `db/init_vector_db.py` (chunking, embedding, metadata)
- `rag/rag_retrieval.py` (retrieval/top-k returning logic)
- All infrastructure (Chroma DB setup, Docker, database connectors) is provided and automated via `run.sh`.
- **You do not need to modify infrastructure, only chunking/embedding and retrieval Python code.**

## Database Access Info
- DB: Chroma
- Host: `<DROPLET_IP>`, Port: 8000
- Collection: `support_documents`
- Vector dim: 384
- Chunk metadata: doc_id, category, priority, date, chunk_idx
- Documents: 8,000 support docs (after chunking, expect thousands of chunks)

## Objectives
- Accurate, contextually relevant retrieval for customer support queries
- Efficient chunking/embedding pipeline (with proper overlap and attached metadata)
- Top-5 cosine similarity matches per query, demonstrably better than initial baseline

## How to Verify
- Use queries in `sample_queries.txt` for manual spot checks & compare to expected `sample_queries_answers.txt` (not provided, but you can judge relevance)
- For each query, verify that chunked answers returned are on-topic and precise, drawing from correct documents and category
- Improved recall@5 should be observable for at least 3 queries compared to baseline
