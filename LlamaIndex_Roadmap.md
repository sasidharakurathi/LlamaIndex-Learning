# LlamaIndex Mastery Roadmap

## MODULE 1 - Core Foundations (LlamaIndex Mental Model)
### Goal: Understand how LlamaIndex actually works internally.
1. What is LlamaIndex? (Architecture vs LangChain vs raw LLM APIs)
2. Documents, Nodes, and Chunking Strategies
3. Index Types Deep Dive:
    - VectorStoreIndex
    - SummaryIndex
    - TreeIndex
4. Storage Context & Persistence (disk, S3, DB)
5. Query Engine vs Retriever

## MODULE 2 - Building Production-Grade RAG
### Goal: Move from toy RAG → reliable systems.
6. Embeddings Deep Dive (Gemini, local, hybrid)
7. Advanced Chunking (semantic splitting, windowing)
8. Retrieval Strategies:
    - Similarity search
    - MMR (Max Marginal Relevance)
    - Hybrid search (BM25 + vector)
9. Response Synthesis (Refine, Compact, Tree Summarize)
10. Evaluation & Debugging RAG (faithfulness, hallucination detection)

## MODULE 3 - Data Connectors & Pipelines
### Goal: Build real ingestion systems.
11. Data Connectors (PDFs, APIs, DBs, Notion, etc.)
12. Ingestion Pipelines (Transformations, metadata enrichment)
13. Incremental Indexing & Real-Time Updates

## MODULE 4 - Advanced Retrieval Systems
### Goal: Build smart retrieval beyond basic similarity.
14. Metadata Filtering & Structured Retrieval
15. Recursive Retrieval (retrieving over retrieved nodes)
16. Router Query Engine (Multi-Index Intelligent Routing)
17. Graph RAG (Relationship-Aware Retrieval)

## MODULE 5 - Agents & Tool Use
18. LlamaIndex Agents (ReAct + Function Calling)