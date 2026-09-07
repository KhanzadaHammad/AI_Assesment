# Agentic Search Intelligence System

The system assesses a brand's visibility in search engines and AI answers (ChatGPT, Local Ollama) for specific target queries. It takes a brand profile and research question, plans required search queries, retrieves live/mock SERP data via DataForSEO, normalizes rankings and competitors, indexes content snippets for RAG, reasons over visibility gaps, and synthesizes a structured report with recommendations.

---

## Architecture

```text
[Start Pipeline]
      │
      ▼
[1. Query Planner] ──(LLM: OpenAI or Ollama parses intent into search arguments)
      │
      ▼
[2. Search Retrieval] ──(Calls DataForSEO tool w/ backoff & jitter)
      │
      ├─► [Transient Error / Exhausted Retries] ──► [Fallback Handler]
      │                                                     │
      ▼                                                     ▼
[3. Extraction / Normalizer] ◄──────────────────────────────┘
      │ (Pydantic parsing of SERP ranks, URLs, & snippets)
      ▼
[4. RAG Indexer & Vector Store] ──(Embedder: OpenAI / Ollama + In-Memory Vector Store)
      │
      ▼
[5. Analysis / Synthesis] ──(LLM analyzes ranking gaps + vector-retrieved context)
      │
      ▼
[6. Report Generator] ──(LLM synthesizes final JSON + executive brief)
      │
      ▼
    [END]
```

Directed Acyclic Graph (DAG): Built with LangGraph.

Single Responsibility Nodes: Separate atomic agents for Planning, Retrieval, Extraction/Normalization, RAG Indexing, Analysis/Synthesis, and Reporting.

Model Agnostic Backend: A unified factory supporting OpenAI or local Ollama for both chat and embeddings.

REST API Layer: FastAPI exposing business profiles, run execution, query inspection, and recommendations.

In-Memory Store: A persistence layer tracking profiles, runs, queries, and recommendations.

Tools Functionality
Schema Validation: Uses Pydantic (SerpQueryArgs) to validate incoming arguments (keywords, location code, language) before making requests.

API Wrapper: Interfaces directly with DataForSEO (or its mock stub) per individual search call.

Resilience & Retry: Handles transient failures with exponential backoff and randomized jitter.

Error Classification: Distinguishes retryable network/rate-limit errors from non-retryable errors, routing persistent failures to a fallback recovery node.

Call Counts: LLM & Embedder
For a single pipeline run with 3 planned queries:

LLM Calls (3 total)
Query Planner Node: 1 call

Analysis / Synthesis Node: 1 call

Report Node: 1 call

Embedder Calls (2 total)
Document Indexing: Embeds the extracted SERP snippets into the vector store.

Query Search: Embeds the target domain query to retrieve top-k similar ranking snippets.
