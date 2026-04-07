# GraphRAG Demo: Design Proposal

## A Tale of Two Cities + AstraeaDB + Ollama

### Vision

Demonstrate that a modest 4B-parameter LLM (gemma3:4b), when given structured access to a knowledge graph via MCP, can answer complex literary questions about *A Tale of Two Cities* with accuracy that would normally require a much larger model or the full novel in context. The key insight: **graph structure compensates for model size**.

---

## What We're Building

A three-stage system:

1. **Knowledge Graph Builder** (Python, offline) -- Extracts entities, relationships, and text passages from the novel and loads them into AstraeaDB with vector embeddings
2. **AstraeaDB Server** -- Serves the knowledge graph with MCP enabled
3. **Interactive Demo** (Python CLI or web) -- User asks questions; gemma3:4b uses AstraeaDB's MCP tools to retrieve relevant graph context, then generates grounded answers

### Architecture

```
                         ┌───────────────────┐
                         │   User Question    │
                         └────────┬──────────┘
                                  │
                         ┌────────▼──────────┐
                         │    gemma3:4b       │
                         │    (via Ollama)    │
                         └────────┬──────────┘
                                  │ MCP tool calls
                         ┌────────▼──────────┐
                         │   AstraeaDB MCP   │
                         │   Server (stdio)   │
                         └────────┬──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │              │
              ┌─────▼────┐ ┌─────▼────┐ ┌──────▼─────┐
              │  Graph    │ │  Vector  │ │ Algorithms │
              │ Traversal │ │  Search  │ │  (Louvain, │
              │ (BFS,     │ │  (HNSW)  │ │  PageRank) │
              │  paths)   │ │          │ │            │
              └──────────┘ └──────────┘ └────────────┘
                         Knowledge Graph
                    (A Tale of Two Cities)
```

---

## Knowledge Graph Schema

### Node Types

| Label | Properties | Embedding? | Description |
|-------|-----------|------------|-------------|
| **Character** | name, aliases, description, first_appearance | Yes (description) | People in the novel (Sydney Carton, Lucie Manette, etc.) |
| **Location** | name, description, city, country | Yes (description) | Places (Tellson's Bank, the Bastille, Defarge wine shop) |
| **Event** | name, description, chapter, book_number | Yes (description) | Key plot events (storming of the Bastille, Darnay's trial) |
| **Theme** | name, description | Yes (description) | Major themes (resurrection, sacrifice, duality, revolution) |
| **Chapter** | title, book_number, chapter_number, summary | Yes (summary) | Each chapter as a structural node |
| **TextChunk** | text, chapter, book_number, chunk_index | Yes (text) | Paragraphs/passages for grounding answers in actual prose |

### Edge Types

| Edge Type | From -> To | Properties | Description |
|-----------|-----------|------------|-------------|
| **RELATED_TO** | Character -> Character | relationship (e.g., "father", "husband", "rival"), description | Family/social relationships |
| **LOCATED_IN** | Character -> Location | context | Where characters live, work, or are imprisoned |
| **PARTICIPATES_IN** | Character -> Event | role | Character involvement in events |
| **OCCURS_IN** | Event -> Location | -- | Where events take place |
| **OCCURS_DURING** | Event -> Chapter | -- | Temporal anchoring |
| **EMBODIES** | Character -> Theme | description | Thematic associations |
| **MENTIONED_IN** | Character/Location/Event -> TextChunk | -- | Provenance links to source text |
| **NEXT_CHAPTER** | Chapter -> Chapter | -- | Narrative sequence |

### Estimated Scale

- ~30-40 Character nodes
- ~15-20 Location nodes
- ~30-40 Event nodes
- ~6-8 Theme nodes
- 45 Chapter nodes
- ~300-500 TextChunk nodes (paragraphs grouped by scene)
- ~500-1000 edges

This is modest enough to build reliably with an LLM extraction pipeline, while rich enough to demonstrate GraphRAG's advantages.

---

## Entity Extraction Strategy

We can't use gemma3:4b for extraction -- it's too small to reliably produce structured JSON from long text. Two practical options:

### Option A: LLM-Assisted Extraction (Recommended)

Use a larger model (qwen3-coder-next, already installed locally) for the one-time offline extraction pipeline:

1. **Chunk the novel** into chapter-sized segments
2. **Schema-guided extraction** -- prompt the model with our schema and each chapter, asking for structured JSON output of entities and relationships
3. **Coreference resolution** -- consolidation pass to merge duplicate entities ("Dr. Manette" / "Alexandre Manette" / "the old man" / "Lucie's father")
4. **Review and fix** -- manual spot-checking of the extracted graph

### Option B: Hybrid (NLP + Manual Curation)

Use spaCy NER for initial entity detection, then manually curate the graph with a predefined list of characters, locations, and events from known literary analysis of the novel.

> **Question for you:** Do you have a preference between Option A (LLM extraction, more automated but may need cleanup) and Option B (more manual but more precise)? Or a hybrid where we use a predefined entity list but LLM-extract the relationships and descriptions?

<!-- RESPONSE: -->

Preprocess the data for the demo using Claude, and store the preprocessed data in a format which can easily be imported into AstraeaDB.
---

## Embedding Strategy

Using **embeddinggemma** (already installed in Ollama):
- 768-dimensional embeddings, truncated to 128 dimensions via Matryoshka representation learning
- Embed node descriptions/summaries and TextChunk content
- These go directly into AstraeaDB's HNSW index (configured for cosine distance, dimension=128)

AstraeaDB stores embeddings on nodes natively, so we embed once during graph construction and the vector index is ready for hybrid search immediately.

> **Question:** AstraeaDB's vector index dimension is set at configuration time. Is 768 (embeddinggemma's output) the dimension you'd like to use, or do you have a different embedding model in mind?

<!-- RESPONSE: -->

Use the default 128 dimension embedding
---

## How the Demo Works (User-Facing)

### Query Flow

1. User types a question: *"What is Sydney Carton's relationship with Charles Darnay?"*
2. The system embeds the question using embeddinggemma (Matryoshka truncation to 128-dim)
3. gemma3:4b receives the question and has access to AstraeaDB MCP tools
4. The model calls MCP tools to gather context:
   - `find_by_label("Character")` to locate Carton and Darnay nodes
   - `neighbors(carton_id)` to get Carton's relationships
   - `shortest_path(carton_id, darnay_id)` to find connection paths
   - `hybrid_search(anchor=carton_id, query=question_embedding, ...)` for semantically relevant context
   - `extract_subgraph(carton_id, hops=2)` for surrounding narrative context
5. With structured graph context in hand, gemma3:4b generates a grounded answer

### Demo Scenarios to Showcase

| Question Type | Example | AstraeaDB Feature Demonstrated |
|--------------|---------|-------------------------------|
| **Character relationships** | "How are Charles Darnay and the Marquis related?" | Graph traversal, shortest_path |
| **Plot events** | "What happens at the Bastille?" | find_by_label, neighbors, extract_subgraph |
| **Thematic analysis** | "What are the major themes of the novel?" | Louvain communities, PageRank on Theme nodes |
| **Multi-hop reasoning** | "How does Dr. Manette's imprisonment connect to Darnay's trial?" | BFS, shortest_path, subgraph extraction |
| **Factual grounding** | "Quote a passage where Carton expresses his feelings for Lucie" | vector_search on TextChunks, MENTIONED_IN edges |
| **Comparison** | "Compare the fates of Carton and Darnay" | Parallel neighbor queries, hybrid_search |

### Comparison Mode (Optional but Powerful)

Side-by-side: ask the same question to raw gemma3:4b (no graph) vs. GraphRAG-augmented gemma3:4b. This vividly demonstrates the value of structured knowledge retrieval.

> **Question:** Would you like the demo to include this comparison mode? It's a compelling way to show the value proposition but adds some UI complexity.

<!-- RESPONSE: -->
Yes, if comparison mode doesn't overly complicate the demo, that would be great.

---

## Technical Decisions

### MCP Integration Approach

Two options for how gemma3:4b interacts with AstraeaDB:

**Implemented: Orchestrator with MCP and prompt-based tool calling**

A Python orchestrator mediates between the user, Ollama, and AstraeaDB's MCP server:

```
User -> Orchestrator -> Ollama (gemma3:4b) -> outputs TOOL_CALL JSON
                     -> MCP Bridge -> AstraeaDB MCP server (stdio) -> returns results
                     -> Ollama (gemma3:4b) -> final answer
                     -> User
```

The orchestrator:
- Includes all tool descriptions in the system prompt
- Instructs the model to output `TOOL_CALL: {"tool": "...", "args": {...}}` when it wants to call a tool
- Parses tool calls from the model's text response via regex
- Executes them against AstraeaDB via the MCP bridge (JSON-RPC 2.0 over stdio)
- Enriches results with node names and edge types for easier model navigation
- Feeds results back to the model for the next turn

**Note:** gemma3:4b does not support Ollama's native tool-calling API, so tool calling is implemented through prompt engineering. The MCP bridge handles the actual JSON-RPC communication with AstraeaDB's MCP server.

> **Question:** Do you want the demo to use the MCP server directly (showing MCP as the integration layer), or is the orchestrator pattern using AstraeaDB's Python client acceptable? The MCP approach is a cleaner demo of the MCP story but the orchestrator is more reliable with a small model.

<!-- RESPONSE: -->
I would prefer to use MCP.

---

## UI Options

For the interactive demo, options ranked by complexity:

1. **CLI (recommended for v1)** -- Simple terminal interface. Clean, no dependencies, easy to demo.
2. **Streamlit app** -- Richer UI with chat interface, graph visualization, and side-by-side comparison. More impressive visually.
3. **Jupyter notebook** -- Step-by-step walkthrough format. Good for educational purposes.

> **Question:** Which format do you prefer? Or should we build the CLI first and add a richer UI later?

<!-- RESPONSE: -->
Build the CLI first, we may add a better UI later.

---

## Component Summary

| Component | Language | Key Dependencies |
|-----------|----------|-----------------|
| Graph builder (extraction pipeline) | Python | Claude (extraction), AstraeaDB Python client (loading) |
| AstraeaDB server | Rust (pre-built) | astraea-cli binary (debug build with `mcp` subcommand) |
| Embedding service | Python | Ollama (embeddinggemma), Matryoshka truncation to 128-dim |
| MCP bridge | Python | AstraeaDB MCP server (JSON-RPC 2.0 over stdio) |
| Query orchestrator | Python | Ollama (gemma3:4b), MCP bridge, prompt-based tool calling |
| Demo interface | Python CLI | (no additional dependencies) |

---

## What Makes This Demo Compelling

1. **Small model, big results** -- gemma3:4b (3.3GB) answering complex multi-hop literary questions accurately
2. **Minimal hallucination** -- answers grounded in actual text passages linked through the graph
3. **Transparent reasoning** -- the graph traversal path shows *how* the answer was found
4. **AstraeaDB showcase** -- demonstrates hybrid search, graph algorithms, subgraph extraction, and MCP integration in a single cohesive use case
5. **Reproducible** -- all components run locally (Ollama + AstraeaDB), no cloud APIs needed

---

## Open Questions Summary

1. **Extraction approach** -- LLM-assisted (Option A) vs. hybrid NLP+manual (Option B)?
2. **Embedding dimension** -- 768 via embeddinggemma, or different model?
3. **Comparison mode** -- Include raw-vs-GraphRAG side-by-side?
4. **MCP vs. orchestrator** -- Direct MCP integration or Python orchestrator?
5. **UI format** -- CLI, Streamlit, Jupyter, or phased approach?

Please respond inline above or here, and I'll refine the design before moving to the implementation plan (Phase 2).
