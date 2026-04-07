# Implementation Plan: GraphRAG Demo

## Overview

Build a CLI demo where gemma3:4b (via Ollama) answers questions about *A Tale of Two Cities* using AstraeaDB's knowledge graph through MCP, with an optional comparison mode showing raw LLM vs. GraphRAG-augmented answers.

---

## Phase 3 Steps

### Step 1: Preprocess the Novel with Claude

**Goal:** Extract a complete knowledge graph from the novel text and save it as JSON files ready for import.

**Output files:**
- `data/characters.json` -- Character nodes with names, aliases, descriptions
- `data/locations.json` -- Location nodes
- `data/events.json` -- Key plot events anchored to chapters
- `data/themes.json` -- Major themes
- `data/chapters.json` -- Chapter summaries
- `data/text_chunks.json` -- Key passages (~300-500 paragraphs/scenes)
- `data/edges.json` -- All relationships between entities

**Process:** Claude reads the novel text and produces structured JSON following the schema from the design proposal. Coreference resolution happens during extraction (e.g., all references to Dr. Manette consolidated into one entity). Each entity gets a `description` field that will be embedded later.

**Schema per file:**

```json
// characters.json
[
  {
    "id": "char_sydney_carton",
    "name": "Sydney Carton",
    "aliases": ["Carton", "the jackal"],
    "description": "A dissolute English barrister who...",
    "first_appearance": {"book": 2, "chapter": 3}
  }
]

// edges.json
[
  {
    "source": "char_sydney_carton",
    "target": "char_charles_darnay",
    "edge_type": "RESEMBLES",
    "properties": {"description": "Physical resemblance exploited at trial..."}
  }
]
```

String IDs (e.g., `char_sydney_carton`) are used in the JSON for readability; the loader script maps them to AstraeaDB's integer node IDs during import.

---

### Step 2: Generate 128-Dimensional Embeddings

**Goal:** Produce embeddings for all node descriptions and text chunks.

**Approach:** Use Ollama's embedding API with embeddinggemma. EmbeddingGemma supports **Matryoshka representation learning**, meaning its embedding dimensions are trained to be useful at any prefix length. We simply truncate the 768-dim output to the first 128 dimensions and L2 re-normalize. No PCA or dimensionality reduction model needed. This matches AstraeaDB's hardcoded HNSW index dimension of 128.

This approach is already proven in `/Users/jamesharris/Documents/astraea-agent-graph/embeddings.py`.

**Script:** `scripts/generate_embeddings.py`

```
Input:  data/*.json (descriptions/text fields)
Output: data/embeddings.json (maps entity ID -> 128-dim float vector)
```

**Dependencies:** `requests` (for Ollama API)

**Embedding function (from astraea-agent-graph pattern):**
```python
def embed(texts: List[str], dim: int = 128) -> List[List[float]]:
    resp = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "embeddinggemma", "input": texts},
    )
    embeddings = resp.json()["embeddings"]
    # Matryoshka truncation + L2 re-normalize
    return [normalize(emb[:dim]) for emb in embeddings]
```

---

### Step 3: Build the Graph Loader

**Goal:** Python script that starts AstraeaDB, creates all nodes (with embeddings) and edges.

**Script:** `scripts/load_graph.py`

**Process:**
1. Connect to AstraeaDB via Python client
2. Create all nodes from `data/*.json`, attaching embeddings from `data/embeddings.json`
3. Create all edges from `data/edges.json`
4. Verify with `graph_stats` call
5. Print summary (node count, edge count, label distribution)

**ID mapping:** Maintains a dict `{string_id: astraeadb_node_id}` to resolve edge source/target references.

---

### Step 4: MCP Bridge for Ollama

**Goal:** Python module that bridges the orchestrator's tool calls to AstraeaDB's MCP server via JSON-RPC over stdio.

**Module:** `src/mcp_bridge.py`

**How it works:**

```
User question
     │
     ▼
┌──────────────┐     tool descriptions      ┌──────────────┐
│   Ollama     │◄───────────────────────────►│  MCP Bridge  │
│  gemma3:4b   │     tool calls / results    │  (Python)    │
│  (tool use)  │                             └──────┬───────┘
└──────────────┘                                    │
                                              JSON-RPC/stdio
                                                    │
                                             ┌──────▼───────┐
                                             │  AstraeaDB   │
                                             │  MCP Server   │
                                             └──────────────┘
```

The bridge:
1. Launches `astraea-cli mcp` as a subprocess (stdio transport)
2. Performs MCP initialization handshake (initialize, notifications)
3. Exposes 10 curated tools to the orchestrator
4. When a tool call is received, sends the corresponding `tools/call` JSON-RPC request to the MCP server
5. **Enriches results:** `find_by_label` results are enriched with node names, `neighbors` results with edge types and target node names, and `vector_search` results with names, labels, and descriptions -- so the model can navigate the graph without extra round-trips
6. Returns enriched tool results to the orchestrator

**Key MCP tools to expose to the model:**
- `find_by_label` -- Find characters, locations, events by type
- `get_node` -- Get full node details
- `neighbors` -- Explore relationships
- `shortest_path` -- Find connections between entities
- `hybrid_search` -- Semantic + graph search
- `extract_subgraph` -- Get linearized neighborhood context
- `vector_search` -- Find semantically similar nodes
- `graph_stats` -- Overview of the graph

We'll curate which tools to expose (not all 29) to keep the tool descriptions within gemma3:4b's context budget.

---

### Step 5: Query Orchestrator

**Goal:** Main query loop that manages the conversation between user, Ollama, and AstraeaDB.

**Module:** `src/orchestrator.py`

**Flow per query:**
1. Accept user question
2. Generate question embedding (Ollama embeddinggemma -> Matryoshka truncation to 128-dim)
3. Build system prompt with:
   - Role description ("You are a literary analyst with access to a knowledge graph of A Tale of Two Cities...")
   - All 10 tool descriptions with parameter schemas
   - Instructions to output `TOOL_CALL: {"tool": "...", "args": {...}}` when calling a tool
   - Instructions to use tools before answering, cite sources
4. Send to gemma3:4b via Ollama chat API (no native tool-use -- prompt-based approach)
5. **Tool-call loop:** Parse the model's text output for `TOOL_CALL:` patterns:
   - If found: execute the tool call via MCP bridge, append result to conversation, send back for next turn
   - If not found: treat the response as the final answer
6. When the model returns a text response (no TOOL_CALL), present to user

**Note:** gemma3:4b does not support Ollama's native tool-calling API (`"tools"` parameter returns a 400 error). The orchestrator implements tool calling through prompt engineering: tool descriptions are included in the system prompt, and the model is instructed to output structured JSON when it wants to call a tool. The orchestrator parses these with regex, executes them via MCP, and feeds results back as follow-up user messages.

**Max tool-call rounds:** 5 (prevents infinite loops with a small model)

---

### Step 6: Comparison Mode

**Goal:** Show raw LLM vs. GraphRAG side-by-side.

**Implementation:** When comparison mode is enabled (`--compare` flag):
1. First, send the question to gemma3:4b **without** tool access (plain chat)
2. Then, run the full GraphRAG pipeline (Step 5)
3. Display both answers labeled "Without Graph" and "With GraphRAG"

This is lightweight -- just an extra Ollama call without tools.

---

### Step 7: CLI Interface

**Goal:** Interactive terminal application.

**Script:** `main.py`

**Features:**
- Interactive REPL with prompt
- Commands:
  - Just type a question to query
  - `/compare` -- Toggle comparison mode on/off
  - `/stats` -- Show graph statistics
  - `/tools` -- Show available MCP tools
  - `/quit` -- Exit
- Colored output (graph-augmented answer vs. raw answer in comparison mode)
- Shows which MCP tools were called and their results (at verbose level)

**CLI arguments:**
- `--compare` -- Start with comparison mode on
- `--verbose` -- Show tool calls and intermediate results

Other settings (AstraeaDB host/port, binary path, Ollama URL, model names) are configured via environment variables in `src/config.py`.

---

## File Structure

```
graphrag-demo/
├── CLAUDE.md
├── design-proposal.md
├── implementation-plan.md
├── tale_of_two_cities.txt
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── data/
│   ├── characters.json              # Extracted character entities
│   ├── locations.json               # Extracted location entities
│   ├── events.json                  # Extracted plot events
│   ├── themes.json                  # Extracted themes
│   ├── chapters.json                # Chapter summaries
│   ├── text_chunks.json             # Key text passages
│   ├── edges.json                   # All relationships
│   └── embeddings.json              # 128-dim embeddings for all entities
├── scripts/
│   ├── generate_embeddings.py       # Embed descriptions via Ollama + Matryoshka truncation
│   ├── load_graph.py                # Load JSON data into AstraeaDB
│   └── merge_extractions.py         # Merge per-book extractions into unified data/
├── src/
│   ├── __init__.py
│   ├── mcp_bridge.py                # MCP client -> AstraeaDB MCP server + result enrichment
│   ├── orchestrator.py              # Prompt-based tool-calling loop: user <-> Ollama <-> MCP
│   ├── embeddings.py                # EmbeddingGemma via Ollama (128-dim Matryoshka)
│   └── config.py                    # Paths, model names, defaults
└── test_demo.py                     # Test suite (24 tests across 4 layers)
```

---

## Dependencies

```
# requirements.txt
requests>=2.31
```

Minimal dependencies -- no numpy or scikit-learn needed since Matryoshka truncation is just a slice + normalize. The AstraeaDB Python client is imported from the local path (`/Users/jamesharris/Documents/astraeadb/python/`).

---

## Build & Run Sequence

```bash
# 1. Ensure Ollama is running with required models
ollama pull gemma3:4b
ollama pull embeddinggemma

# 2. Preprocess with Claude (done in conversation -- outputs data/*.json)

# 3. Generate embeddings
python scripts/generate_embeddings.py

# 4. Start AstraeaDB and load the graph
/Users/jamesharris/Documents/astraeadb/target/debug/astraea-cli serve &
python scripts/load_graph.py

# 5. Run the demo
python main.py
# Or with comparison mode:
python main.py --compare
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| gemma3:4b produces malformed tool calls | Robust JSON parsing with fallback; retry with simplified prompt; max 5 tool rounds |
| Matryoshka 128-dim truncation loses semantic signal | EmbeddingGemma is trained for this; proven in astraea-agent-graph; test retrieval quality during loading |
| MCP stdio subprocess management | Clean startup/shutdown; health check via `ping` tool; automatic restart on crash |
| Entity extraction misses relationships | Manual review of Claude's output before loading; iterative refinement |
| Small model ignores tool results | Strong system prompt emphasizing "answer ONLY based on tool results"; comparison mode makes this visible |

---

## Success Criteria

The demo is successful when:
1. gemma3:4b correctly answers multi-hop questions (e.g., "How does Dr. Manette's imprisonment relate to Darnay's trial?") by traversing the graph
2. Answers cite or reference specific text passages from the novel
3. Comparison mode shows clear improvement over raw LLM responses
4. The full pipeline runs locally with no cloud dependencies
5. Graph operations (traversal, hybrid search, subgraph extraction) are visible in verbose mode, showing AstraeaDB at work
