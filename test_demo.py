#!/usr/bin/env python3
"""End-to-end test suite for the GraphRAG demo.

Tests each layer independently, then tests the full pipeline.
Run with: python3 test_demo.py
"""

import json
import sys
import os
import traceback

# Track results
passed = 0
failed = 0
errors = []


def test(name):
    """Decorator to register and run a test."""
    def decorator(fn):
        global passed, failed
        print(f"  {name}...", end=" ", flush=True)
        try:
            fn()
            print("PASS")
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            errors.append((name, traceback.format_exc()))
            failed += 1
    return decorator


# ============================================================
# Layer 1: Data integrity
# ============================================================
print("\n=== Layer 1: Data Files ===")


@test("All data files exist")
def _():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    for f in ['characters.json', 'locations.json', 'events.json', 'themes.json',
              'chapters.json', 'text_chunks.json', 'edges.json', 'embeddings.json']:
        assert os.path.exists(os.path.join(data_dir, f)), f"Missing {f}"


@test("Entity counts are correct")
def _():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    counts = {}
    for name in ['characters', 'locations', 'events', 'themes', 'chapters', 'text_chunks']:
        with open(os.path.join(data_dir, f'{name}.json')) as f:
            counts[name] = len(json.load(f))
    assert counts['characters'] == 29, f"Expected 29 characters, got {counts['characters']}"
    assert counts['themes'] == 6, f"Expected 6 themes, got {counts['themes']}"
    assert counts['chapters'] == 45, f"Expected 45 chapters, got {counts['chapters']}"
    total = sum(counts.values())
    assert total == 229, f"Expected 229 total nodes, got {total}"


@test("Edges reference valid entity IDs")
def _():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    all_ids = set()
    for name in ['characters', 'locations', 'events', 'themes', 'chapters', 'text_chunks']:
        with open(os.path.join(data_dir, f'{name}.json')) as f:
            for item in json.load(f):
                all_ids.add(item['id'])
    with open(os.path.join(data_dir, 'edges.json')) as f:
        edges = json.load(f)
    assert len(edges) == 317, f"Expected 317 edges, got {len(edges)}"
    for i, e in enumerate(edges):
        assert e['source'] in all_ids, f"Edge {i}: dangling source '{e['source']}'"
        assert e['target'] in all_ids, f"Edge {i}: dangling target '{e['target']}'"


@test("Embeddings exist for all entities and are 128-dim")
def _():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    with open(os.path.join(data_dir, 'embeddings.json')) as f:
        embeddings = json.load(f)
    assert len(embeddings) == 229, f"Expected 229 embeddings, got {len(embeddings)}"
    for eid, vec in embeddings.items():
        assert len(vec) == 128, f"Embedding {eid} has {len(vec)} dims, expected 128"


# ============================================================
# Layer 2: Ollama connectivity
# ============================================================
print("\n=== Layer 2: Ollama ===")

import requests
from src import config


@test("Ollama is reachable")
def _():
    resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
    resp.raise_for_status()


@test("gemma3:4b model is available")
def _():
    resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
    models = [m['name'] for m in resp.json()['models']]
    assert any('gemma3' in m for m in models), f"gemma3:4b not found in {models}"


@test("embeddinggemma model is available")
def _():
    resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
    models = [m['name'] for m in resp.json()['models']]
    assert any('embeddinggemma' in m for m in models), f"embeddinggemma not found in {models}"


@test("Embedding generation works (128-dim)")
def _():
    from src.embeddings import embed
    vec = embed("test sentence")
    assert len(vec) == 128, f"Expected 128 dims, got {len(vec)}"
    # Check it's normalized (L2 norm ~= 1.0)
    import math
    norm = math.sqrt(sum(x*x for x in vec))
    assert abs(norm - 1.0) < 0.01, f"Not normalized: L2 norm = {norm}"


@test("Chat completion works")
def _():
    resp = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": config.CHAT_MODEL,
        "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
        "stream": False,
    }, timeout=60)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    assert len(content) > 0, "Empty response from model"


# ============================================================
# Layer 3: AstraeaDB + MCP bridge
# ============================================================
print("\n=== Layer 3: MCP Bridge ===")

from src.mcp_bridge import McpBridge

mcp = McpBridge()
mcp.start()


@test("MCP ping succeeds")
def _():
    result = mcp.ping()
    assert result.get("pong") is True, f"Ping failed: {result}"


@test("Graph data is loaded (not empty)")
def _():
    result = mcp.call_tool("graph_stats", {})
    labels = result.get("labels", {})
    char_count = labels.get("Character", 0)
    assert char_count > 0, (
        f"No Character nodes found -- the graph is empty. "
        f"Run 'python3 scripts/load_graph.py' to load data into AstraeaDB. "
        f"(AstraeaDB stores data in memory; it must be reloaded after each server restart.)"
    )


@test("find_by_label returns enriched Character nodes")
def _():
    result = mcp.call_tool("find_by_label", {"label": "Character"})
    assert "nodes" in result, f"Expected 'nodes' key, got: {list(result.keys())}"
    assert len(result["nodes"]) == 29, f"Expected 29 characters, got {len(result['nodes'])}"
    # Check enrichment: each node should have id and name
    first = result["nodes"][0]
    assert "id" in first, "Missing 'id' in enriched node"
    assert "name" in first, "Missing 'name' in enriched node"
    assert isinstance(first["name"], str) and len(first["name"]) > 0, "Empty name"


@test("find_by_label returns all label types")
def _():
    for label, expected in [("Location", 28), ("Event", 44), ("Theme", 6), ("Chapter", 45), ("TextChunk", 77)]:
        result = mcp.call_tool("find_by_label", {"label": label})
        count = len(result.get("nodes", []))
        assert count == expected, f"{label}: expected {expected}, got {count}"


@test("get_node returns properties")
def _():
    # Get a character node
    chars = mcp.call_tool("find_by_label", {"label": "Character"})
    nid = chars["nodes"][0]["id"]
    node = mcp.call_tool("get_node", {"id": nid})
    assert "labels" in node, "Missing 'labels'"
    assert "properties" in node, "Missing 'properties'"
    assert "Character" in node["labels"], f"Expected Character label, got {node['labels']}"
    assert "name" in node["properties"], "Missing 'name' property"


@test("neighbors returns enriched results with edge types")
def _():
    # Find a character with known relationships
    chars = mcp.call_tool("find_by_label", {"label": "Character"})
    # Try each character until we find one with neighbors
    found = False
    for c in chars["nodes"]:
        result = mcp.call_tool("neighbors", {"id": c["id"], "direction": "both"})
        if result.get("neighbors"):
            nbr = result["neighbors"][0]
            assert "node_name" in nbr, "Missing 'node_name' in enriched neighbor"
            assert "edge_type" in nbr, "Missing 'edge_type' in enriched neighbor"
            assert "node_label" in nbr, "Missing 'node_label' in enriched neighbor"
            found = True
            break
    assert found, "No character had any neighbors"


@test("shortest_path finds connected characters")
def _():
    with open(os.path.join('data', 'id_map.json')) as f:
        id_map = json.load(f)
    # Manette and Lucie should be directly connected
    manette = id_map["char_doctor_manette"]
    lucie = id_map["char_lucie_manette"]
    result = mcp.call_tool("shortest_path", {"from": manette, "to": lucie})
    assert result.get("path") is not None, f"No path found between Manette and Lucie: {result}"
    assert manette in result["path"], "Manette not in path"
    assert lucie in result["path"], "Lucie not in path"


@test("vector_search returns enriched results")
def _():
    from src.embeddings import embed
    vec = embed("sacrifice and redemption")
    result = mcp.call_tool("vector_search", {"query": vec, "k": 5})
    assert "results" in result, f"Expected 'results' key, got: {list(result.keys())}"
    assert len(result["results"]) == 5, f"Expected 5 results, got {len(result['results'])}"
    first = result["results"][0]
    assert "name" in first, "Missing 'name' in enriched result"
    assert "label" in first, "Missing 'label' in enriched result"
    assert "distance" in first, "Missing 'distance'"


@test("extract_subgraph returns text")
def _():
    with open(os.path.join('data', 'id_map.json')) as f:
        id_map = json.load(f)
    carton = id_map["char_sydney_carton"]
    result = mcp.call_tool("extract_subgraph", {"center": carton, "hops": 1, "max_nodes": 10})
    assert "text" in result, f"Expected 'text' key, got: {list(result.keys())}"
    assert "Carton" in result["text"], "Subgraph text doesn't mention Carton"


@test("GQL query works")
def _():
    result = mcp.call_tool("query", {"gql": "MATCH (n:Theme) RETURN n.name"})
    assert "rows" in result, f"Expected 'rows' key, got: {list(result.keys())}"
    assert len(result["rows"]) == 6, f"Expected 6 theme rows, got {len(result['rows'])}"


@test("graph_stats returns correct structure")
def _():
    result = mcp.call_tool("graph_stats", {})
    assert "total_nodes" in result, "Missing total_nodes"
    assert "total_edges" in result, "Missing total_edges"
    assert "labels" in result, "Missing labels"
    assert result["labels"].get("Character") == 29, f"Expected 29 Characters"


# ============================================================
# Layer 4: Orchestrator
# ============================================================
print("\n=== Layer 4: Orchestrator ===")

from src.orchestrator import Orchestrator, _parse_tool_call

@test("Tool call parser: valid TOOL_CALL format")
def _():
    text = 'Let me search for that.\nTOOL_CALL: {"tool": "find_by_label", "args": {"label": "Character"}}'
    tool, args = _parse_tool_call(text)
    assert tool == "find_by_label", f"Expected find_by_label, got {tool}"
    assert args == {"label": "Character"}, f"Wrong args: {args}"


@test("Tool call parser: no tool call returns None")
def _():
    text = "Sydney Carton is a character who sacrifices his life."
    tool, args = _parse_tool_call(text)
    assert tool is None, f"Should be None, got {tool}"


@test("Tool call parser: JSON block fallback")
def _():
    text = 'I will look that up. {"tool": "get_node", "args": {"id": 42}}'
    tool, args = _parse_tool_call(text)
    assert tool == "get_node", f"Expected get_node, got {tool}"
    assert args == {"id": 42}, f"Wrong args: {args}"


@test("Raw query returns non-empty answer")
def _():
    orch = Orchestrator(mcp, verbose=False)
    answer = orch.query_raw("Who wrote A Tale of Two Cities?")
    assert len(answer) > 10, f"Answer too short: '{answer}'"
    assert "dickens" in answer.lower(), f"Answer doesn't mention Dickens: '{answer[:200]}'"


@test("GraphRAG query uses tools and returns answer")
def _():
    orch = Orchestrator(mcp, verbose=False)
    answer = orch.query("What are the themes of the novel?")
    assert len(answer) > 50, f"Answer too short: '{answer[:200]}'"
    assert len(orch.tool_calls_log) > 0, "No tool calls were made"
    # Should have called find_by_label for themes
    tool_names = [tc["tool"] for tc in orch.tool_calls_log]
    assert any("find_by_label" in t or "query" in t for t in tool_names), \
        f"Expected find_by_label or query tool call, got: {tool_names}"


mcp.stop()


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed} failed")
if errors:
    print(f"\nFailed tests:")
    for name, tb in errors:
        print(f"\n  {name}:")
        for line in tb.strip().split('\n')[-3:]:
            print(f"    {line}")
print(f"{'=' * 60}")

sys.exit(0 if failed == 0 else 1)
