"""MCP bridge: connects to AstraeaDB's MCP server via stdio and exposes tool calling.

Handles the JSON-RPC 2.0 protocol, initialization handshake, and tool execution.
"""

import json
import subprocess
import sys
from typing import Optional

from src import config

# Tools to expose to the LLM (curated subset of AstraeaDB's 29 tools)
EXPOSED_TOOLS = [
    {
        "name": "find_by_label",
        "description": "Find all nodes with a given label. Use this to find characters, locations, events, themes, chapters, or text passages. Labels: Character, Location, Event, Theme, Chapter, TextChunk.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "The label to search for (e.g., 'Character', 'Location', 'Event', 'Theme', 'Chapter', 'TextChunk')"}
            },
            "required": ["label"]
        }
    },
    {
        "name": "get_node",
        "description": "Get full details of a node by its numeric ID. Returns the node's labels, properties (including name, description, etc.), and embedding status.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The numeric node ID"}
            },
            "required": ["id"]
        }
    },
    {
        "name": "neighbors",
        "description": "Get all neighbors of a node, optionally filtered by direction and edge type. Shows relationships like RELATED_TO, LOCATED_IN, PARTICIPATES_IN, EMBODIES, MENTIONED_IN, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The node ID to query"},
                "direction": {"type": "string", "enum": ["outgoing", "incoming", "both"], "description": "Edge direction (default: outgoing)"},
                "edge_type": {"type": "string", "description": "Optional: filter by edge type (e.g., 'RELATED_TO', 'PARTICIPATES_IN')"}
            },
            "required": ["id"]
        }
    },
    {
        "name": "shortest_path",
        "description": "Find the shortest path between two nodes. Returns the sequence of node IDs connecting them.",
        "parameters": {
            "type": "object",
            "properties": {
                "from": {"type": "integer", "description": "Start node ID"},
                "to": {"type": "integer", "description": "End node ID"},
                "weighted": {"type": "boolean", "description": "Use weighted edges (default: false)"}
            },
            "required": ["from", "to"]
        }
    },
    {
        "name": "vector_search",
        "description": "Search for nodes semantically similar to a query embedding vector. Returns the k most similar nodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "array", "items": {"type": "number"}, "description": "128-dimensional query embedding vector"},
                "k": {"type": "integer", "description": "Number of results to return (default: 10)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "hybrid_search",
        "description": "Combined vector similarity + graph proximity search. Starts from an anchor node and finds results that are both semantically similar and graph-close.",
        "parameters": {
            "type": "object",
            "properties": {
                "anchor": {"type": "integer", "description": "Anchor node ID"},
                "query": {"type": "array", "items": {"type": "number"}, "description": "128-dimensional query embedding vector"},
                "max_hops": {"type": "integer", "description": "Maximum graph distance (default: 3)"},
                "k": {"type": "integer", "description": "Number of results (default: 10)"},
                "alpha": {"type": "number", "description": "Blend factor: 0.0=pure graph, 1.0=pure vector (default: 0.5)"}
            },
            "required": ["anchor", "query"]
        }
    },
    {
        "name": "extract_subgraph",
        "description": "Extract a local neighborhood around a node and return it as readable text. Great for getting context about an entity and its surroundings.",
        "parameters": {
            "type": "object",
            "properties": {
                "center": {"type": "integer", "description": "Center node ID"},
                "hops": {"type": "integer", "description": "How many hops to traverse (default: 2)"},
                "max_nodes": {"type": "integer", "description": "Maximum nodes to include (default: 50)"},
                "format": {"type": "string", "enum": ["prose", "structured", "triples", "json"], "description": "Output format (default: structured)"}
            },
            "required": ["center"]
        }
    },
    {
        "name": "graph_stats",
        "description": "Get statistics about the knowledge graph: node count, edge count, label distribution.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "bfs",
        "description": "Breadth-first search from a starting node. Returns all reachable nodes up to a given depth.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "Starting node ID"},
                "max_depth": {"type": "integer", "description": "Maximum search depth (default: 3)"}
            },
            "required": ["start"]
        }
    },
    {
        "name": "query",
        "description": "Execute a GQL (Graph Query Language) query. Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT. Example: MATCH (n:Character) WHERE n.name = 'Sydney Carton' RETURN n",
        "parameters": {
            "type": "object",
            "properties": {
                "gql": {"type": "string", "description": "The GQL query string"}
            },
            "required": ["gql"]
        }
    },
]


def tools_for_ollama():
    """Return tool definitions in Ollama's tool-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
        }
        for tool in EXPOSED_TOOLS
    ]


class McpBridge:
    """Manages a stdio connection to AstraeaDB's MCP server."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._request_id = 0

    def start(self):
        """Launch the AstraeaDB MCP server subprocess."""
        cmd = [config.ASTRAEA_BIN, "mcp"]
        if config.ASTRAEA_DATA_DIR:
            cmd.extend(["--embedded", "--data-dir", config.ASTRAEA_DATA_DIR])
        else:
            cmd.extend(["--address", f"{config.ASTRAEA_HOST}:{config.ASTRAEA_PORT}"])

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._initialize()

    def stop(self):
        """Stop the MCP server subprocess."""
        if self._proc:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def _next_id(self):
        self._request_id += 1
        return self._request_id

    def _send(self, method, params=None, is_notification=False):
        """Send a JSON-RPC 2.0 request and return the response."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if not is_notification:
            msg["id"] = self._next_id()
        if params is not None:
            msg["params"] = params

        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

        if is_notification:
            return None

        # Read response
        response_line = self._proc.stdout.readline()
        if not response_line:
            raise ConnectionError("MCP server closed connection")
        return json.loads(response_line)

    def _initialize(self):
        """Perform MCP initialization handshake."""
        resp = self._send("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "graphrag-demo", "version": "1.0"}
        })
        if resp.get("error"):
            raise RuntimeError(f"MCP init failed: {resp['error']}")
        # Send initialized notification
        self._send("notifications/initialized", {}, is_notification=True)

    def call_tool(self, tool_name, arguments):
        """Call an MCP tool and return the result content."""
        resp = self._send("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if resp.get("error"):
            return {"error": resp["error"].get("message", str(resp["error"]))}
        result = resp.get("result", {})
        # MCP tool results have a "content" array with type/text entries
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            try:
                parsed = json.loads(content[0]["text"])
            except (json.JSONDecodeError, KeyError):
                return {"text": content[0].get("text", "")}
        else:
            parsed = result

        # Enrich find_by_label results with node names so the model can navigate
        if tool_name == "find_by_label" and "node_ids" in parsed:
            enriched = []
            for nid in parsed["node_ids"]:
                node = self.call_tool("get_node", {"id": nid})
                props = node.get("properties", {})
                name = props.get("name", props.get("title", props.get("context", "")))
                enriched.append({"id": nid, "name": name})
            return {"nodes": enriched}

        # Enrich neighbors results with edge types and target node names
        if tool_name == "neighbors" and "neighbors" in parsed:
            enriched = []
            for nbr in parsed["neighbors"]:
                nid = nbr["node_id"]
                eid = nbr["edge_id"]
                edge = self.call_tool("get_edge", {"id": eid})
                node = self.call_tool("get_node", {"id": nid})
                props = node.get("properties", {})
                name = props.get("name", props.get("title", props.get("context", "")))
                label = node.get("labels", [""])[0]
                edge_type = edge.get("edge_type", "")
                edge_props = edge.get("properties", {})
                enriched.append({
                    "node_id": nid,
                    "node_name": name,
                    "node_label": label,
                    "edge_type": edge_type,
                    "edge_properties": edge_props,
                })
            return {"neighbors": enriched}

        # Enrich vector_search results with node names and labels
        if tool_name == "vector_search" and "results" in parsed:
            enriched = []
            for r in parsed["results"]:
                nid = r["node_id"]
                node = self.call_tool("get_node", {"id": nid})
                props = node.get("properties", {})
                name = props.get("name", props.get("title", props.get("context", "")))
                label = node.get("labels", [""])[0]
                desc = props.get("description", props.get("text", props.get("summary", "")))
                enriched.append({
                    "node_id": nid,
                    "name": name,
                    "label": label,
                    "distance": r.get("distance"),
                    "description": desc[:300] if desc else "",
                })
            return {"results": enriched}

        return parsed

    def ping(self):
        """Health check."""
        return self.call_tool("ping", {})

    def list_tools(self):
        """List available MCP tools."""
        resp = self._send("tools/list", {})
        return resp.get("result", {}).get("tools", [])
