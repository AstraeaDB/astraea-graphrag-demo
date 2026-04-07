"""Query orchestrator: manages the conversation loop between user, Ollama, and AstraeaDB.

Since gemma3:4b doesn't support Ollama's native tool-calling API, we implement
tool calling through prompt engineering: the model outputs JSON tool calls in a
structured format, and we parse and execute them.
"""

import json
import re
import requests

from src import config
from src.embeddings import embed
from src.mcp_bridge import McpBridge, EXPOSED_TOOLS

MAX_TOOL_ROUNDS = 5


def _build_tool_descriptions():
    """Build a text description of available tools for the system prompt."""
    lines = []
    for tool in EXPOSED_TOOLS:
        params = tool["parameters"].get("properties", {})
        required = tool["parameters"].get("required", [])
        param_strs = []
        for pname, pdef in params.items():
            req = " (required)" if pname in required else ""
            param_strs.append(f"    - {pname}: {pdef.get('description', pdef.get('type', ''))}{req}")
        params_block = "\n".join(param_strs) if param_strs else "    (none)"
        lines.append(f"  {tool['name']}: {tool['description']}\n  Parameters:\n{params_block}")
    return "\n\n".join(lines)


TOOL_DESCRIPTIONS = _build_tool_descriptions()

SYSTEM_PROMPT = f"""You are a literary analyst with access to a knowledge graph of Charles Dickens's "A Tale of Two Cities" stored in AstraeaDB.

You have access to these tools to query the graph:

{TOOL_DESCRIPTIONS}

HOW TO USE TOOLS:
- To call a tool, output EXACTLY this format on its own line:
  TOOL_CALL: {{"tool": "tool_name", "args": {{...}}}}
- You may call ONE tool per response.
- After you receive the tool result, you can call another tool or provide your final answer.
- When you have enough information, provide your answer as plain text (no TOOL_CALL).

IMPORTANT RULES:
1. ALWAYS use tools to look up information before answering. Do not rely on your own knowledge.
2. To find characters/locations/events, start with find_by_label, then use get_node for details.
3. For relationships between entities, use neighbors or shortest_path.
4. For quotes or passages, use vector_search (the embedding will be provided for you).
5. Base your answers ONLY on what the tools return. Cite specific details from the graph.
6. If a tool returns an error, try a different approach.
7. Be concise and factual."""


def _parse_tool_call(text):
    """Extract a TOOL_CALL JSON from the model's response text."""
    # Look for TOOL_CALL: {...}
    match = re.search(r'TOOL_CALL:\s*(\{.*\})', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return parsed.get("tool"), parsed.get("args", {})
        except json.JSONDecodeError:
            pass

    # Fallback: look for any JSON object with "tool" key (handles nested braces)
    for match in re.finditer(r'\{[^{}]*"tool"\s*:', text):
        start = match.start()
        # Find the matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i+1])
                        if "tool" in parsed:
                            return parsed["tool"], parsed.get("args", {})
                    except json.JSONDecodeError:
                        pass
                    break

    return None, None


class Orchestrator:
    """Manages the query loop between user, Ollama (gemma3:4b), and AstraeaDB."""

    def __init__(self, mcp: McpBridge, verbose=False):
        self.mcp = mcp
        self.verbose = verbose
        self.tool_calls_log = []

    def query(self, question, question_embedding=None):
        """Run a full query cycle and return the answer text."""
        self.tool_calls_log = []

        if question_embedding is None:
            question_embedding = embed(question)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for round_num in range(MAX_TOOL_ROUNDS):
            response = self._chat(messages)
            if response is None:
                return "Error: No response from the model."

            content = response.get("message", {}).get("content", "")

            # Check if the model wants to call a tool
            tool_name, tool_args = _parse_tool_call(content)

            if tool_name is None:
                # No tool call -- this is the final answer
                # Strip any preamble before the actual answer
                return content.strip()

            # Auto-inject question embedding for vector/hybrid search
            if tool_name in ("vector_search", "hybrid_search") and "query" not in tool_args:
                tool_args["query"] = question_embedding

            if self.verbose:
                args_preview = json.dumps(tool_args, default=str)
                if len(args_preview) > 200:
                    args_preview = args_preview[:200] + "..."
                print(f"  [Tool] {tool_name}({args_preview})")

            result = self.mcp.call_tool(tool_name, tool_args)
            result_str = json.dumps(result, default=str)

            # Truncate very large results to fit in context
            if len(result_str) > 4000:
                result_str = result_str[:4000] + "... (truncated)"

            self.tool_calls_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result_preview": result_str[:500],
            })

            if self.verbose:
                print(f"  [Result] {result_str[:300]}")

            # Add the assistant's tool call and the result to the conversation
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"TOOL_RESULT for {tool_name}:\n{result_str}\n\nUse another tool if needed, or provide your final answer."})

        # Exhausted rounds -- ask for final answer
        messages.append({"role": "user", "content": "Please provide your final answer based on everything you've learned."})
        response = self._chat(messages)
        if response:
            return response.get("message", {}).get("content", "").strip()
        return "Error: Could not generate answer."

    def query_raw(self, question):
        """Query the model WITHOUT tools (for comparison mode)."""
        messages = [
            {"role": "system", "content": "You are a literary analyst. Answer questions about Charles Dickens's 'A Tale of Two Cities' based on your knowledge. Be concise."},
            {"role": "user", "content": question},
        ]
        response = self._chat(messages)
        if response:
            return response.get("message", {}).get("content", "").strip()
        return "Error: No response from the model."

    def _chat(self, messages):
        """Send a chat request to Ollama."""
        payload = {
            "model": config.CHAT_MODEL,
            "messages": messages,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  [Error] Ollama request failed: {e}")
            return None
