#!/usr/bin/env python3
"""GraphRAG Demo: A Tale of Two Cities + AstraeaDB + Ollama.

Interactive CLI demonstrating how a modest LLM (gemma3:4b) can answer
complex literary questions using a knowledge graph via MCP.

Usage:
    python main.py                  # Standard mode
    python main.py --compare        # Start with comparison mode on
    python main.py --verbose        # Show tool calls
"""

import argparse
import json
import sys

from src import config
from src.mcp_bridge import McpBridge
from src.orchestrator import Orchestrator

# ANSI color codes
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def print_header():
    print(f"""
{BOLD}{'=' * 60}
  GraphRAG Demo: A Tale of Two Cities
  AstraeaDB + Ollama ({config.CHAT_MODEL}) + MCP
{'=' * 60}{RESET}

Type a question about the novel, or use a command:
  {CYAN}/compare{RESET}  - Toggle comparison mode (raw LLM vs GraphRAG)
  {CYAN}/stats{RESET}    - Show knowledge graph statistics
  {CYAN}/tools{RESET}    - Show available MCP tools
  {CYAN}/verbose{RESET}  - Toggle verbose mode (show tool calls)
  {CYAN}/quit{RESET}     - Exit
""")


def print_answer(label, text, color=GREEN):
    print(f"\n{color}{BOLD}[{label}]{RESET}")
    print(text)


def print_tool_log(log):
    if not log:
        return
    print(f"\n{DIM}--- Tool calls ---{RESET}")
    for entry in log:
        print(f"  {DIM}{entry['tool']}() -> {entry['result_preview'][:100]}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Demo CLI")
    parser.add_argument("--compare", action="store_true", help="Start with comparison mode enabled")
    parser.add_argument("--verbose", action="store_true", help="Show tool calls and results")
    args = parser.parse_args()

    compare_mode = args.compare
    verbose_mode = args.verbose

    print_header()

    print(f"{DIM}Connecting to AstraeaDB MCP server...{RESET}")
    try:
        mcp = McpBridge()
        mcp.start()
    except Exception as e:
        print(f"{RED}Failed to start MCP bridge: {e}{RESET}")
        print(f"\nMake sure AstraeaDB is running:")
        print(f"  {config.ASTRAEA_BIN} serve")
        sys.exit(1)

    try:
        ping_result = mcp.ping()
        print(f"{GREEN}Connected to AstraeaDB.{RESET}")
    except Exception as e:
        print(f"{RED}MCP ping failed: {e}{RESET}")
        print(f"Make sure AstraeaDB server is running on {config.ASTRAEA_HOST}:{config.ASTRAEA_PORT}")
        mcp.stop()
        sys.exit(1)

    orchestrator = Orchestrator(mcp, verbose=verbose_mode)

    if compare_mode:
        print(f"{YELLOW}Comparison mode: ON{RESET}")

    # Main REPL
    while True:
        try:
            question = input(f"\n{BOLD}Question:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        # Handle commands
        if question.startswith("/"):
            cmd = question.lower()
            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd == "/compare":
                compare_mode = not compare_mode
                state = "ON" if compare_mode else "OFF"
                print(f"{YELLOW}Comparison mode: {state}{RESET}")
                continue
            elif cmd == "/verbose":
                verbose_mode = not verbose_mode
                orchestrator.verbose = verbose_mode
                state = "ON" if verbose_mode else "OFF"
                print(f"{YELLOW}Verbose mode: {state}{RESET}")
                continue
            elif cmd == "/stats":
                try:
                    stats = mcp.call_tool("graph_stats", {})
                    print(f"\n{CYAN}Graph Statistics:{RESET}")
                    print(json.dumps(stats, indent=2) if isinstance(stats, dict) else stats)
                except Exception as e:
                    print(f"{RED}Error: {e}{RESET}")
                continue
            elif cmd == "/tools":
                from src.mcp_bridge import EXPOSED_TOOLS
                print(f"\n{CYAN}Available MCP Tools:{RESET}")
                for tool in EXPOSED_TOOLS:
                    print(f"  {BOLD}{tool['name']}{RESET} - {tool['description'][:80]}")
                continue
            else:
                print(f"{DIM}Unknown command. Use /quit, /compare, /verbose, /stats, or /tools{RESET}")
                continue

        # Comparison mode: raw answer first
        if compare_mode:
            print(f"\n{DIM}Querying {config.CHAT_MODEL} without graph...{RESET}")
            raw_answer = orchestrator.query_raw(question)
            print_answer("Without Graph", raw_answer, color=YELLOW)

        # GraphRAG answer
        print(f"\n{DIM}Querying {config.CHAT_MODEL} with GraphRAG via MCP...{RESET}")
        answer = orchestrator.query(question)
        print_answer("With GraphRAG", answer, color=GREEN)

        if verbose_mode:
            print_tool_log(orchestrator.tool_calls_log)


if __name__ == "__main__":
    main()
