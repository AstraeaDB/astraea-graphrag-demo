"""Configuration for the GraphRAG demo."""

import os

# AstraeaDB
ASTRAEA_HOST = os.getenv("ASTRAEA_HOST", "127.0.0.1")
ASTRAEA_PORT = int(os.getenv("ASTRAEA_PORT", "7687"))
ASTRAEA_BIN = os.getenv("ASTRAEA_BIN", "astraea-cli")
ASTRAEA_DATA_DIR = os.getenv("ASTRAEA_DATA_DIR", "")  # empty = proxy mode

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
EMBEDDING_DIM = 128

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
