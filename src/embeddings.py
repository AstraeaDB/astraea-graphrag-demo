"""Embedding generation using EmbeddingGemma via Ollama.

Matryoshka truncation to 128 dimensions with L2 re-normalization.
"""

import math
import requests
from src import config


def _normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed(text):
    """Embed a single text string. Returns a 128-dim float list."""
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": [text]},
        timeout=30,
    )
    resp.raise_for_status()
    emb = resp.json()["embeddings"][0]
    return _normalize(emb[:config.EMBEDDING_DIM])
