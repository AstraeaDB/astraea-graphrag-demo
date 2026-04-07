"""Generate 128-dimensional embeddings for all graph entities.

Uses EmbeddingGemma via Ollama with Matryoshka truncation to 128 dimensions.
"""

import json
import math
import os
import sys
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'embeddinggemma')
EMBEDDING_DIM = 128
BATCH_SIZE = 32


def normalize(vec):
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_batch(texts):
    """Embed a batch of texts via Ollama, truncate to EMBEDDING_DIM, and re-normalize."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    embeddings = resp.json()["embeddings"]
    return [normalize(emb[:EMBEDDING_DIM]) for emb in embeddings]


def get_text_for_entity(entity, entity_type):
    """Extract the text to embed for a given entity."""
    if entity_type == 'text_chunks':
        return entity.get('text', '')
    if entity_type == 'chapters':
        return f"{entity.get('title', '')}. {entity.get('summary', '')}"
    if entity_type == 'characters':
        name = entity.get('name', '')
        aliases = ', '.join(entity.get('aliases', []))
        desc = entity.get('description', '')
        return f"{name} (also known as: {aliases}). {desc}" if aliases else f"{name}. {desc}"
    if entity_type == 'themes':
        return f"{entity.get('name', '')}. {entity.get('description', '')}"
    # locations, events
    name = entity.get('name', '')
    desc = entity.get('description', '')
    return f"{name}. {desc}"


def main():
    # Collect all texts to embed
    items = []  # (entity_id, text)
    for entity_type in ['characters', 'locations', 'events', 'themes', 'chapters', 'text_chunks']:
        filepath = os.path.join(DATA_DIR, f'{entity_type}.json')
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            entities = json.load(f)
        for entity in entities:
            text = get_text_for_entity(entity, entity_type)
            if text.strip():
                items.append((entity['id'], text))

    print(f"Generating embeddings for {len(items)} entities...")
    print(f"  Model: {OLLAMA_EMBED_MODEL}")
    print(f"  Dimension: {EMBEDDING_DIM} (Matryoshka truncation)")

    # Embed in batches
    embeddings = {}
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        vecs = embed_batch(texts)
        for eid, vec in zip(ids, vecs):
            embeddings[eid] = vec
        done = min(i + BATCH_SIZE, len(items))
        print(f"  [{done}/{len(items)}]")

    # Save
    output_path = os.path.join(DATA_DIR, 'embeddings.json')
    with open(output_path, 'w') as f:
        json.dump(embeddings, f)
    print(f"\nSaved {len(embeddings)} embeddings to {output_path}")

    # Validate dimensions
    dims = {len(v) for v in embeddings.values()}
    print(f"Embedding dimensions: {dims}")


if __name__ == '__main__':
    main()
