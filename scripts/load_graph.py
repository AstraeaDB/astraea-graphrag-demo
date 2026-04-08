"""Load the extracted knowledge graph into AstraeaDB.

Usage:
    # Start AstraeaDB first:
    #   astraea-cli serve
    # Then:
    #   python3 scripts/load_graph.py

Set ASTRAEA_PYTHON_PATH to the directory containing the AstraeaDB Python
client if it is not already on your PYTHONPATH (e.g., /path/to/astraeadb/python).
"""

import json
import os
import sys

# Add the AstraeaDB Python client to the path if specified
_astraea_python = os.getenv("ASTRAEA_PYTHON_PATH", "")
if _astraea_python:
    sys.path.insert(0, _astraea_python)
from astraeadb import AstraeaClient

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
HOST = os.environ.get('ASTRAEA_HOST', '127.0.0.1')
PORT = int(os.environ.get('ASTRAEA_PORT', '7687'))


def load_json(filename):
    with open(os.path.join(DATA_DIR, filename)) as f:
        return json.load(f)


def main():
    # Load data
    characters = load_json('characters.json')
    locations = load_json('locations.json')
    events = load_json('events.json')
    themes = load_json('themes.json')
    chapters = load_json('chapters.json')
    text_chunks = load_json('text_chunks.json')
    edges = load_json('edges.json')
    embeddings = load_json('embeddings.json')

    # ID mapping: string_id -> AstraeaDB node_id (int)
    id_map = {}

    print(f"Connecting to AstraeaDB at {HOST}:{PORT}...")
    with AstraeaClient(host=HOST, port=PORT) as client:
        client.ping()
        print("Connected.\n")

        # --- Create nodes ---
        def create_nodes(entities, label, props_fn):
            count = 0
            for entity in entities:
                eid = entity['id']
                props = props_fn(entity)
                emb = embeddings.get(eid)
                node_id = client.create_node([label], props, embedding=emb)
                id_map[eid] = node_id
                count += 1
            return count

        print("Creating nodes...")

        n = create_nodes(characters, 'Character', lambda e: {
            'name': e['name'],
            'aliases': json.dumps(e.get('aliases', [])),
            'description': e.get('description', ''),
            'string_id': e['id'],
        })
        print(f"  Characters: {n}")

        n = create_nodes(locations, 'Location', lambda e: {
            'name': e['name'],
            'description': e.get('description', ''),
            'city': e.get('city') or '',
            'country': e.get('country', ''),
            'string_id': e['id'],
        })
        print(f"  Locations: {n}")

        n = create_nodes(events, 'Event', lambda e: {
            'name': e['name'],
            'description': e.get('description', ''),
            'book': e.get('book', 0),
            'chapter': e.get('chapter', 0),
            'string_id': e['id'],
        })
        print(f"  Events: {n}")

        n = create_nodes(themes, 'Theme', lambda e: {
            'name': e['name'],
            'description': e.get('description', ''),
            'string_id': e['id'],
        })
        print(f"  Themes: {n}")

        n = create_nodes(chapters, 'Chapter', lambda e: {
            'title': e.get('title', ''),
            'book_number': e.get('book_number', 0),
            'chapter_number': e.get('chapter_number', 0),
            'summary': e.get('summary', ''),
            'string_id': e['id'],
        })
        print(f"  Chapters: {n}")

        n = create_nodes(text_chunks, 'TextChunk', lambda e: {
            'text': e.get('text', ''),
            'book': e.get('book', 0),
            'chapter': e.get('chapter', 0),
            'chunk_index': e.get('chunk_index', 0),
            'context': e.get('context', ''),
            'string_id': e['id'],
        })
        print(f"  TextChunks: {n}")

        total_nodes = len(id_map)
        print(f"\nTotal nodes created: {total_nodes}")

        # --- Create edges ---
        print("\nCreating edges...")
        edge_count = 0
        skipped = 0
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            if src not in id_map or tgt not in id_map:
                skipped += 1
                continue
            src_id = id_map[src]
            tgt_id = id_map[tgt]
            edge_type = edge['edge_type']
            props = edge.get('properties', {})
            client.create_edge(src_id, tgt_id, edge_type, props)
            edge_count += 1

        print(f"  Edges created: {edge_count}")
        if skipped:
            print(f"  Edges skipped (missing nodes): {skipped}")

        # --- Save ID map for reference ---
        id_map_path = os.path.join(DATA_DIR, 'id_map.json')
        with open(id_map_path, 'w') as f:
            json.dump(id_map, f, indent=2)
        print(f"\nID map saved to {id_map_path}")

        print("\nGraph loaded successfully!")
        print(f"  {total_nodes} nodes, {edge_count} edges")


if __name__ == '__main__':
    main()
