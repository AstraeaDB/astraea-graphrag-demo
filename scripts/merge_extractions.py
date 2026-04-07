"""Merge per-book entity extractions into unified data/*.json files."""

import json
import os

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'knowledge_graph')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ID normalization map: variant -> canonical
ID_ALIASES = {
    'char_alexandre_manette': 'char_doctor_manette',
    'char_therese_defarge': 'char_madame_defarge',
    'char_marquis_st_evremonde': 'char_marquis_st_evremonde_elder',
}


def normalize_id(entity_id):
    return ID_ALIASES.get(entity_id, entity_id)


def load_book(filename):
    with open(os.path.join(KNOWLEDGE_DIR, filename)) as f:
        return json.load(f)


def merge_entities(books, key, id_field='id'):
    """Merge entities, keeping the latest (highest book number) version for duplicates."""
    seen = {}
    for book in books:
        for entity in book.get(key, []):
            eid = normalize_id(entity[id_field])
            entity[id_field] = eid
            # For duplicates, prefer the entry from the latest book (richer description)
            # But keep the earliest first_appearance
            if eid in seen:
                existing = seen[eid]
                # Merge: keep earliest first_appearance, latest description
                if 'first_appearance' in existing and 'first_appearance' in entity:
                    ea = existing['first_appearance']
                    na = entity['first_appearance']
                    if (ea['book'], ea['chapter']) < (na['book'], na['chapter']):
                        entity['first_appearance'] = ea
                # Merge aliases
                if 'aliases' in existing and 'aliases' in entity:
                    all_aliases = list(set(existing['aliases'] + entity['aliases']))
                    entity['aliases'] = all_aliases
                # Accumulate description from all books
                if existing.get('description') and entity.get('description'):
                    if existing['description'] != entity['description']:
                        entity['description'] = existing['description'] + ' ' + entity['description']
            seen[eid] = entity
    return list(seen.values())


def merge_edges(books):
    """Merge all edges, normalizing IDs and deduplicating."""
    seen = set()
    result = []
    for book in books:
        for edge in book.get('edges', []):
            edge['source'] = normalize_id(edge['source'])
            edge['target'] = normalize_id(edge['target'])
            # Dedup key: source + target + edge_type + relationship
            rel = edge.get('properties', {}).get('relationship', '')
            dedup_key = (edge['source'], edge['target'], edge['edge_type'], rel)
            if dedup_key not in seen:
                seen.add(dedup_key)
                result.append(edge)
    return result


def add_themes():
    """Create theme nodes based on the novel's major themes."""
    return [
        {
            "id": "theme_resurrection",
            "name": "Resurrection",
            "description": "The central theme of being 'recalled to life' -- Dr. Manette's release from prison, Darnay's acquittals, and ultimately Carton's spiritual resurrection through self-sacrifice. The phrase 'recalled to life' echoes throughout the novel as characters are repeatedly brought back from death or near-death."
        },
        {
            "id": "theme_sacrifice",
            "name": "Sacrifice",
            "description": "Self-sacrifice as the highest expression of love, embodied most powerfully by Sydney Carton's decision to die in Darnay's place. Also seen in Lucie's devotion to her father, and in smaller acts throughout the novel."
        },
        {
            "id": "theme_duality",
            "name": "Duality",
            "description": "The novel is structured around doubles and opposites: two cities (London and Paris), two nations, two lookalikes (Carton and Darnay), order and disorder, love and hatred, light and darkness. The famous opening paragraph establishes this theme of paired extremes."
        },
        {
            "id": "theme_revolution_violence",
            "name": "Revolution and Violence",
            "description": "The French Revolution as both justified response to aristocratic cruelty and a force that consumes the innocent along with the guilty. The cycle of oppression breeding violence breeding more oppression is a central concern."
        },
        {
            "id": "theme_fate_justice",
            "name": "Fate and Justice",
            "description": "The tension between human justice and cosmic fate. The aristocrats' sins are visited upon their descendants; Madame Defarge's personal vendetta masquerades as revolutionary justice; the courts are shown to be arbitrary in both England and France."
        },
        {
            "id": "theme_imprisonment",
            "name": "Imprisonment and Freedom",
            "description": "Both literal and psychological imprisonment pervade the novel. Dr. Manette's physical imprisonment echoes in his psychological relapses to shoemaking. Darnay is imprisoned multiple times. Even Carton is imprisoned by his wasted life until his final act frees him."
        }
    ]


def add_theme_edges(characters):
    """Create EMBODIES edges connecting characters to themes."""
    char_ids = {c['id'] for c in characters}
    theme_edges = [
        {"source": "char_sydney_carton", "target": "theme_sacrifice", "edge_type": "EMBODIES", "properties": {"description": "Carton sacrifices his life for Darnay, achieving spiritual resurrection through selfless love"}},
        {"source": "char_sydney_carton", "target": "theme_resurrection", "edge_type": "EMBODIES", "properties": {"description": "Carton is 'recalled to life' spiritually through his sacrifice -- 'It is a far, far better thing that I do'"}},
        {"source": "char_sydney_carton", "target": "theme_duality", "edge_type": "EMBODIES", "properties": {"description": "Carton is Darnay's physical double but his moral opposite -- the dissolute man who becomes the novel's greatest hero"}},
        {"source": "char_doctor_manette", "target": "theme_resurrection", "edge_type": "EMBODIES", "properties": {"description": "Manette is literally 'recalled to life' after eighteen years buried alive in the Bastille"}},
        {"source": "char_doctor_manette", "target": "theme_imprisonment", "edge_type": "EMBODIES", "properties": {"description": "Manette's physical imprisonment becomes psychological -- he relapses to shoemaking under stress even after release"}},
        {"source": "char_lucie_manette", "target": "theme_sacrifice", "edge_type": "EMBODIES", "properties": {"description": "Lucie devotes herself entirely to restoring her father and supporting her husband, the 'golden thread' binding everyone"}},
        {"source": "char_madame_defarge", "target": "theme_revolution_violence", "edge_type": "EMBODIES", "properties": {"description": "Madame Defarge represents the revolution's consuming vengeance, knitting the names of the condemned into her register"}},
        {"source": "char_madame_defarge", "target": "theme_fate_justice", "edge_type": "EMBODIES", "properties": {"description": "Her personal vendetta against the Evremonde family drives the prosecution, blurring justice and revenge"}},
        {"source": "char_charles_darnay", "target": "theme_duality", "edge_type": "EMBODIES", "properties": {"description": "Darnay lives a double identity -- French aristocrat by birth, English gentleman by choice -- torn between two cities"}},
        {"source": "char_charles_darnay", "target": "theme_fate_justice", "edge_type": "EMBODIES", "properties": {"description": "Darnay is condemned for the sins of his family despite renouncing them, showing how revolutionary justice consumes the innocent"}},
        {"source": "char_jarvis_lorry", "target": "theme_duality", "edge_type": "EMBODIES", "properties": {"description": "Lorry straddles two worlds as a man who claims to be 'a mere machine of business' but repeatedly shows deep compassion"}},
        {"source": "char_miss_pross", "target": "theme_sacrifice", "edge_type": "EMBODIES", "properties": {"description": "Miss Pross sacrifices her hearing in the confrontation with Madame Defarge, defending Lucie with fierce loyalty"}},
    ]
    # Only include edges for characters that exist
    return [e for e in theme_edges if e['source'] in char_ids]


def add_chapter_sequence(chapters):
    """Create NEXT_CHAPTER edges between sequential chapters."""
    edges = []
    sorted_chapters = sorted(chapters, key=lambda c: (c['book_number'], c['chapter_number']))
    for i in range(len(sorted_chapters) - 1):
        curr = sorted_chapters[i]
        nxt = sorted_chapters[i + 1]
        # Only link within same book
        if curr['book_number'] == nxt['book_number']:
            edges.append({
                "source": curr['id'],
                "target": nxt['id'],
                "edge_type": "NEXT_CHAPTER",
                "properties": {}
            })
    return edges


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    b1 = load_book('book1_entities.json')
    b2 = load_book('book2_entities.json')
    b3 = load_book('book3_entities.json')
    books = [b1, b2, b3]

    # Merge entities
    characters = merge_entities(books, 'characters')
    locations = merge_entities(books, 'locations')
    events = merge_entities(books, 'events')
    chapters = merge_entities(books, 'chapters')
    text_chunks = merge_entities(books, 'text_chunks')
    themes = add_themes()

    # Merge and enrich edges
    edges = merge_edges(books)
    edges.extend(add_theme_edges(characters))
    edges.extend(add_chapter_sequence(chapters))

    # Write output files
    def save(filename, data):
        path = os.path.join(DATA_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  {filename}: {len(data)} items")

    print("Writing data files:")
    save('characters.json', characters)
    save('locations.json', locations)
    save('events.json', events)
    save('themes.json', themes)
    save('chapters.json', chapters)
    save('text_chunks.json', text_chunks)
    save('edges.json', edges)

    # Summary
    total_nodes = len(characters) + len(locations) + len(events) + len(themes) + len(chapters) + len(text_chunks)
    print(f"\nTotal: {total_nodes} nodes, {len(edges)} edges")


if __name__ == '__main__':
    main()
