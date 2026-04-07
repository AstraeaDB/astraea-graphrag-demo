# Demo Script: GraphRAG with AstraeaDB

A voice-over and action script for demonstrating the GraphRAG system live. Estimated total time: 12-15 minutes.

---

## Pre-Demo Checklist

Before starting, ensure everything is running:

```bash
# Terminal 1: Ollama should already be running (check with)
ollama list
# Should show: gemma3:4b, embeddinggemma

# Terminal 2: Start AstraeaDB
/Users/jamesharris/Documents/astraeadb/target/debug/astraea-cli serve

# Terminal 3: Verify the graph is loaded (if not, run):
python3 scripts/load_graph.py

# Terminal 3: Run the test suite to confirm everything works
python3 test_demo.py
# Should show: Results: 24/24 passed, 0 failed
```

Have two terminal windows ready:
- **Terminal A**: For running the demo (`python3 main.py`)
- **Terminal B**: Visible but idle (for showing startup commands if needed)

---

## Act 1: The Problem (2 minutes)

### Voice-Over

> Large language models are impressive, but they have a well-known weakness: when you ask about specific documents or domain knowledge, they hallucinate. They fill in gaps with plausible-sounding but incorrect information.
>
> Today I'm going to show you how a very small model -- Google's Gemma 3 at just 4 billion parameters, running entirely on my local machine -- can answer detailed, multi-hop questions about a 300-page novel with remarkable accuracy. The secret isn't a bigger model. It's a smarter way to give the model access to information.
>
> We're going to use three things: AstraeaDB, a graph database that combines property graphs with vector search; Ollama for local model inference; and MCP -- the Model Context Protocol -- which lets the LLM call database tools directly.
>
> The novel is Charles Dickens's *A Tale of Two Cities*. We've extracted a knowledge graph: 229 nodes covering characters, locations, events, themes, chapters, and actual text passages, connected by 317 edges representing relationships, event participation, and thematic links. Every node has a 128-dimensional embedding for semantic search.

### Action

*(No terminal action yet -- this is the intro.)*

---

## Act 2: What the Model Knows Without Help (3 minutes)

### Voice-Over

> Let's start by seeing what gemma3:4b can do on its own -- no graph, no tools, just the raw model.

### Action

```bash
python3 main.py --compare --verbose
```

### Voice-Over

> I'm starting the demo in comparison mode, which asks the same question twice: once to the raw model, and once with GraphRAG. I've also enabled verbose mode so you can see exactly which graph tools the model calls.

*(Wait for the startup messages to appear.)*

> Let's start with something specific.

### Action: Type the first question

```
How does Dr. Manette's imprisonment connect to Charles Darnay's trial?
```

### Voice-Over (while waiting for response)

> This is a multi-hop question. The answer requires connecting Dr. Manette's backstory in Book One to the trial in Book Three, passing through the Evremonde family connection. Let's see how the raw model handles it.

*(Wait for the "Without Graph" answer to appear.)*

> So the raw model gives us... *(read and react to the response -- it will likely be vague, generic, or contain inaccuracies about specific plot details)*.
>
> Now watch what happens with GraphRAG.

*(Wait for the tool calls and "With GraphRAG" answer.)*

> Look at the tool calls in verbose mode. The model first searched for the characters, then explored their relationships, then found the connecting events. It's *reasoning through the graph structure*. And the answer is grounded in specific nodes and edges -- not hallucinated.

---

## Act 3: Exploring Relationships (3 minutes)

### Voice-Over

> Let's try a relationship question. This is where graph traversal really shines.

### Action: Type the second question

```
What is Sydney Carton's relationship with Charles Darnay, and how does it drive the plot?
```

### Voice-Over (while waiting)

> Carton and Darnay are central to the novel's theme of duality -- they're physical doubles. The graph should capture this through RELATED_TO edges and their shared PARTICIPATES_IN edges with key events.

*(Point out the tool calls as they appear in verbose mode.)*

> Notice the model is calling `neighbors` to find Carton's relationships, then `get_node` to read the details. It's building its understanding step by step from the graph, just like a researcher would.

*(After the answer appears.)*

> The GraphRAG answer traces the connection from their physical resemblance to the trial scene to the final sacrifice. Every claim is backed by a node or edge in the graph.

---

## Act 4: Thematic Analysis (2 minutes)

### Voice-Over

> Now let's ask something more abstract -- a thematic question that typically trips up small models.

### Action: Type the third question

```
What are the major themes of the novel and which characters embody them?
```

### Voice-Over

> Themes are tricky for a 4-billion parameter model. But our graph has explicit Theme nodes connected to characters via EMBODIES edges. The model can discover these structured relationships rather than guessing.

*(Wait for response.)*

> The model found all six themes -- Resurrection, Sacrifice, Duality, Revolution, Fate, and Imprisonment -- and correctly mapped characters to them. Carton embodies Sacrifice and Resurrection. Madame Defarge embodies Revolution and Fate. This level of structured thematic analysis would be very difficult for gemma3:4b to produce from memory alone.

---

## Act 5: Text Grounding (2 minutes)

### Voice-Over

> One of the most powerful features is the ability to ground answers in actual text from the novel. Let's ask for a specific passage.

### Action: Type the fourth question

```
What is the most famous opening line, and what themes does it establish?
```

### Voice-Over

> Watch the tool calls -- the model should use `vector_search` here to find the TextChunk node containing the opening paragraph. The embedding similarity will match our question to the right passage.

*(Wait for response. The model should find and quote "It was the best of times, it was the worst of times..." and connect it to the Duality theme.)*

> There it is -- the actual text from the novel, retrieved by semantic search, and the model correctly identifies it as establishing the theme of duality. The answer includes the real quote, not a paraphrase from memory.

---

## Act 6: Graph Statistics (1 minute)

### Voice-Over

> Let me show you what's actually in this knowledge graph.

### Action: Type the command

```
/stats
```

### Voice-Over

> *(Read the statistics as they appear.)* 229 nodes across six types, 317 edges connecting them. This is a modest graph -- it fits the novel. But it gives the model structured access to every important character, location, event, and theme in the book, plus 77 key text passages.

### Action: Type the command

```
/tools
```

### Voice-Over

> And these are the ten MCP tools the model can call. It can search by label, get node details, traverse relationships, find shortest paths between entities, run semantic or hybrid searches, extract readable subgraph summaries, and even execute GQL queries directly. The model decides which tools to use based on the question.

---

## Act 7: The Power of Graph Traversal (2 minutes)

### Voice-Over

> Let me show one more question that really demonstrates why graph structure matters. This requires following a chain of connections that pure vector search would miss.

### Action: Type the fifth question

```
How is the spilled wine outside Defarge's shop connected to the storming of the Bastille?
```

### Voice-Over

> This is a question about narrative foreshadowing that spans from Book One to Book Three. The spilled wine that stains the people of Saint Antoine red is Dickens's metaphor for the blood that will flow in the revolution. Let's see if the model can trace this through the graph.

*(Wait for the response. Point out the tool calls.)*

> The model found the wine cask event in Book One, the storming of the Bastille in Book Three, and the connecting location -- Saint Antoine. It followed edges through the graph to build a narrative arc across the entire novel. A raw embedding search couldn't do this -- it requires structural traversal.

---

## Act 8: Wrap-Up (1 minute)

### Voice-Over

> Let me turn off comparison mode and summarize what we've seen.

### Action

```
/compare
```

### Voice-Over

> What we demonstrated today is that you don't need a 70-billion parameter model or massive context windows to get accurate, grounded answers about a specific text. A 4-billion parameter model running on a laptop, combined with a knowledge graph in AstraeaDB accessed through MCP tools, can:
>
> - Answer multi-hop relationship questions by traversing the graph
> - Perform thematic analysis using structured theme-to-character edges
> - Ground answers in actual novel text via semantic vector search
> - Trace narrative connections across hundreds of pages using shortest-path algorithms
>
> The graph structure provides what the model lacks: precise, retrievable, structured knowledge. And MCP provides the bridge -- it lets the model call database tools as naturally as it generates text.
>
> Everything runs locally. No cloud APIs, no data leaving the machine. Just Ollama, AstraeaDB, and MCP.
>
> Thank you.

### Action

```
/quit
```

---

## Backup Questions

If the demo has time or a live question comes up, here are additional questions that showcase different capabilities:

| Question | Showcases |
|----------|-----------|
| "Who is Miss Pross and what role does she play in the climax?" | Character lookup + event participation |
| "Compare the fates of Madame Defarge and Sydney Carton." | Parallel neighbor queries, thematic contrast |
| "What happened at the Bastille?" | Location-based event discovery |
| "Quote a passage where the narrator describes hunger in Paris." | Vector search on TextChunks |
| "What is the significance of 'recalled to life' in the novel?" | Theme exploration + text passages |
| "Trace the path from Dr. Manette's imprisonment to Carton's sacrifice." | Multi-hop shortest_path traversal |
| "Which characters are associated with the theme of duality?" | Theme node + EMBODIES edge traversal |
| "What is Tellson's Bank and why is it important?" | Location node + connected characters/events |

## Troubleshooting During Demo

| Problem | Fix |
|---------|-----|
| "Failed to start MCP bridge" | AstraeaDB isn't running. Open Terminal B: `/Users/jamesharris/Documents/astraeadb/target/debug/astraea-cli serve` |
| Tool calls return errors | The graph may not be loaded. Run `python3 scripts/load_graph.py` |
| Model doesn't call tools | This happens occasionally with small models. Ask a more specific question, or restart |
| Very slow responses | Check that Ollama is using GPU. Run `ollama ps` to verify |
| "No response from the model" | Ollama may have crashed. Check `ollama ps` and restart if needed |
