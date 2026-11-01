# GraphRAG POC: Hotel Assistant

A proof of concept for **Retrieval-Augmented Generation over a knowledge graph**, built as the
retrieval core for a conversational (voice) assistant. It answers multi-hop questions such as
*"Which pet-friendly hotels are within 2 km of a fort?"*, which plain vector RAG struggles with
because the answer is spread across several connected facts.

## Why a knowledge graph?

| | Vector RAG | Graph RAG |
|---|---|---|
| Retrieves | Text chunks similar to the question | Entities and the relationships between them |
| Good at | Direct, single-fact questions | Multi-hop, filtered, relational questions |
| Weak at | Combining facts spread across documents | Fuzzy, open-ended phrasing (so this POC uses both) |

This POC uses a **hybrid** approach: vector search finds the right starting entities,
graph traversal and Cypher queries collect the connected facts, and the LLM writes the answer.

## Architecture

```
LOCAL (one-time ingestion)               CLOUD
data/ ──► ingestion/build_graph.py ──► Neo4j AuraDB ◄── backend/ (FastAPI) ◄── frontend/ (Next.js)
           (Gemini extracts entities          (graph + vector index)        on Vercel
            and policies from text)
```

Ingestion runs separately from the query service, as in production systems:
the slow, LLM-heavy graph build never runs inside a request.

## Stack

| Layer | Choice |
|---|---|
| Graph + vector store | Neo4j AuraDB |
| GraphRAG framework | neo4j-graphrag (Neo4j's first-party Python library) |
| LLM | Google Gemini |
| Embeddings | Gemini Embedding 001 |
| Backend | FastAPI |
| Frontend + hosting | Next.js + Vercel AI SDK on Vercel |

## Data

Synthetic dataset of 20 fictional hotels across Mumbai, Goa, Jaipur, Manali and Udaipur
(attractions are real places):

- `data/structured/`: hotels, room types, amenities, attractions and distances (CSV)
- `data/unstructured/`: one document per hotel; **policies exist only here** and are extracted by the LLM
- `data/test_questions.md`: 8 multi-hop questions with ground-truth answers, used to compare vector RAG vs graph RAG

Regenerate with `python scripts/generate_data.py`.

### Graph schema

```
(Hotel)-[:LOCATED_IN]->(City)
(Hotel)-[:HAS_AMENITY]->(Amenity)
(Hotel)-[:OFFERS]->(RoomType {price, max_guests})
(Hotel)-[:NEAR {distance_km}]->(Attraction)-[:LOCATED_IN]->(City)
(Hotel)-[:HAS_POLICY]->(Policy)
```

## Project structure

```
data/         dataset and test questions
scripts/      data generation
ingestion/    graph building (runs locally)
backend/      FastAPI question-answering API
frontend/     Next.js demo UI
```

## Status

- [x] Dataset and graph schema
- [x] Load structured data into Neo4j
- [x] LLM extraction of policies from text (Gemini, structured output, 120/120 fields correct)
- [x] Vector index on hotel documents (Gemini Embedding 001)
- [ ] Router + text-to-Cypher + hybrid retrieval (graph / vector / both, with fallback)
- [ ] Vector RAG baseline and evaluation on test questions
- [ ] FastAPI backend
- [ ] Next.js frontend on Vercel

## Setup

1. Copy `.env.example` to `.env` and fill in your Neo4j and Gemini credentials.
2. Install dependencies: `pip install -r requirements.txt`
3. Check connections: `python scripts/test_connection.py`
4. Build the graph (run once, in order):
   ```bash
   python ingestion/load_structured.py    # CSVs -> graph
   python ingestion/extract_policies.py   # text -> policies (uses cached extractions if present)
   python ingestion/embed_documents.py    # documents -> embeddings + vector index
   ```
5. Try semantic search: `python scripts/vector_search.py "a quiet place for a family trip in the mountains"`

## Design notes

Why things were built this way, issues encountered and how they were fixed, results and known limitations: [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)
