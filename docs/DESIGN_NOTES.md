# Design notes: decisions, issues and learnings

A running log of *why* things were built the way they were, what went wrong, and what was learned.
Kept up to date as the project evolves.

---

## 1. Goal

Build an end-to-end proof of concept of **RAG over a knowledge graph**, as the retrieval core for a
conversational (voice) assistant. The assistant must answer precise, multi-hop questions such as
*"Which pet-friendly hotels are within 2 km of a fort?"*

---

## 2. Key decisions and why

### Graph RAG instead of plain vector RAG
- Vector RAG retrieves text chunks that *look similar* to the question. It struggles when the answer
  is spread across several facts (location + amenity + policy + distance).
- A knowledge graph stores entities and relationships, so the system can *follow connections* and
  filter on them exactly.
- Plan: **hybrid**. Vector search finds the starting point for fuzzy questions; graph traversal and
  Cypher collect the connected facts; the LLM writes the answer.

### Domain and data
- **Hotels**: naturally relationship-heavy (city, amenities, rooms, attractions, policies) and close
  to what a real voice bot would answer.
- **Synthetic data** with fictional hotel names (real attractions): no confidentiality issues.
- **Policies exist only in the text files**, on purpose. Real company data is mostly documents, so the
  POC must show knowledge-graph construction from unstructured text, not just a CSV import.
- **8 test questions with ground-truth answers computed from the data** (`data/test_questions.md`),
  so vector RAG vs graph RAG can be compared objectively instead of by eye.

### Stack
| Choice | Why |
|---|---|
| Neo4j AuraDB (free) | Graph database that also supports vector indexes: one store for both |
| neo4j-graphrag | Neo4j's first-party library, narrower and more deliberate than general frameworks |
| Gemini (LLM + Gemini Embedding 001) | Strong models, free tier, one provider and one key |
| FastAPI + Next.js on Vercel | Python for AI logic, polished UI, simple deployment |

- Groq was considered and kept as an option: its very low latency matters for **voice**, where
  response time is part of the user experience.

### Ingestion is separate from serving
- Graph building is slow and LLM-heavy, so it runs **locally as one-time scripts** (`ingestion/`).
- Only question answering is deployed. Serverless functions (Vercel) have time limits that a full
  graph build would exceed.
- This mirrors production systems, where ingestion pipelines run separately from the query service.
- Supabase was considered for hosting the backend, but its Edge Functions run TypeScript on Deno, not
  Python, so FastAPI and neo4j-graphrag cannot run there.

### Graph modelling
```
(Hotel)-[:LOCATED_IN]->(City)
(Hotel)-[:HAS_AMENITY]->(Amenity)
(Hotel)-[:OFFERS]->(RoomType {price_inr, max_guests})
(Hotel)-[:NEAR {distance_km}]->(Attraction)-[:LOCATED_IN]->(City)
(Hotel)-[:HAS_POLICY {details}]->(Policy {name})
(Document)-[:DESCRIBES]->(Hotel)
```
- **Shared Policy nodes** (e.g. one "Pets Allowed" node) with hotel-specific details on the
  relationship (pet fee, cancellation hours). Questions become simple patterns:
  `(h)-[:HAS_POLICY]->(:Policy {name: 'Pets Allowed'})`.
- **Distance lives on the NEAR relationship**, because it belongs to the pair, not to either node.
- **Document nodes keep the source text** linked to each hotel: provenance (every fact traceable to
  its source) and the input for vector search later.
- **Uniqueness constraints + MERGE** make every ingestion script safe to re-run without duplicates.

### Cypher, not SQL
- Cypher describes the *shape* of what you want: `(h:Hotel)-[:NEAR]->(a:Attraction)`.
- The same question in SQL needs a JOIN per hop, which grows quickly for multi-hop questions.
- Later, the LLM will generate Cypher from natural language (text-to-Cypher), the graph equivalent of
  text-to-SQL.

### LLM extraction with structured output
- Gemini must return a fixed JSON schema (Pydantic model), at temperature 0, with an instruction not
  to guess. This keeps extraction consistent and machine-checkable.
- Extractions are **cached** in `data/extracted/policies.json`, so the graph can be rebuilt without
  calling the LLM again.

### Vector search
- **Gemini Embedding 001**, 3,072 dimensions, cosine similarity, stored on the `Document` nodes with a
  Neo4j vector index (`document_embeddings`). Graph and vectors live in the same database, so a vector
  hit can be followed straight into the graph (document → hotel → city, policies, amenities).
- **Task types:** documents are embedded as `RETRIEVAL_DOCUMENT`, questions as `RETRIEVAL_QUERY`;
  the model optimises each side for retrieval.
- **One document = one chunk**, because each hotel document is short. Real documents would be chunked first.
- All 20 documents are embedded in **one API request**.

### Choosing between graph and vector search (routing)
Vector search *understands* loosely worded questions; graph queries *answer precisely*. The system uses
each for what it does best:

| Question type | Example | Route |
|---|---|---|
| Precise: clear conditions, numbers, filters | "Pet-friendly hotels within 2 km of a fort" | Graph (text-to-Cypher) |
| Fuzzy: feelings, descriptions | "Somewhere romantic with a nice view" | Vector |
| Mixed: fuzzy wish plus hard rule | "A quiet place in the mountains that allows pets" | Both: vector candidates, filtered by graph conditions |

```
User question
     │
     ▼
Router (Gemini): picks graph / vector / both, writes Cypher if needed
     │
     ├── Graph  → run the Cypher → exact facts
     ├── Vector → embed question → similar hotels → add their facts from the graph
     └── Both   → vector candidates, filtered by the graph conditions
     │
     ▼
Answer (Gemini): turns the retrieved facts into a short, spoken-style answer
```

- **Text-to-Cypher:** Gemini receives the graph schema and the question, and writes the Cypher query.
- **Routing and Cypher generation happen in one call**, so each question costs 2 LLM calls
  (route + answer), which matters under free-tier limits.
- **Fallback:** if generated Cypher fails or returns nothing, fall back to vector search instead of
  answering "I don't know". LLM-written queries are sometimes wrong.
- **Read-only safeguard:** generated Cypher containing `CREATE`, `DELETE`, `SET`, `MERGE` etc. is rejected.
  An LLM never writes to the database unchecked.
- **Transparency:** every answer returns the route taken and the Cypher used, so it can be checked.

### Secrets
- Credentials live only in `.env`, which `.gitignore` excludes. `.env.example` documents the needed
  variables without real values. `git status` is checked before every commit.

---

## 3. Issues hit and how they were fixed

| Issue | Cause | Fix | Lesson |
|---|---|---|---|
| Neo4j login details differ from tutorials | Newer AuraDB instances use the instance ID as username and database name | Read credentials from the downloaded file; pass `NEO4J_DATABASE` explicitly | Trust the provider's credentials file over generic examples |
| `ServerError` (5xx) during extraction | Gemini servers temporarily overloaded | Retry with increasing waits | Transient errors are normal; design for them |
| A crash lost all extracted results | Results were only saved at the end of the run | Save progress after every batch | Long LLM jobs must be resumable |
| `429 RESOURCE_EXHAUSTED` | Free tier allows ~20 requests/day per model; retries also count | Batch 5 hotels per call (20 requests → 4), pause between calls, stop immediately on a daily-quota error | Retries can burn quota; fail fast on limits that will not recover |
| `404` for `gemini-2.5-flash-lite` | Model listed, but no longer available to new API keys | Listed models via the API and tested candidates with a one-word request; switched to `gemini-3.5-flash-lite`; fail fast on 404 | Verify model availability programmatically instead of guessing |
| One model for everything would exhaust quota | Daily limits are per model | `EXTRACTION_MODEL` (one-time ingestion) separate from `GEMINI_MODEL` (live answers) | Match model to task: lighter models for simple extraction |
| AuraDB Free instance paused | Free tier pauses instances after a few days without activity | Resume from the console (data is kept); check it is running before demos | Free infrastructure needs keep-alive habits; ingestion is scripted so the graph can be rebuilt if needed |
| "Property `embedding` does not exist" warning | First run queried a property no node had yet | None needed: harmless, disappears once embeddings exist | Read warnings before reacting; not every warning is an error |
| `db.index.vector.queryNodes` deprecated | Newer Neo4j versions replace it with the `SEARCH` clause | Still works; the retrieval layer will use the neo4j-graphrag library | Track platform changes, but don't rewrite working code mid-phase |

---

## 4. Results so far

- Structured load: **106 nodes, 208 relationships** from 5 CSVs.
- Policy extraction: **120 / 120 fields correct** (cancellation hours, pets, breakfast, payment,
  check-in/out for 20 hotels), using **4 batched API calls** with `gemini-3.5-flash-lite`.
- Test question 2 (*pet-friendly hotels within 2 km of a fort*) answered correctly with one Cypher
  query combining CSV facts and LLM-extracted facts: **Amber Fort View Hotel**, in milliseconds.
- Gemini Embedding 001 returns **3,072-dimension** vectors (needed for the vector index).
- Vector index: **20 documents embedded in one request**, index online and fully populated.
- Fuzzy question *"a quiet place for a family trip in the mountains"*: **4 of the top 5 results are
  Manali hotels**, though "Manali" is not in the question (the 5th, a hill resort, is also reasonable).
- Precise question *"pet-friendly hotel near a fort"* with vector search alone: the correct hotel ranked
  first, but **4 of the top 5 were wrong** (no pets, or no fort nearby), and all scores sat between
  0.835 and 0.865, so there is no clear cut-off. The graph query returns exactly one, correct, result.
  This is the main argument for hybrid retrieval.

---

## 5. Nuances and limitations (worth stating honestly)

### Inconsistent handling of unstated values
For two hotels whose text only says pets are welcome, without mentioning a fee, Gemini returned a
fee of **₹0 for one (H13)** and **unknown (null) for the other (H20)**. Neither is wrong, but similar
wording was treated differently.

How to handle it in production:
- Stricter schema: replace a nullable number with an explicit status, e.g.
  `pet_fee_status: "free" | "charged" | "deposit" | "not_mentioned"`.
- Validation rules after extraction (e.g. a fee of 0 must be backed by words like "free" or
  "no extra cost" in the text).
- Keep an evaluation set and re-score extraction whenever the prompt, schema or model changes.

### Other limitations
- Synthetic text is clean and consistent; real documents are messier, so real-world extraction
  accuracy will be lower than 100% and needs ongoing evaluation.
- Free-tier quotas (about 20 requests/day per model) are too tight for continuous development or
  evaluation runs; a paid tier with a budget cap would be used beyond the POC.
- The policy schema is fixed in code. New kinds of facts need a schema change; fully open-ended
  extraction is more flexible but less predictable.

---

## 6. Next steps

- [x] Vector index on hotel documents (Gemini Embedding 001)
- [ ] Router + text-to-Cypher + hybrid retrieval (graph / vector / both, with fallback)
- [ ] Plain vector RAG baseline, scored on the 8 test questions
- [ ] FastAPI backend
- [ ] Next.js frontend on Vercel showing the answer and the graph path used
