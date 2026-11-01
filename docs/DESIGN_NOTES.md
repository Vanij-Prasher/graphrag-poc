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

### Hybrid retrieval implementation (`backend/rag.py`)
- **Written directly instead of using neo4j-graphrag retrievers.** The router has a custom "both" mode
  (vector candidates filtered by a graph query) and every answer must expose the exact Cypher used;
  both are simpler to control in our own code. neo4j-graphrag remains an option for later.
- **The router sees the real values in the graph** (city, policy, amenity and attraction names), so
  generated Cypher uses exact spellings like `'Pets Allowed'` instead of guessing.
- **Hard conditions vs descriptions:** only explicit conditions (named city, price, capacity, distance,
  named amenity or policy, rating) go into Cypher. Vague words ("mountains", "quiet", "romantic") go to
  vector search. Learned the hard way: see issues table.
- **Two layers of write protection:** keyword check on generated Cypher, then execution in a read-only
  transaction, so Neo4j itself refuses writes.
- **Answers are written for speech:** at most 3 short sentences, at most 2 recommendations, numbers
  written as words so a text-to-speech engine reads them naturally and consistently.
- **Low thinking level and a 60-second timeout** on every Gemini call: lower latency, and the system can
  never hang waiting for an API.
- **Router model is configurable** (`ROUTER_MODEL`, `ANSWER_MODEL`); currently `gemini-3.5-flash-lite`
  for both, end-to-end latency about 3 to 4.5 seconds per question.

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
| Script hung for minutes, then "server disconnected" | `gemini-3.8-flash` was failing on Google's side | 60-second timeout on all calls; router switched to `gemini-3.5-flash-lite` via `ROUTER_MODEL` | External APIs fail; timeouts and swappable models are essential |
| Same question produced different Cypher; once filtered by `Manali, Udaipur, Jaipur` for "mountains" | The router turned a vague word into a guessed hard filter; LLM output varies between runs | Prompt rule: only explicit conditions go into Cypher, vague descriptions go to vector search | Guessed conditions look precise but are invented; test the same question more than once |
| Answer said "I found seven hotels" for a mountain question | In the "both" route the Cypher rows are only a filter (all pet-friendly hotels), but they were passed to the answer model, which counted them | Pass filter rows to the answer model only in the "graph" route | Be explicit about what each piece of context *means* when giving it to an LLM |
| Answer formatting inconsistent (digits vs words, "5600rupees") | Answer prompt didn't specify number format | Prompt: write numbers as words, as spoken; max 3 sentences | Voice output needs explicit formatting rules |

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
- Hybrid retrieval, end to end (router + retrieval + spoken answer), about 3 to 4.5 seconds per question:

| Question | Route | Outcome |
|---|---|---|
| Pet-friendly hotels within 2 km of a fort | Graph | Correct: Amber Fort View Hotel; generated Cypher matched the hand-written one |
| Somewhere romantic with a beautiful view | Vector | Reasonable picks, but missed Lake Pichola Palace (see limitations) |
| A quiet place in the mountains that allows pets | Both | After fixes: Cypher filters pets only, vector ranking puts Manali hotels first |
- **Evaluation, 8 test questions, vector RAG vs hybrid GraphRAG** (`scripts/evaluate.py`, full table in
  `data/eval/report.md`). Same answer model and same hotel facts for both systems; only retrieval differs.

| System | Correct answers | Retrieved all expected hotels | Avg hotels retrieved | Avg seconds |
|---|---|---|---|---|
| Vector RAG (baseline) | 4/8 | 6/8 | 5.0 (always top 5) | 3.4 |
| GraphRAG (hybrid) | **8/8** | **8/8** | 1.9 (exactly the expected number) | 4.2 |

  - Vector RAG failed in two distinct ways: **retrieval misses** on "list all" questions (Q4, Q7: the
    answers span more hotels than any top-5 search can return), and **reasoning errors** when the right
    hotel *was* retrieved but the model also named wrong ones (Q3 cheapest room, Q6 distance + breakfast).
  - GraphRAG retrieved exactly the right hotels every time (15 expected across 8 questions, 15 retrieved),
    so the answer model had nothing wrong to choose from. Cost: about 0.8 s extra for the router call.
  - Scoring is automatic (hotel names in the answer vs ground truth), with no LLM judge.

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

### Hybrid retrieval limitations
- **Non-determinism:** the same question can produce different routes, Cypher or recommendations
  across runs, even at temperature 0. Needs evaluation over repeated runs, not single demos.
- **Similarity follows wording:** "romantic with a beautiful view" ranked hotels whose text contains
  "view" above the lakeside palace that best fits the meaning. Fix in production: enrich descriptions
  or tag themes (romantic, family, business) during extraction.
- **No terrain data:** "mountains" can only be handled by similarity. A `terrain` or `region` property on
  cities would make it a checkable fact.
- **Model quirks:** the lite model occasionally drops spaces ("OldManali"); a stronger answer model or a
  light post-processing step would fix this.

### Evaluation limitations
- Small test set (8 questions), one run each: results are indicative, not statistically strong.
- All 8 questions have checkable conditions, so all routed to the graph. The set favours GraphRAG by
  design and does not yet test the vector or "both" routes; fuzzy questions should be added.
- With 20 hotels, a top-5 vector search already sees 25% of the data. On a realistic dataset vector RAG
  would miss far more, so the gap would likely grow.
- Name-matching scoring is strict: a wrong hotel mentioned only to rule it out still counts as an error.

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
- [x] Router + text-to-Cypher + hybrid retrieval (graph / vector / both, with fallback)
- [x] Plain vector RAG baseline, scored on the 8 test questions
- [ ] FastAPI backend
- [ ] Next.js frontend on Vercel showing the answer and the graph path used
