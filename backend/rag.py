"""Hybrid GraphRAG: routes each question to graph search, vector search or both,
then answers in a short, spoken style.

    from backend.rag import GraphRAG
    rag = GraphRAG()
    result = rag.ask("Which pet-friendly hotels are within 2 km of a fort?")

Two LLM calls per question:
    1. Router (ROUTER_MODEL): picks the route and writes the Cypher query if needed
    2. Answer (ANSWER_MODEL): turns the retrieved facts into a short answer
"""
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from neo4j import READ_ACCESS, GraphDatabase, NotificationMinimumSeverity
from pydantic import BaseModel, Field

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_INDEX = "document_embeddings"
VECTOR_TOP_K = 5             # hotels returned by a pure vector search
VECTOR_CANDIDATES_BOTH = 10  # candidates searched before graph filtering ("both" route)
MAX_HOTELS = 5               # hotels passed to the answer model

# Any generated query containing these words is rejected: the LLM may only read.
FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|CALL|LOAD)\b", re.IGNORECASE
)


class QuotaExceeded(Exception):
    pass


class RouteDecision(BaseModel):
    route: str = Field(description="Exactly one of: graph, vector, both")
    reasoning: str = Field(description="One short sentence explaining the choice")
    cypher: Optional[str] = Field(
        description="Read-only Cypher query for the 'graph' or 'both' route; null for 'vector'"
    )
    search_text: Optional[str] = Field(
        description="The descriptive part of the question to search by meaning, "
        "for the 'vector' or 'both' route; null for 'graph'"
    )


SCHEMA = """
Nodes and properties:
  (:Hotel {id, name, area, star_rating: int, guest_rating: float, check_in, check_out, children_policy})
  (:City {name})
  (:RoomType {name, price_inr: int (per night), max_guests: int})
  (:Amenity {name})
  (:Attraction {name, category})
  (:Policy {name})

Relationships:
  (:Hotel)-[:LOCATED_IN]->(:City)
  (:Hotel)-[:OFFERS]->(:RoomType)
  (:Hotel)-[:HAS_AMENITY]->(:Amenity)
  (:Hotel)-[:NEAR {distance_km: float}]->(:Attraction)
  (:Attraction)-[:LOCATED_IN]->(:City)
  (:Hotel)-[:HAS_POLICY {hours_before_checkin, conditions, fee_inr_per_night}]->(:Policy)
"""

ROUTER_PROMPT = """You route questions for a hotel assistant backed by a Neo4j knowledge graph.

GRAPH SCHEMA:
{schema}
VALUES THAT EXIST IN THE GRAPH (use exactly these spellings):
  Cities: {cities}
  Policy names: {policies}
  Amenity names: {amenities}
  Attraction categories: {categories}
  Attraction names: {attractions}

Choose ONE route:
- "graph": the question has clear, checkable conditions (city, price, capacity, distance, amenity,
  policy, rating, named attraction). Write a Cypher query that answers it exactly.
- "vector": the question is descriptive or about feel/atmosphere ("romantic", "quiet", "good views")
  with no checkable conditions. Put the descriptive part in search_text. No Cypher.
- "both": the question mixes a descriptive wish with checkable conditions. Put the descriptive part in
  search_text, and write a Cypher query for ONLY the checkable conditions.

Hard conditions vs descriptions:
- Only EXPLICIT conditions go into Cypher: a named city or attraction, a price or capacity, a distance,
  a named amenity or policy, a rating.
- Vague descriptions ("mountains", "beach", "quiet", "romantic", "near nature") NEVER go into Cypher.
  Do not guess which cities match a description; put the description in search_text instead.

Cypher rules:
- Read-only: use only MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, UNWIND.
- Always return h.id AS hotel_id and h.name AS hotel, plus the fields that answer the question
  (e.g. distance, price, room name, city).
- Use the variable h for Hotel. Add LIMIT 20.
- Policies are nodes: (h)-[:HAS_POLICY]->(:Policy {{name: 'Pets Allowed'}}).
- Distance is on the NEAR relationship: (h)-[n:NEAR]->(a:Attraction) WHERE n.distance_km <= 2.

QUESTION: {question}"""

ANSWER_PROMPT = """You are a friendly hotel voice assistant. Answer the guest's question using ONLY the
information below.
- Speak naturally, as in a phone call: at most 3 short sentences. No lists, no markdown, no emojis.
- {selection_rule}
- Give the specific facts that answer the question.
- Write every number as words, the way it is spoken: "three thousand six hundred rupees",
  "zero point seven kilometres", "seventy-two hours". This text is read aloud by a speech engine.
- If the information does not answer the question, or no hotel matches, say so honestly.

How these hotels were found: {note}

QUESTION: {question}

{query_rows}HOTEL INFORMATION:
{hotels}"""

ROUTE_NOTES = {
    "graph": "a database query that checked the question's conditions exactly.",
    "vector": "similarity search. They may not all fit; only recommend the ones that do.",
    "both": "similarity search, then filtered by a database query on the question's conditions.",
}

SELECTION_SPOKEN = "If several hotels fit, say how many there are, then describe at most the 2 best fits."
SELECTION_COMPLETE = "Name every hotel that fits (used for evaluation), with one short fact each."

HOTEL_FACTS = """
UNWIND $ids AS hid
MATCH (h:Hotel {id: hid})-[:LOCATED_IN]->(c:City)
OPTIONAL MATCH (d:Document)-[:DESCRIBES]->(h)
OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)
WITH h, c, d, collect(DISTINCT a.name) AS amenities
OPTIONAL MATCH (h)-[hp:HAS_POLICY]->(p:Policy)
WITH h, c, d, amenities, collect(DISTINCT {name: p.name, details: properties(hp)}) AS policies
OPTIONAL MATCH (h)-[:OFFERS]->(r:RoomType)
WITH h, c, d, amenities, policies,
     collect(DISTINCT {name: r.name, price: r.price_inr, guests: r.max_guests}) AS rooms
OPTIONAL MATCH (h)-[n:NEAR]->(at:Attraction)
RETURN h.id AS hotel_id, h.name AS name, c.name AS city, h.area AS area,
       h.star_rating AS stars, h.guest_rating AS rating,
       h.check_in AS check_in, h.check_out AS check_out, h.children_policy AS children,
       amenities, policies, rooms,
       collect(DISTINCT {name: at.name, category: at.category, km: n.distance_km}) AS nearby,
       d.text AS description
"""


class GraphRAG:
    def __init__(self):
        # 60 s timeout: never hang forever waiting for the API
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=60_000),
        )
        self.router_model = os.environ.get("ROUTER_MODEL") or os.environ["GEMINI_MODEL"]
        self.answer_model = (
            os.environ.get("ANSWER_MODEL")
            or os.environ.get("EXTRACTION_MODEL")
            or os.environ["GEMINI_MODEL"]
        )
        self.db = os.environ.get("NEO4J_DATABASE")
        self.driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
            notifications_min_severity=NotificationMinimumSeverity.OFF,
        )
        self.values = self._load_values()

    def close(self):
        self.driver.close()

    # ---------- public ----------

    def ask(self, question: str, answer: bool = True, complete: bool = False) -> dict:
        start = time.time()
        decision = self._route(question)
        route = decision.route.strip().lower()
        if route not in ROUTE_NOTES:
            route = "vector"

        result = {
            "question": question,
            "route": route,
            "reasoning": decision.reasoning,
            "cypher": decision.cypher if route != "vector" else None,
            "search_text": decision.search_text,
            "fallback": None,
            "query_rows": [],
            "hotel_ids": [],
        }

        if route == "graph":
            rows, error = self._run_generated(decision.cypher)
            if error or not rows:
                result["fallback"] = error or "The query returned no hotels; used similarity search instead."
                route = result["route"] = "vector"
                result["hotel_ids"] = [h for h, _ in self._vector_search(question, VECTOR_TOP_K)]
            else:
                result["query_rows"] = rows
                result["hotel_ids"] = _unique(r["hotel_id"] for r in rows)

        elif route == "both":
            candidates = [h for h, _ in self._vector_search(decision.search_text or question, VECTOR_CANDIDATES_BOTH)]
            rows, error = self._run_generated(decision.cypher)
            if error:
                result["fallback"] = error + " Used similarity search only."
                route = result["route"] = "vector"
                result["hotel_ids"] = candidates[:VECTOR_TOP_K]
            else:
                allowed = _unique(r["hotel_id"] for r in rows)
                # keep similarity order for hotels that meet the conditions
                result["hotel_ids"] = [h for h in candidates if h in allowed] + [h for h in allowed if h not in candidates]
                result["query_rows"] = rows

        else:  # vector
            result["hotel_ids"] = [h for h, _ in self._vector_search(decision.search_text or question, VECTOR_TOP_K)]

        result["hotel_ids"] = result["hotel_ids"][:MAX_HOTELS]
        hotels = self._hotel_facts(result["hotel_ids"])
        result["hotels"] = [h["name"] for h in hotels]

        if answer:
            # In the "both" route the query is only a filter; its rows are not the answer,
            # so only the ranked hotels are passed on (otherwise the model counts filter rows).
            rows_for_answer = result["query_rows"] if route == "graph" else []
            result["answer"] = self._answer(question, route, rows_for_answer, hotels, complete)
        result["seconds"] = round(time.time() - start, 1)
        return result

    def vector_rag(self, question: str, complete: bool = False) -> dict:
        """Baseline for evaluation: plain vector RAG with no router and no graph query."""
        start = time.time()
        ids = [h for h, _ in self._vector_search(question, VECTOR_TOP_K)]
        hotels = self._hotel_facts(ids)
        return {
            "question": question,
            "route": None,
            "cypher": None,
            "fallback": None,
            "hotels": [h["name"] for h in hotels],
            "answer": self._answer(question, "vector", [], hotels, complete),
            "seconds": round(time.time() - start, 1),
        }

    # ---------- steps ----------

    def _route(self, question):
        prompt = ROUTER_PROMPT.format(schema=SCHEMA, question=question, **self.values)
        return self._generate(self.router_model, prompt, schema=RouteDecision)

    def _run_generated(self, cypher):
        """Runs LLM-written Cypher safely. Returns (rows, error_message)."""
        if not cypher:
            return [], "No Cypher query was generated."
        if FORBIDDEN.search(cypher):
            return [], "Generated query was rejected because it was not read-only."
        if "hotel_id" not in cypher:
            return [], "Generated query did not return hotel_id."
        try:
            with self.driver.session(database=self.db, default_access_mode=READ_ACCESS) as s:
                rows = s.execute_read(lambda tx: [r.data() for r in tx.run(cypher)])
            return rows, None
        except Exception as e:
            return [], f"Generated query failed ({e.__class__.__name__})."

    def _vector_search(self, text, k):
        vector = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        ).embeddings[0].values
        records, _, _ = self.driver.execute_query(
            """
            CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node, score
            MATCH (node)-[:DESCRIBES]->(h:Hotel)
            RETURN h.id AS hotel_id, score ORDER BY score DESC
            """,
            index=VECTOR_INDEX, k=k, vector=vector, database_=self.db,
        )
        return [(r["hotel_id"], r["score"]) for r in records]

    def _hotel_facts(self, ids):
        if not ids:
            return []
        records, _, _ = self.driver.execute_query(HOTEL_FACTS, ids=ids, database_=self.db)
        by_id = {r["hotel_id"]: r.data() for r in records}
        return [by_id[i] for i in ids if i in by_id]

    def _answer(self, question, route, rows, hotels, complete=False):
        query_rows = ""
        if rows:
            query_rows = "DATABASE QUERY RESULT:\n" + "\n".join(str(r) for r in rows[:20]) + "\n\n"
        hotel_text = "\n\n".join(_format_hotel(h) for h in hotels) or "No hotels matched."
        prompt = ANSWER_PROMPT.format(
            note=ROUTE_NOTES[route], question=question, query_rows=query_rows, hotels=hotel_text,
            selection_rule=SELECTION_COMPLETE if complete else SELECTION_SPOKEN,
        )
        return self._generate(self.answer_model, prompt, temperature=0.3).strip()

    # ---------- helpers ----------

    def _load_values(self):
        def names(query):
            records, _, _ = self.driver.execute_query(query, database_=self.db)
            return ", ".join(sorted(r["v"] for r in records if r["v"]))
        return {
            "cities": names("MATCH (c:City) RETURN c.name AS v"),
            "policies": names("MATCH (p:Policy) RETURN p.name AS v"),
            "amenities": names("MATCH (a:Amenity) RETURN a.name AS v"),
            "categories": names("MATCH (a:Attraction) RETURN DISTINCT a.category AS v"),
            "attractions": names("MATCH (a:Attraction) RETURN a.name AS v"),
        }

    def _generate(self, model, prompt, schema=None, temperature=0.0):
        # Low thinking keeps latency down; routing and short answers don't need deep reasoning.
        kwargs = {
            "temperature": temperature,
            "thinking_config": types.ThinkingConfig(thinking_level=os.environ.get("THINKING_LEVEL", "LOW")),
        }
        if schema:
            kwargs.update(response_mime_type="application/json", response_schema=schema)
        config = types.GenerateContentConfig(**kwargs)
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(model=model, contents=prompt, config=config)
                return resp.parsed if schema else resp.text
            except Exception as e:
                msg = str(e)
                if "429" in msg and "PerDay" in msg:
                    raise QuotaExceeded(f"Daily free-tier limit reached for {model}.") from e
                if "404" in msg or "400" in msg:
                    raise
                time.sleep(30 if "429" in msg else 5 * (attempt + 1))
        raise RuntimeError(f"{model} failed after retries")


def _unique(items):
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _format_hotel(h):
    lines = [f"{h['name']} ({h['area']}, {h['city']}): {h['stars']}-star, guest rating {h['rating']}."]
    if h.get("rooms"):
        lines.append("Rooms: " + "; ".join(
            f"{r['name']} ₹{r['price']}/night up to {r['guests']} guests" for r in h["rooms"] if r.get("name")))
    if h.get("amenities"):
        lines.append("Amenities: " + ", ".join(h["amenities"]))
    pols = []
    for p in h.get("policies") or []:
        if not p.get("name"):
            continue
        d = {k: v for k, v in (p.get("details") or {}).items() if v not in (None, "")}
        pols.append(p["name"] + (f" ({', '.join(f'{k}: {v}' for k, v in d.items())})" if d else ""))
    if pols:
        lines.append("Policies: " + "; ".join(pols))
    if h.get("check_in"):
        lines.append(f"Check-in {h['check_in']}, check-out {h['check_out']}. Children: {h.get('children')}")
    near = [f"{n['name']} ({n['category']}, {n['km']} km)" for n in h.get("nearby") or [] if n.get("name")]
    if near:
        lines.append("Nearby: " + "; ".join(near))
    if h.get("description"):
        about = h["description"].split("## About")[-1].split("## Rooms")[0].strip()
        lines.append("Description: " + about)
    return "\n".join(lines)
