"""Searches hotel documents by meaning, using the vector index.

Run from the repo root:
    python scripts/vector_search.py "a quiet place for a family trip in the mountains"
"""
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from neo4j import GraphDatabase

load_dotenv()

question = " ".join(sys.argv[1:]) or "a quiet place for a family trip in the mountains"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
query_vector = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
).embeddings[0].values

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
records, _, _ = driver.execute_query(
    """
    CALL db.index.vector.queryNodes('document_embeddings', 5, $vector)
    YIELD node, score
    MATCH (node)-[:DESCRIBES]->(h:Hotel)-[:LOCATED_IN]->(c:City)
    RETURN h.name AS hotel, c.name AS city, round(score, 3) AS score
    ORDER BY score DESC
    """,
    vector=query_vector,
    database_=os.environ.get("NEO4J_DATABASE"),
)
driver.close()

print(f'Question: "{question}"\n')
for i, r in enumerate(records, 1):
    print(f"{i}. {r['hotel']} ({r['city']})  similarity {r['score']}")
