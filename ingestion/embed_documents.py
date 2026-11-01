"""Embeds each hotel Document with Gemini Embedding and creates a vector index in Neo4j.

Run from the repo root:
    python ingestion/embed_documents.py            # embeds only documents without an embedding
    python ingestion/embed_documents.py --refresh  # re-embeds every document

Each hotel document is short, so the whole document is one chunk.
Real-world documents would be split into chunks first.
"""
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from neo4j import GraphDatabase

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
DIMENSIONS = 3072
INDEX_NAME = "document_embeddings"


def main():
    refresh = "--refresh" in sys.argv
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    db = os.environ.get("NEO4J_DATABASE")

    where = "" if refresh else "WHERE d.embedding IS NULL"
    docs, _, _ = driver.execute_query(
        f"MATCH (d:Document) {where} RETURN d.id AS id, d.text AS text ORDER BY d.id",
        database_=db,
    )
    print(f"{len(docs)} documents to embed with {EMBEDDING_MODEL}")

    if docs:
        # One request for all documents. RETRIEVAL_DOCUMENT tells the model these are
        # the texts to be searched (questions use RETRIEVAL_QUERY).
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[d["text"] for d in docs],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        rows = [{"id": d["id"], "embedding": e.values} for d, e in zip(docs, result.embeddings)]
        driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.id})
            CALL db.create.setNodeVectorProperty(d, 'embedding', row.embedding)
            """,
            rows=rows,
            database_=db,
        )
        print(f"Stored {len(rows)} embeddings ({len(rows[0]['embedding'])} dimensions each)")

    driver.execute_query(
        f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (d:Document) ON d.embedding
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {DIMENSIONS},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
        database_=db,
    )
    driver.execute_query("CALL db.awaitIndexes(300)", database_=db)

    records, _, _ = driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, state, populationPercent WHERE name = $name "
        "RETURN state, populationPercent",
        name=INDEX_NAME,
        database_=db,
    )
    r = records[0]
    print(f"Vector index '{INDEX_NAME}': {r['state']}, {r['populationPercent']}% populated")
    driver.close()


if __name__ == "__main__":
    main()
