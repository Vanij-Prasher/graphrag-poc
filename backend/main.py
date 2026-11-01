"""FastAPI backend for the GraphRAG hotel assistant.

Run locally from the repo root:
    uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000/docs to try the API in the browser.

Endpoints:
    GET  /health   checks the API and the Neo4j connection
    POST /ask      {"question": "..."} -> answer, route, reasoning, Cypher, hotels, timing
"""
import os
import threading
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.rag import GraphRAG, QuotaExceeded

load_dotenv()

app = FastAPI(
    title="GraphRAG Hotel Assistant",
    description="Hybrid knowledge-graph + vector RAG over a hotel dataset (Neo4j + Gemini).",
    version="0.1.0",
)

# Which websites may call this API from a browser. Comma-separated in .env;
# defaults to the local Next.js dev server.
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# One GraphRAG instance, created on first use and shared by all requests.
# Lazy creation (instead of at startup) also suits serverless hosting such as Vercel.
_rag: Optional[GraphRAG] = None
_lock = threading.Lock()


def get_rag() -> GraphRAG:
    global _rag
    if _rag is None:
        with _lock:
            if _rag is None:
                _rag = GraphRAG()
    return _rag


class AskRequest(BaseModel):
    # Length limits protect the LLM quota from very long or empty inputs.
    question: str = Field(..., min_length=3, max_length=300, examples=["Which pet-friendly hotels are within 2 km of a fort?"])


class AskResponse(BaseModel):
    question: str
    answer: str
    route: str
    reasoning: str
    cypher: Optional[str] = None
    search_text: Optional[str] = None
    fallback: Optional[str] = None
    hotels: list[str]
    seconds: float


@app.get("/health")
def health():
    try:
        get_rag().driver.verify_connectivity()
        return {"status": "ok", "neo4j": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e.__class__.__name__}")


# A plain "def" (not async): FastAPI runs it in a worker thread, so slow LLM calls
# don't block other requests.
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        r = get_rag().ask(req.question.strip())
    except QuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not answer: {e.__class__.__name__}")
    return AskResponse(
        question=r["question"],
        answer=r.get("answer", ""),
        route=r["route"],
        reasoning=r["reasoning"],
        cypher=r.get("cypher"),
        search_text=r.get("search_text"),
        fallback=r.get("fallback"),
        hotels=r.get("hotels", []),
        seconds=r["seconds"],
    )
