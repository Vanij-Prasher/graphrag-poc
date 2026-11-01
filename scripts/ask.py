"""Ask the hotel assistant a question from the terminal.

Run from the repo root:
    python scripts/ask.py "Which pet-friendly hotels are within 2 km of a fort?"
    python scripts/ask.py --no-answer "..."   # routing + retrieval only (saves one LLM call)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.rag import GraphRAG, QuotaExceeded  # noqa: E402

no_answer = "--no-answer" in sys.argv
question = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
if not question:
    sys.exit('Usage: python scripts/ask.py "your question"')

rag = GraphRAG()
try:
    r = rag.ask(question, answer=not no_answer)
except QuotaExceeded as e:
    sys.exit(f"Stopped: {e} Try again after the daily reset, or set ROUTER_MODEL / ANSWER_MODEL in .env.")
finally:
    rag.close()

print(f"\nQuestion : {r['question']}")
print(f"Route    : {r['route'].upper()}  ({r['reasoning']})")
if r["fallback"]:
    print(f"Fallback : {r['fallback']}")
if r["search_text"] and r["route"] in ("vector", "both"):
    print(f"Searched : \"{r['search_text']}\"")
if r["cypher"]:
    print("Cypher   :")
    for line in r["cypher"].strip().splitlines():
        print(f"    {line}")
print(f"Hotels   : {', '.join(r['hotels']) or 'none'}")
if "answer" in r:
    print(f"\nAnswer   : {r['answer']}")
print(f"\n({r['seconds']}s)")
