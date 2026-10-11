"""Measure retrieval separately from Ollama generation, with cold and warm runs."""
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from retrieval import retrieve  # noqa: E402

QUERIES = [
    "How long can I keep a normal book?",
    "What is the daily overdue fine?",
    "Is The Hobbit available?",
    "How do I place a hold on a checked-out book?",
    "What time does the library close on Saturday?",
    "Can I renew a book if another patron has a hold?",
]

if __name__ == "__main__":
    first = []
    warm = []
    for query in QUERIES:
        result = retrieve(query)
        first.append(result["latency_ms"])
        print(f"{result['latency_ms']:8.2f} ms | {len(result['matches'])} matches | {query}")
        if result["error"]:
            print("  ERROR:", result["error"])
    for query in QUERIES:
        result = retrieve(query)
        warm.append(result["latency_ms"])
    print("\nRetrieval-only benchmark (generation excluded)")
    print(f"First run median: {statistics.median(first):.2f} ms")
    print(f"Warm cached median: {statistics.median(warm):.2f} ms")
    print("Record results from your own laptop in README; results vary by CPU and cache state.")
