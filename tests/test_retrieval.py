import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from retrieval import chunk_text, rank_vectors, format_context, format_sources

class RetrievalUnitTests(unittest.TestCase):
    def test_chunking_has_overlap_and_metadata(self):
        text = " ".join(f"word{i}" for i in range(20))
        chunks = chunk_text(text, "policies/policy.md", "Test policy", chunk_words=8, overlap_words=2)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].source, "policies/policy.md")
        self.assertEqual(chunks[0].title, "Test policy")
        self.assertEqual(chunks[0].text.split()[-2:], chunks[1].text.split()[:2])

    def test_rank_vectors_orders_by_cosine(self):
        embeddings = np.array([[1, 0], [0, 1], [0.8, 0.2]], dtype=np.float32)
        chunks = [
            {"chunk_id": "a", "source": "a.md", "title": "A", "text": "alpha"},
            {"chunk_id": "b", "source": "b.md", "title": "B", "text": "beta"},
            {"chunk_id": "c", "source": "c.md", "title": "C", "text": "gamma"},
        ]
        ranked = rank_vectors(np.array([1, 0]), embeddings, chunks, top_k=3)
        self.assertEqual([r.chunk_id for r in ranked], ["a", "c", "b"])
        self.assertGreaterEqual(ranked[0].score, ranked[1].score)

    def test_context_is_bounded_and_sources_deduplicated(self):
        matches = [
            {"title": "Loan policy", "source": "policy.md", "score": 0.7, "text": "x" * 300},
            {"title": "Loan policy", "source": "policy.md", "score": 0.6, "text": "y" * 300},
        ]
        self.assertLessEqual(len(format_context(matches, max_chars=250)), 250)
        self.assertEqual(format_sources(matches).count("Loan policy"), 1)

if __name__ == "__main__":
    unittest.main()
