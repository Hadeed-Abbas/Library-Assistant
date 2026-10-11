"""Build/update Willow's local vector index. Run from the repository root."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from retrieval import build_index  # noqa: E402

if __name__ == "__main__":
    result = build_index(ROOT / "documents", ROOT / "backend" / "rag_index")
    print("Index built successfully")
    for key, value in result.items():
        print(f"{key}: {value}")
