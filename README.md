# Assignment 2 — Willow Library Assistant (Local RAG)

This project extends Assignment 1's **Fairview Public Library** assistant, Willow, with local Retrieval-Augmented Generation (RAG). It keeps the existing FastAPI backend, browser chat UI, conversation history, patron notes, local Ollama generation, and `/ws/chat` JSON streaming contract.

## What is included

- `documents/`: **60** realistic Markdown documents (30 catalogue records, 15 policy documents, 15 service guides).
- `scripts/build_index.py`: re-runnable document chunking, local embedding, and index creation.
- `backend/retrieval.py`: CPU embedding, persistent NumPy vector index, cosine ranking, relevance threshold, query-result cache, and source formatting.
- `backend/conversation_manager.py`: RAG integration with the existing streaming generation and memory flow.
- `backend/main.py`: existing REST and WebSocket API contract.
- `frontend/index.html`: existing chat interface; retrieved source citations appear at the end of grounded answers.
- `scripts/benchmark_retrieval.py`: measures retrieval separately from model generation.
- `tests/test_retrieval.py`: unit tests for chunking, ranking, context bounds, and source de-duplication.

All library names, policies, prices, schedules, and contact details in `documents/` are fictional assignment data. They are not live library information.

## Requirements and hardware notes

- Windows 10/11, macOS, or Linux.
- Python **3.10 or 3.11** recommended.
- Ollama installed and running locally.
- The model used by the original project: `qwen2.5:1.5b-instruct`.
- Internet access is needed the first time to download the Python dependencies and the embedding model. After those are installed/cached and the index is built, document retrieval and model inference run locally. Do not delete the Hugging Face model cache if you need fully offline operation.
- On the HP ProBook 650 G1, use CPU mode and expect indexing to take a while. Build the index once; it is persisted on disk. No GPU is required.

## Quick start (Windows PowerShell)

Run commands from the extracted `Assignment2` project folder.

### 1. Start Ollama and prepare the local chat model

If you already have the model pulled, you can skip the pull command.

```powershell
ollama pull qwen2.5:1.5b-instruct
ollama create A1 -f Modelfile
ollama list
```

The provided `Modelfile` uses the same base model and adds Willow's system instructions. **Important:** the uploaded Assignment 1 archive's file named `Modelfile` contained README prose rather than Ollama `FROM`/`SYSTEM` directives, so this package replaces that malformed file with a valid Modelfile. Your base model remains Qwen2.5 1.5B Instruct.

Keep Ollama running in the background. The backend calls `http://localhost:11434/api/chat` using model name `A1`.

### 2. Create a Python environment and install dependencies

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

If PowerShell blocks activation, use this command for the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Build the local document index (first run, and after editing documents)

```powershell
python scripts\build_index.py
```

The script reads supported `.md`, `.txt`, and `.html` files in `documents/`, splits them into overlapping chunks, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and writes the index to `backend/rag_index/`.

- Chunk size: **180 words**.
- Overlap: **35 words**.
- Retrieval: cosine similarity over normalized vectors; **top-k = 4** (the code enforces a minimum k of 3).
- Relevance threshold: **0.30 cosine similarity**, configurable using `RAG_RELEVANCE_THRESHOLD`.
- Vector store: a small persistent NumPy matrix plus JSON chunk metadata. For this 60-document CPU assignment corpus, this is a simple local exact-search index and avoids a separate database service or compiled FAISS dependency.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, a compact sentence embedding model. The embedding model is used for retrieval only; Ollama/Qwen generates the answer.

The build command can be run again whenever documents change. It rebuilds the stored index from the current document directory rather than appending duplicate vectors. Restart the backend after rebuilding so it reloads the new index.

### 4. Start the backend

From the project root, with the virtual environment activated:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

For development auto-reload, use `python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000`. Keep this terminal open.

Check `http://127.0.0.1:8000/health`; it should return `{"status":"ok"}`.

### 5. Open the frontend

Open `frontend/index.html` in a browser. No JavaScript build step is required. Ask questions such as:

- `How long can I borrow a normal book?`
- `Can I renew a book if someone has placed a hold?`
- `Is The Hobbit available, and how many copies are there?`
- `What is the overdue fine?`
- `What time does the library close on Saturday?`
- `How do I book a study room?`

Answers grounded in retrieved documents end with a `Sources:` line naming the source documents. This citation line is appended by the application so it does not depend on the small model remembering to cite sources.

## Running tests and the retrieval benchmark

Run unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

Run retrieval-only performance measurements (the index must already exist):

```powershell
python scripts\benchmark_retrieval.py
```

The benchmark reports retrieval time separately from generation time, including first-run and repeated-query cache behaviour. **Do not copy an assumed latency into a submission.** Run it on your HP ProBook and record the observed median below; CPU load, whether the embedding model is already loaded, and disk/model cache state affect the result.

### Benchmark record (fill in with your actual run)

| Measurement | Your result |
|---|---|
| Hardware / Python version | HP ProBook 650 G1 / fill in Python version |
| First-run retrieval median | Not measured in the packaging environment; run the benchmark on the HP ProBook |
| Warm cached retrieval median | Not measured in the packaging environment; run the benchmark on the HP ProBook |
| Target | Retrieval alone under 1 second |

The first query can include model-loading overhead if the embedding model has not yet been loaded into the backend process. The script reports the measurements it observes; it does not measure Ollama generation.

## Architecture

```text
Browser frontend (frontend/index.html)
      |  WebSocket JSON: /ws/chat (same protocol as Assignment 1)
      v
FastAPI backend (backend/main.py)
      |
      v
Conversation manager (backend/conversation_manager.py)
      |                         |
      | retrieve query           | stream prompt + tokens
      v                         v
Retrieval module             Ollama local API
(backend/retrieval.py)        (Qwen2.5 1.5B, model A1)
      |
      +-- query embedding: all-MiniLM-L6-v2 (CPU)
      +-- NumPy cosine search over persistent vectors
      +-- top-k relevant chunks + source metadata
      +-- cached query results
      |
      v
Retrieved chunks are inserted into a bounded prompt with recent history
      |
      v
Streamed answer tokens + application-appended source citations
```

## Retrieval, prompt budget, and failure handling

1. Each user query is embedded locally and compared with the normalized document vectors.
2. The four highest-ranked chunks are considered; chunks below the relevance threshold are discarded. If none pass the threshold, the model receives no fabricated evidence and is instructed not to guess specific library facts.
3. The selected chunk text is capped at approximately 4,200 characters. The conversation manager applies a conservative 10,000-character input budget and removes the oldest history messages first, preserving the system instructions, available evidence, and newest user question. Ollama is configured with a 4,096-token context window and a 350-token output cap.
4. Retrieval runs via `asyncio.to_thread`, so embedding work does not synchronously block FastAPI's event loop. A bounded in-process LRU cache stores repeated query results; a warm cache hit avoids re-embedding the query.
5. If the index is missing, the embedding model fails to load, or retrieval raises an error, the chat continues without retrieved evidence. The error is logged on the backend, and the prompt tells Willow not to guess library-specific facts. Run the index build and verify dependencies to restore retrieval.
6. The existing WebSocket messages remain `session`, `token`, `done`, and `error`; no new protocol messages were introduced. Each socket still has its own session/history. The vector index is read-only during normal requests and shared safely between users in one backend process.

## API contract (unchanged)

- `GET /health` — liveness check.
- `GET /sessions` — saved session summaries.
- `GET /sessions/{session_id}` — saved transcript.
- `GET /memory` and `DELETE /memory` — view/clear demo patron notes.
- `WS /ws/chat` — accepts `{"message":"..."}` and `{"type":"reset"}`; emits the same `session`, `token`, `done`, and `error` event types as Assignment 1.

## Known limitations

- Qwen2.5 1.5B is a small model and may misread evidence or phrase an answer poorly. Retrieval improves grounding but does not guarantee perfect generation. Check cited source documents when accuracy matters.
- Similarity threshold `0.30` is a starting point, not a universal value. If unrelated questions retrieve library chunks, raise it; if valid questions often return no match, lower it slightly and re-test. Use the benchmark and example questions rather than tuning on one query only.
- The embedding model needs to be downloaded once. The index is local after creation, but deleting `backend/rag_index/` means it must be rebuilt.
- NumPy exact search is suitable for this small corpus. A much larger collection may need FAISS or another vector database.
- Patron long-term memory remains a single demo file from Assignment 1, not a multi-account production system. The assistant cannot read or update real patron accounts, place actual holds, check out books, or perform renewals.
- The corpus contains 60 fictional documents rather than live library records. Availability and policy answers reflect the included snapshot only.

## Viva preparation: be ready to explain

- Why RAG is needed: the model retrieves relevant document chunks instead of relying only on facts in its weights or a long hard-coded prompt.
- Why MiniLM: compact local embeddings are practical for a CPU-only laptop; Qwen is still used for answer generation.
- Why chunking and overlap: chunking keeps context manageable; overlap reduces the chance that a fact at a chunk boundary is separated from related text.
- Why cosine similarity and a threshold: rank semantic matches, then reject weak matches rather than forcing the model to answer from irrelevant documents.
- Why citations are appended in code: the source list remains visible even if the 1.5B model forgets to cite.
- Why retrieval latency is benchmarked separately: it distinguishes embedding/search cost from the much slower token generation time.
- Why prompt trimming is needed: system prompt, evidence, history, and output must fit within the local model's context window.

## Suggested test cases

See `test_cases.md` for expected behaviour across grounded questions, low-relevance questions, conversation follow-ups, and failure handling.
