# Fairview Public Library "Willow" Chatbot

A fully local, CPU-run conversational assistant for the **Library Assistant** business case. Patrons can search the catalogue, ask about borrowing policy, and simulate renewals  entirely through prompt orchestration and conversational memory, with **no RAG and no tools/agents**, per the assignment constraint.

---

## 1. Business Use Case

Willow is the virtual assistant for a fictional public library, Fairview Public Library. It helps patrons:

- Search the catalogue by title, author, or genre and check availability
- Understand borrowing policy: loan periods, renewals, overdue fines, holds
- Simulate checking or renewing a due date
- Get general library-operations info (hours, how holds work, card requirements)

It explicitly refuses anything outside library operations  general knowledge questions, other business domains, casual chit-chat, or attempts to override its instructions  and redirects the patron back to what it can help with.

---

## 2. Conversation Flow Design

Willow moves through six stages, tracked implicitly through the conversation rather than as a rigid state machine:

1. **Greeting**  welcomes the patron, invites their request
2. **Intent Identification**  determines what they want: search, policy question, renewal, due date, or hold
3. **Catalogue or Policy Lookup**  answers using the data embedded in the system prompt
4. **Confirmation**  confirms any action taken (e.g. "Renewed  new due date is...")
5. **Off-topic Redirect**  can interrupt any stage
6. **Closing**  friendly sign-off

**Handling topic changes mid-conversation:** if a patron goes off-topic and gets redirected, and then returns to their earlier library-related request, Willow resumes that request using the context already in the conversation  it does not restart the conversation or ask the patron to repeat themselves. This was explicitly instructed in the system prompt and validated in manual testing (see Example Dialogue 2 below).

---


## 3. Architecture

```
 ┌────────────────┐        WebSocket (JSON)         ┌──────────────────┐
 │  frontend/       │  ────────────────────────────▶ │  backend/main.py  │
 │  index.html      │  ◀────────────────────────────  │  (FastAPI)         │
 │  (vanilla JS)    │        streamed tokens          └─────────┬─────────┘
 └────────┬─────────┘                                             │
          │  REST: GET /sessions, GET /sessions/{id}               │
          └─────────────────────────────────────────────────────┘
                                                                    │
                                                    ┌───────────────▼────────────────┐
                                                    │ backend/conversation_manager.py │
                                                    │  - regex guard (pre-LLM)         │
                                                    │  - session + sliding-window       │
                                                    │    history                        │
                                                    │  - saves transcripts to           │
                                                    │    backend/sessions/*.json        │
                                                    └───────────────┬────────────────┘
                                                                    │ HTTP (streaming)
                                                    ┌───────────────▼────────────────┐
                                                    │        Ollama (local)            │
                                                    │  qwen2.5:1.5b-instruct, Q4_K_M    │
                                                    └───────────────────────────────────┘
```

No cloud model APIs, no RAG, no external tools  the LLM Engine is 100% local inference via Ollama, and all "intelligence" comes from the system prompt (persona, policy, embedded catalogue) plus the deterministic guard layer.

---

## 4. Model Selection

**Chosen model: Qwen2.5-1.5B-Instruct, Q4_K_M quantization, served via Ollama**

**Hardware:** HP ProBook 650 G1  Intel i5 3rd gen (Ivy Bridge, no AVX2), 8GB RAM, no GPU.

**Why this model:**
- 3rd-gen Intel CPUs lack AVX2 (introduced in 4th-gen/Haswell), which most llama.cpp/Ollama performance assumptions are built around. Ollama was chosen specifically because it auto-detects CPU features and doesn't require hand-compiling with the right instruction-set flags.
- A 3–4B model was benchmarked as a mental estimate before committing: generation cost scales roughly with parameter count, so a 4B model would be ~2.5–3x slower per token than 1.5B  projected to ~5–7 tok/s, versus the actual ~13–16 tok/s measured with 1.5B. For a WebSocket streaming demo, that's the difference between feeling responsive and feeling stalled.
- The domain (library operations) is narrow and rule-based  well suited to a smaller model with a tight system prompt, rather than needing the broader reasoning capacity of a larger model.
- Q4_K_M was chosen over cruder quantizations (e.g. Q4_0) as the standard "good quality, small size" balance.

---

## 5. Context Memory Management Scheme

**Approach: fixed sliding window of the last 6 exchanges (12 messages)**, no summarization.

**Rationale:** summarization would require an extra LLM call per turn to compress history, which on this hardware would roughly double per-turn latency. Given the domain doesn't require long-term memory beyond the recent conversation (a patron's book search or policy question rarely needs context from 20 turns ago), a flat sliding window is simpler, cheaper, and predictable  a deliberate trade-off of long-term recall for latency, appropriate to the hardware constraint.

The system prompt (persona, policy, catalogue) is **not** part of this window  it's sent in full on every request, since it's static and doesn't grow with the conversation.

---

## 6. Latency Benchmarks

Measured on the HP ProBook 650 G1 (i5 3rd gen, 8GB RAM, no GPU) via Ollama's `--verbose` output.

| Scenario | Prompt size | Time to First Token | Sustained generation |
|---|---|---|---|
| Short prompt ("hi"), warm cache | ~30 tokens | ~0.08s | ~16.0 tok/s |
| Longer prompt, 613-token generation | ~72 tokens | ~1.15s | ~13.4 tok/s |
| Cold start after a Modelfile rebuild (full ~2,300-token system prompt evaluated fresh) | ~2,300 tokens | up to ~58s | ~9–11 tok/s |

**Key finding:** the system prompt (persona + policy + catalogue, ~2,000+ tokens) dominates latency on the *first* message of a session, since it must be evaluated in full. Ollama's prompt caching means subsequent turns in the same session reuse that cached prefix and only pay for the new message, bringing per-turn latency down to ~1–5 seconds depending on response length. This is worth knowing for the demo: the first message after a fresh model load will feel slow; everything after is much faster.

**4B model, for comparison:** not directly benchmarked, but based on the ~2.5–3x compute scaling observed in early estimation, projected at roughly 5–7 tok/s sustained generation  the 1.5B model was kept as the better fit for this hardware and domain (see Section 5).

---

## 7. Known Limitations

These were discovered through deliberate adversarial testing (Phase VI) and are documented honestly rather than glossed over, along with the mitigations attempted for each.

### 7.1 Jailbreak resistance doesn't generalize from prompt engineering alone
Initial testing with only a system-prompt instruction against "ignore your instructions"-style attacks failed completely  the model complied with jailbreak attempts and answered off-domain questions. Adding an explicit anti-override section plus few-shot examples (both as prose and as real `MESSAGE` chat-turns in the Modelfile) fixed refusal for the **exact phrasings** shown in the examples, but a **close paraphrase** ("ignore the previous instructions" vs. the trained "ignore your previous instructions... instead") still got through. This shows the 1.5B model is pattern-matching specific phrasing rather than generalizing the underlying rule.

**Mitigation:** a deterministic regex-based guard (`conversation_manager.is_blocked`) intercepts messages **before** they reach the LLM, checking for the *concept* of an override/off-topic attempt rather than exact phrasing. This was tested against both the original attack and the paraphrase that beat the model, and correctly blocks both while not falsely blocking genuine library questions (validated with a manual test suite  see inline comments in `conversation_manager.py`).

### 7.2 Catalogue hallucination
Even after compacting the catalogue format, repositioning it for recency (placing it last in the prompt, closest to generation), and adding explicit "do not invent titles" rules with a named example, the model still occasionally **invented catalogue entries that don't exist** (e.g. fabricating Harry Potter sequels not present in the actual 28-book catalogue) and contradicted its own stated availability numbers.

This is treated as a genuine capacity limitation: faithfully performing exact lookups across ~140 structured data points (28 books × 5 fields) in-context is a fundamentally different  and harder  task for a 1.5B model than fluent text generation. No further prompt-engineering mitigation fully resolved this within the scope of this assignment; it's documented here as an accepted limitation of small local models for this kind of task, rather than claimed to be solved.

### 7.3 Repetition anchoring
Early testing showed the model would sometimes get "stuck" repeating its last refusal response verbatim regardless of the new question asked, traced to the exact refusal sentence appearing multiple times in context (few-shot examples + stored conversation history). **Mitigated** by removing redundant prose examples once the guard layer took over jailbreak enforcement, and by storing a neutral placeholder (`"[off-topic request declined]"`) in session history instead of the literal refusal sentence, so it doesn't compound across turns.

---

## 8. API Documentation

### REST
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check, returns `{"status": "ok"}` |
| `/sessions` | GET | List saved past conversations (metadata only), newest first |
| `/sessions/{session_id}` | GET | Full transcript of one saved conversation |

### WebSocket  `/ws/chat`
JSON messages both directions.

**Client → Server:**
```json
{"message": "do you have the hobbit?"}
{"type": "reset"}
```

**Server → Client:**
```json
{"type": "session", "session_id": "..."}   // sent on connect and after a reset
{"type": "token", "content": "..."}          // one per streamed token
{"type": "done"}                             // end of a response
{"type": "error", "message": "..."}          // malformed request or model error;
                                              // connection stays open, not closed
```

Each WebSocket connection is its own async session with its own history  concurrent users do not block each other (see `backend/main.py`).

---

## 9. Setup Instructions

**Prerequisites:** [Ollama](https://ollama.com) installed, Python 3.10+.

```bash
# 1. Pull the base model and build the custom persona
ollama pull qwen2.5:1.5b-instruct
ollama create A1 -f Modelfile

# 2. Backend
cd backend
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Frontend
# Just open frontend/index.html directly in a browser  no build step needed.
```

Confirm the backend is up at `http://localhost:8000/health`.

---

## 10. Bonus: UX Polish (chosen bonus category)

Between the two bonus options, **UX/persona polish** was chosen over cloud deployment after evaluating hosting feasibility:

- **Vercel** only supports Python as short-timeout serverless functions (~10s), incompatible with a long-lived streaming WebSocket connection, and cannot host Ollama or model weights at all.
- **Render/Railway** free tiers provide a real persistent process (which would work for the WebSocket server itself), but free-tier RAM (~512MB) is well below what's needed to load even a Q4-quantized 1.5B model (~1–2GB).
- Given the assignment's core requirement is fully local CPU inference with no cloud model APIs, a genuine zero-cost cloud deployment of the *working system* isn't achievable  this is stated plainly rather than worked around with a substitute cloud model, which would violate the assignment's local-inference requirement.

**What was built instead, beyond Phase V's requirements:**
- **Persistent conversation history sidebar**  every conversation is saved to disk (`backend/sessions/*.json`) and browsable, not just visible within a single live session
- **Read-only history viewing** with a clear "viewing history" state, distinct from live chat, so past and present conversations are never confused
- **A deliberate visual identity** grounded in the subject matter (library card-catalog aesthetic: warm ivory background, deep library green and brass accents, serif headers/assistant text vs. sans UI chrome, ruled dividers instead of generic chat bubbles) rather than a default chat-widget look
- **Defense-in-depth persona robustness**: a validated, tested guard layer that catches jailbreak/off-topic attempts the model itself doesn't reliably resist on paraphrased input (see Section 8.1)  arguably the more meaningful form of "staying in character under adversarial testing" than prompt engineering alone could achieve
- Mobile-responsive sidebar (collapses behind a toggle under 720px width)