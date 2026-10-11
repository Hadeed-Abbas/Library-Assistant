# Assignment 2 test cases

Run the tests after building the index and starting Ollama/backend. Record actual outputs and timings for the demo; do not claim a test passed unless you ran it.

| ID | User query / action | Expected behaviour |
|---|---|---|
| RAG-01 | “How long can I borrow a normal book?” | Answer says 21 calendar days and cites the standard loan-period policy. |
| RAG-02 | “Can I renew a book if another patron has a hold?” | Answer says a hold prevents renewal and cites the renewal rules. |
| RAG-03 | “What is the overdue fine?” | Answer gives PKR 20 per item per calendar day, capped at PKR 600, and cites the overdue-fines policy. |
| RAG-04 | “Is The Hobbit available?” | Answer uses the catalogue snapshot and cites The Hobbit record; should say 1 of 4 copies is available. |
| RAG-05 | “What time does the library close on Saturday?” | Answer says 4:00 PM and cites opening hours. |
| RAG-06 | “How do I place a hold?” | Answer explains hold eligibility and the 3-calendar-day pickup window with a source citation. |
| RAG-07 | “What is the weather in Islamabad?” | No weather facts should be invented; politely redirects to library assistance. |
| RAG-08 | “Tell me the library's exact policy on a topic not in the documents.” | Says no reliable matching document was found and recommends staff rather than inventing policy. |
| RAG-09 | Ask a catalogue question, then a follow-up using “it” or the book title | Recent conversation context is retained and answer remains grounded in retrieved records. |
| RAG-10 | Repeat the same question | Warm query-result cache should be used; benchmark should show a lower/near-zero retrieval-only duration for the cached result. |
| RAG-11 | Temporarily rename `backend/rag_index/` and ask a question | Chat should not crash; backend logs retrieval failure and the assistant must not invent specific library facts. Restore the folder afterward. |
| RAG-12 | Open two browser tabs and ask questions in both | Both WebSocket sessions should remain responsive and stream token events. |
| RAG-13 | Send malformed JSON / missing `message` through a WebSocket test client | Backend sends an `error` event and leaves the socket open, preserving Assignment 1 behaviour. |

The NumPy-based unit tests can be run without downloading the embedding model:

```powershell
python -m unittest discover -s tests -v
```
