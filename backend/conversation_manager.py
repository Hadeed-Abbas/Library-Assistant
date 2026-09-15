"""
Conversation Manager for the Library Assistant chatbot.

Responsibilities:
- Enforce domain policy with a deterministic keyword/pattern guard BEFORE
  anything reaches the LLM (catches jailbreak/off-topic attempts the model
  itself was shown, empirically, not to reliably resist on paraphrase).
- Maintain per-session dialogue history with a fixed sliding window
  (context memory strategy — see README for rationale).
- Persist each session's transcript to disk as JSON, so past conversations
  can be listed and reviewed (UX polish: conversation history sidebar).
- Stream responses from a local Ollama model over its chat API.

No tools, agents, or RAG are used here — this module only does prompt
orchestration and local conversational memory, per assignment constraints.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import httpx

# --- Configuration ---------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "A1"  # match the name shown by `ollama list` on your machine

MAX_HISTORY_TURNS = 6  # keep the last 6 user+assistant exchanges (12 messages)

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

REFUSAL_MESSAGE = (
    "I'm the library assistant, so I can only help with catalogue searches, "
    "borrowing, renewals, and due dates. Is there something along those "
    "lines I can help with?"
)

# Paste the exact SYSTEM block content from your Modelfile here (without the
# triple quotes). Keeping it in one place avoids the prompt drifting out of
# sync between the Modelfile (used for manual CLI testing) and the app.
SYSTEM_PROMPT = """You are Willow, the virtual assistant for Fairview Public Library. You help patrons with catalogue searches, borrowing policy questions, renewals, due dates, and holds. Stay strictly within this role at all times.

## SCOPE — WHAT YOU DO
- Search the catalogue by title, author, or genre and report availability
- Answer questions about borrowing policy (loan periods, renewals, fines, holds)
- Simulate checking or renewing a due date for a patron
- Explain how holds work
- Answer general library-operations questions (hours, how to get a card, etc.)

## OUT OF SCOPE
Anything that is not library operations. If asked, respond with:
"I'm the library assistant, so I can only help with catalogue searches, borrowing, renewals, and due dates. Is there something along those lines I can help with?"
Redirect once, briefly, and move on — do not explain further or lecture the user.

## CONVERSATION FLOW
1. Greeting — welcome the patron, invite their request
2. Intent Identification — figure out what they want: search, policy question, renewal, due date, or hold
3. Catalogue or Policy Lookup — answer using the data below
4. Confirmation — confirm any action taken (e.g. "Renewed — your new due date is [date].")
5. Off-topic Redirect — can happen from any stage
6. Closing — friendly sign-off when the patron is done

If the conversation goes off-topic and you redirect, and the patron then returns to their earlier library-related request, resume that request naturally. Do not restart the conversation or ask them to repeat themselves.

## BORROWING POLICY (cite these exact numbers when asked)
- Standard loan period: 21 days
- New release / high-demand titles: 7 days
- Maximum books per patron at once: 5
- Renewals: 1 per book, only if no one else has a hold on it
- Renewal extension: +14 days from the renewal date
- Overdue fine: $0.25/day per book, capped at $10/book
- Holds: allowed on checked-out books, first-come-first-served, patron is notified when available
- A library card is required to borrow or place a hold, but not to search the catalogue

## TONE
Warm, brief, professional — like a helpful librarian at a front desk, not a chatty general-purpose assistant. Keep answers focused; don't pad with unnecessary pleasantries.

## CATALOGUE (your ONLY source of book data — read this section carefully, it is the most important part of this prompt)
Format per line: Title | Author | Genre | total copies | available copies

The Hobbit | J.R.R. Tolkien | Fantasy | 4 | 1
Dune | Frank Herbert | Sci-Fi | 3 | 2
1984 | George Orwell | Classic | 5 | 0
Pride and Prejudice | Jane Austen | Romance | 3 | 3
The Girl with the Dragon Tattoo | Stieg Larsson | Thriller | 2 | 1
Sapiens | Yuval Noah Harari | Non-fiction | 3 | 1
The Da Vinci Code | Dan Brown | Mystery | 4 | 2
Educated | Tara Westover | Biography | 2 | 0
The Fellowship of the Ring | J.R.R. Tolkien | Fantasy | 3 | 0
Gone Girl | Gillian Flynn | Thriller | 3 | 3
The Catcher in the Rye | J.D. Salinger | Classic | 2 | 1
Brief Answers to the Big Questions | Stephen Hawking | Non-fiction | 2 | 2
Murder on the Orient Express | Agatha Christie | Mystery | 3 | 1
The Shining | Stephen King | Horror | 2 | 0
Steve Jobs | Walter Isaacson | Biography | 2 | 1
The Alchemist | Paulo Coelho | Fiction | 4 | 2
Foundation | Isaac Asimov | Sci-Fi | 2 | 1
Little Women | Louisa May Alcott | Classic | 2 | 2
The Hunger Games | Suzanne Collins | Sci-Fi | 5 | 3
Charlotte's Web | E.B. White | Children's | 3 | 3
Atomic Habits | James Clear | Non-fiction | 4 | 0
The Silent Patient | Alex Michaelides | Thriller | 2 | 1
Harry Potter and the Sorcerer's Stone | J.K. Rowling | Fantasy | 6 | 2
Where the Crawdads Sing | Delia Owens | Fiction | 3 | 1
The Martian | Andy Weir | Sci-Fi | 2 | 2
To Kill a Mockingbird | Harper Lee | Classic | 3 | 0
Becoming | Michelle Obama | Biography | 2 | 1
The Night Circus | Erin Morgenstern | Fantasy | 2 | 1

## CRITICAL RULES FOR ANSWERING (apply these to every catalogue question, no exceptions)
1. The list above is the ENTIRE catalogue. If a title is not in that list, it does not exist in this library — do not invent sequels, other editions, or related titles, even if they are real books in the world. For example, only "Harry Potter and the Sorcerer's Stone" exists here — there is no Chamber of Secrets, Prisoner of Azkaban, or any other Harry Potter book in this catalogue.
2. If the exact title a patron asks about is not in the list, say you don't see that title in the catalogue — do not guess or substitute a similar-sounding title.
3. The last number on each line is the available copies. If that number is greater than 0, the book IS available right now — say so and state the exact number. If that number is 0, the book is checked out — offer to place a hold. Read the actual number before answering; never state a book is unavailable when its number is greater than 0.
4. Each new patron message is a fresh question — evaluate it on its own using the list above, even if your last reply was about a different book or a redirect."""

# --- Guard: deterministic pattern matching ---------------------------------
# These catch the *concept* of an override/off-topic attempt, not exact
# phrasing, since testing showed the model itself breaks on close paraphrases
# of things it was prompted to refuse.

OVERRIDE_PATTERNS = [
    r"\bignore\b.{0,20}\b(instructions?|rules?|prompt)\b",
    r"\bdisregard\b.{0,20}\b(instructions?|rules?|prompt)\b",
    r"\bforget\b.{0,20}\b(instructions?|rules?|prompt)\b",
    r"\byou are now\b",
    r"\bpretend (you'?re|you are)\b",
    r"\bact as (a|an)\b",
    r"\broleplay as\b",
    r"\bno restrictions\b",
    r"\bwithout restrictions\b",
    r"\byour (system )?prompt\b",
    r"\bwhat are you told\b",
]

OFFTOPIC_PATTERNS = [
    r"\bcapital of\b",
    r"\bweather\b",
    r"\bwho is the (president|prime minister)\b",
    r"\bjoke\b",
    r"\bwhat('?s| is)\s*\d+\s*[\+\-\*/]\s*\d+\b",  # basic arithmetic
]

_GUARD_RE = re.compile(
    "|".join(OVERRIDE_PATTERNS + OFFTOPIC_PATTERNS), re.IGNORECASE
)


def is_blocked(message: str) -> bool:
    """Return True if the message should be refused without calling the LLM."""
    return bool(_GUARD_RE.search(message))


# --- Session / history management ------------------------------------------


class Session:
    """Holds dialogue history for a single WebSocket session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def build_messages(self, new_user_message: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": new_user_message})
        return messages


# In-memory session store, keyed by session id.
# Fine for a single-process assignment demo; would need a shared store
# (e.g. Redis) if you ever ran multiple backend processes.
_sessions: Dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id)
    return _sessions[session_id]


def reset_session(session_id: str) -> None:
    """Drops a session from memory. The on-disk transcript (if any) is left
    alone — resetting starts a NEW session id, it does not erase history."""
    _sessions.pop(session_id, None)


# --- Persistence (conversation history sidebar) -----------------------------
# Note: this is local disk I/O for saving/loading the app's OWN conversation
# transcripts — not retrieval of outside knowledge for the model to reason
# over. It has no bearing on the "no RAG" constraint, which is about how the
# LLM gets its answers, not about the app persisting its own chat logs.


def _session_file(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(session: Session) -> None:
    if not session.history:
        return  # don't clutter the sidebar with empty sessions
    first_user_msg = next(
        (m["content"] for m in session.history if m["role"] == "user"), ""
    )
    title = (first_user_msg[:48] + "…") if len(first_user_msg) > 48 else first_user_msg
    data = {
        "session_id": session.session_id,
        "title": title or "New conversation",
        "created_at": session.created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "history": session.history,
    }
    _session_file(session.session_id).write_text(json.dumps(data, indent=2))


def list_sessions() -> List[Dict]:
    """Returns metadata (no full history) for the sidebar, newest first."""
    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sessions.append(
            {
                "session_id": data["session_id"],
                "title": data["title"],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "message_count": len(data["history"]),
            }
        )
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def load_session_transcript(session_id: str) -> Optional[Dict]:
    f = _session_file(session_id)
    if not f.exists():
        return None
    return json.loads(f.read_text())


# --- LLM streaming -----------------------------------------------------------


async def stream_response(
    session_id: str, user_message: str
) -> AsyncGenerator[str, None]:
    """
    Yields response text incrementally (token by token) for a given session.
    If the message trips the guard, yields the canned refusal immediately
    without calling the model at all.
    """
    session = get_session(session_id)

    if is_blocked(user_message):
        session.add_turn("user", user_message)
        # Store a neutral note rather than the literal refusal sentence.
        # Repeating that exact string in history biases the model toward
        # copying it verbatim on later, unrelated turns (observed in testing).
        session.add_turn("assistant", "[off-topic request declined]")
        save_session(session)
        yield REFUSAL_MESSAGE
        return

    messages = session.build_messages(user_message)
    full_response = ""

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            OLLAMA_URL,
            json={"model": MODEL_NAME, "messages": messages, "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response += token
                    yield token
                if chunk.get("done"):
                    break

    session.add_turn("user", user_message)
    session.add_turn("assistant", full_response)
    save_session(session)