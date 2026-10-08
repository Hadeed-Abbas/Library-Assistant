"""
Conversation Manager for the Library Assistant chatbot.

All of the assistant's behaviour comes from prompt orchestration and
conversational memory only. There are no tools, agents, plugins, RAG, or
rule based filters in this module.

Memory has two layers:
- Short term: a sliding window of the last few exchanges of the current session.
- Long term: a short notes file about the patron. When a session ends, the
  model itself rewrites these notes from the conversation, and the notes are
  added to the system prompt of every later session.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import httpx

# --- Configuration ---------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "A1"  # must match the name shown by `ollama list`

MAX_HISTORY_TURNS = 6       # last 6 exchanges (12 messages) are sent to the model
MEMORY_MAX_CHARS = 800      # cap on the long term notes
MEMORY_CONVO_CHARS = 3000   # how much of a finished chat the note taker reads

BASE_DIR = Path(__file__).parent
SESSIONS_DIR = BASE_DIR / "sessions"
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "patron_memory.txt"
SESSIONS_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)


def _load_system_prompt() -> str:
    """Read the SYSTEM block straight from the Modelfile so the app and the
    command line always use the same prompt."""
    text = (BASE_DIR.parent / "Modelfile").read_text(encoding="utf-8")
    match = re.search(r'SYSTEM """(.*?)"""', text, re.DOTALL)
    if not match:
        raise RuntimeError("Could not find the SYSTEM block in Modelfile")
    return match.group(1).strip()


SYSTEM_PROMPT = _load_system_prompt()


# --- Long term memory (carried across chats) ---------------------------------
# One notes file for the single demo patron. The model writes the notes, and
# they are injected into the system prompt. Nothing is retrieved or looked up.

_memory_lock = asyncio.Lock()
_background_tasks = set()

MEMORY_SYSTEM_PROMPT = (
    "You write short notes about a library patron so that a librarian can "
    "remember them on the next visit. You output only the notes."
)


def load_memory() -> str:
    if MEMORY_FILE.exists():
        return MEMORY_FILE.read_text(encoding="utf-8").strip()
    return ""


def clear_memory() -> None:
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()


def build_system_prompt() -> str:
    """Base prompt, plus the patron notes when there are any. The notes go at
    the end of the prompt, closest to generation."""
    memory = load_memory()
    if not memory:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + "\n\n## WHAT YOU REMEMBER ABOUT THIS PATRON FROM PREVIOUS VISITS\n"
        + memory
        + "\nUse these notes naturally when they are relevant, for example to "
        "suggest books they might like. Do not recite them unprompted, and "
        "never treat them as catalogue data."
    )


def _memory_request(existing: str, conversation: str) -> str:
    return (
        "Existing notes about the patron:\n"
        f"{existing or '(none)'}\n\n"
        "What the patron said in the chat that just ended:\n"
        f"{conversation}\n\n"
        "Update the notes using ONLY facts the patron stated themselves. "
        "Rules:\n"
        "- Name: only if the patron said their name.\n"
        "- Likes: only genres or authors the patron said they like.\n"
        "- Asked to borrow: only books the patron said they want to borrow, "
        "check out, or hold. Asking whether a book is available is NOT a "
        "request to borrow it.\n"
        "- Never write placeholders such as Unknown, None, or N/A. If you have "
        "no fact for a line, leave that line out completely.\n"
        "- Never guess, never add books that were not named, and never record "
        "card numbers or ID numbers.\n"
        "Keep the existing notes unless the patron contradicted them. Write at "
        "most 5 short lines, each starting with '- '.\n\n"
        "Example of the format only (never copy its facts). If the patron said "
        "'I'm Sam. I love fantasy. Is Dune available? I'd like to borrow "
        "Foundation.' the notes are:\n"
        "- Name: Sam\n"
        "- Likes: fantasy\n"
        "- Asked to borrow: Foundation\n\n"
        "If the patron stated nothing worth remembering, output exactly: NONE"
    )


async def _update_memory(transcript: List[Dict[str, str]]) -> None:
    patron_lines = [m["content"] for m in transcript if m["role"] == "user"]
    if not patron_lines:
        return
    # Only the patron's own words are shown to the note taker. Willow's replies
    # can contain mistakes, and notes built from them would store those mistakes.
    conversation = "\n".join(f"Patron: {line}" for line in patron_lines)[
        -MEMORY_CONVO_CHARS:
    ]

    async with _memory_lock:
        existing = load_memory()
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 200},
                        "messages": [
                            {"role": "system", "content": MEMORY_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": _memory_request(existing, conversation),
                            },
                        ],
                    },
                )
                response.raise_for_status()
                notes = response.json()["message"]["content"].strip()
        except Exception as exc:
            print(f"Memory update skipped: {exc}")
            return
        if not notes or notes.upper().startswith("NONE"):
            return
        MEMORY_FILE.write_text(notes[:MEMORY_MAX_CHARS], encoding="utf-8")


# --- Session / short term memory ---------------------------------------------


class Session:
    """Dialogue state for one chat."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []     # sliding window sent to the model
        self.transcript: List[Dict[str, str]] = []  # full chat, saved and summarised
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_turn(self, role: str, content: str) -> None:
        self.transcript.append({"role": role, "content": content})
        self.history.append({"role": role, "content": content})
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def build_messages(self, new_user_message: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": build_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": new_user_message})
        return messages


# In-memory session store, keyed by session id (single process demo).
_sessions: Dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id)
    return _sessions[session_id]


async def end_session(session_id: str) -> None:
    """Close a chat: save it, then let the model update the patron notes."""
    session = _sessions.pop(session_id, None)
    if session is None:
        return
    save_session(session)
    await _update_memory(session.transcript)


def schedule_end_session(session_id: str) -> None:
    """Run end_session in the background so the UI is never blocked."""
    task = asyncio.create_task(end_session(session_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# --- Saved transcripts (history sidebar) ---------------------------------------
# This is the app saving its own chat logs. The model is never given anything
# looked up from these files, apart from the notes described above.


def _session_file(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(session: Session) -> None:
    if not session.transcript:
        return
    first_user_msg = next(
        (m["content"] for m in session.transcript if m["role"] == "user"), ""
    )
    title = (first_user_msg[:48] + "...") if len(first_user_msg) > 48 else first_user_msg
    data = {
        "session_id": session.session_id,
        "title": title or "New conversation",
        "created_at": session.created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "history": session.transcript,
    }
    _session_file(session.session_id).write_text(json.dumps(data, indent=2))


def list_sessions() -> List[Dict]:
    """Metadata for the sidebar, newest first."""
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


# --- LLM streaming ---------------------------------------------------------------


async def stream_response(
    session_id: str, user_message: str
) -> AsyncGenerator[str, None]:
    """Yield the model's reply token by token for one patron message."""
    session = get_session(session_id)

    # If a memory update from the previous chat is still running, wait for it
    # so this chat starts with the latest notes.
    async with _memory_lock:
        pass

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
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response += token
                    yield token
                if chunk.get("done"):
                    break

    session.add_turn("user", user_message)
    session.add_turn("assistant", full_response)
    save_session(session)