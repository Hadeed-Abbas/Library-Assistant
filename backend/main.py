"""
FastAPI backend for the Library Assistant chatbot.

Endpoints:
- GET  /health              — basic liveness check (REST)
- GET  /sessions            — list saved past conversations, newest first (REST)
- GET  /sessions/{id}       — full transcript of one saved conversation (REST)
- WS   /ws/chat              — real-time chat, one session per connection

WebSocket message protocol (JSON both directions):

Client -> Server:
    {"message": "do you have the hobbit?"}          # normal chat turn
    {"type": "reset"}                                 # start a new session

Server -> Client:
    {"type": "session", "session_id": "..."}           # sent on connect AND
                                                          # after a reset (new id)
    {"type": "token", "content": "..."}                 # one per streamed token
    {"type": "done"}                                    # end of a response
    {"type": "error", "message": "..."}                 # malformed request or
                                                          # model/inference error;
                                                          # connection stays open
"""

import json
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from conversation_manager import (
    list_sessions,
    load_session_transcript,
    reset_session,
    stream_response,
)

app = FastAPI(title="Library Assistant Chatbot")

# Allows the frontend (served from a different origin/port) to connect.
# Fine for an assignment demo; tighten allow_origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/sessions")
async def get_sessions():
    """Metadata for every saved conversation, for the history sidebar."""
    return list_sessions()


@app.get("/sessions/{session_id}")
async def get_session_transcript(session_id: str):
    """Full transcript of one saved conversation, read-only."""
    transcript = load_session_transcript(session_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return transcript


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()

    # One session per connection — this is what gives you "multiple
    # concurrent users without blocking each other": each connection is its
    # own asyncio task with its own session_id and history in
    # conversation_manager, so one user's slow generation doesn't hold up
    # another user's turn.
    session_id = str(uuid.uuid4())
    await websocket.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            raw = await websocket.receive_text()

            # --- Parse incoming message ---
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Malformed request: expected valid JSON.",
                    }
                )
                continue

            if not isinstance(data, dict):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Malformed request: expected a JSON object.",
                    }
                )
                continue

            # --- Handle a reset: start a brand new session id so the old
            # conversation's saved transcript is left intact on disk rather
            # than being overwritten by the next turn. ---
            if data.get("type") == "reset":
                reset_session(session_id)
                session_id = str(uuid.uuid4())
                await websocket.send_json(
                    {"type": "session", "session_id": session_id}
                )
                continue

            # --- Validate the chat message ---
            user_message = data.get("message")
            if not user_message or not isinstance(user_message, str):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Malformed request: 'message' (string) is required.",
                    }
                )
                continue

            # --- Stream the response ---
            try:
                async for token in stream_response(session_id, user_message):
                    await websocket.send_json({"type": "token", "content": token})
                await websocket.send_json({"type": "done"})
            except Exception as exc:
                # Covers Ollama being unreachable, a model error, etc.
                # The connection stays open so the user can just try again.
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"The assistant hit an error generating a response: {exc}",
                    }
                )

    except WebSocketDisconnect:
        reset_session(session_id)