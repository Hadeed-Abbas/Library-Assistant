"""
FastAPI backend for the Library Assistant chatbot.

Endpoints:
- GET    /health          liveness check
- GET    /sessions        list saved past conversations, newest first
- GET    /sessions/{id}   full transcript of one saved conversation
- GET    /memory          the long term notes Willow keeps about the patron
- DELETE /memory          clear those notes
- WS     /ws/chat         real-time chat, one session per connection

WebSocket message protocol (JSON both directions):

Client -> Server:
    {"message": "do you have the hobbit?"}     a normal chat turn
    {"type": "reset"}                            end this chat and start a new one

Server -> Client:
    {"type": "session", "session_id": "..."}    sent on connect and after a reset
    {"type": "token", "content": "..."}          one per streamed token
    {"type": "done"}                             end of a response
    {"type": "error", "message": "..."}          malformed request or model error;
                                                 the connection stays open
"""

import json
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from conversation_manager import (
    clear_memory,
    list_sessions,
    load_memory,
    load_session_transcript,
    schedule_end_session,
    stream_response,
)

app = FastAPI(title="Library Assistant Chatbot")

# The frontend is opened from a file, so it is a different origin from the API.
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
    return list_sessions()


@app.get("/sessions/{session_id}")
async def get_session_transcript(session_id: str):
    transcript = load_session_transcript(session_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return transcript


@app.get("/memory")
async def get_memory():
    return {"memory": load_memory()}


@app.delete("/memory")
async def delete_memory():
    clear_memory()
    return {"memory": ""}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()

    # One session per connection. Each connection is its own asyncio task, so
    # one user's slow generation does not block another user's turn.
    session_id = str(uuid.uuid4())
    await websocket.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Malformed request: expected valid JSON."}
                )
                continue

            if not isinstance(data, dict):
                await websocket.send_json(
                    {"type": "error", "message": "Malformed request: expected a JSON object."}
                )
                continue

            # Reset: close this chat (saved, and the model updates the patron
            # notes in the background) and start a fresh session id.
            if data.get("type") == "reset":
                schedule_end_session(session_id)
                session_id = str(uuid.uuid4())
                await websocket.send_json({"type": "session", "session_id": session_id})
                continue

            user_message = data.get("message")
            if not user_message or not isinstance(user_message, str):
                await websocket.send_json(
                    {"type": "error", "message": "Malformed request: 'message' (string) is required."}
                )
                continue

            try:
                async for token in stream_response(session_id, user_message):
                    await websocket.send_json({"type": "token", "content": token})
                await websocket.send_json({"type": "done"})
            except Exception as exc:
                # Ollama unreachable, model missing, and so on. The connection
                # stays open so the user can simply try again.
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"The assistant hit an error generating a response: {exc}",
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        # Any other failure on a dead or broken connection (for example a send
        # after the client closed mid-reply) must not skip the cleanup below.
        print(f"Connection ended with an error: {exc}")
    finally:
        # Always close the chat: save it and let the model update the patron
        # notes, however the connection ended.
        schedule_end_session(session_id)