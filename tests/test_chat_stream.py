"""
Tests for the streaming `/chat` endpoint (main.py:chat_endpoint).

These exercise the SSE framing/filtering logic end to end through FastAPI's
TestClient, with the LangGraph agent and the database swapped out for fakes
via dependency overrides + monkeypatching `ChatBot.astream_events` — no real
OpenAI calls and no writes to the real Postgres database.
"""
import json

import pytest
from fastapi.testclient import TestClient

import main
from utils.db_models import ChatSession


class FakeQuery:
    """Stand-in for a SQLAlchemy Query: `.filter(...).first()` returns a preset row."""

    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class FakeDB:
    """Stand-in for the SQLAlchemy Session used by the endpoint."""

    def __init__(self, existing_session=None):
        self.existing_session = existing_session
        self.committed = False
        self.rolled_back = False

    def query(self, model):
        return FakeQuery(self.existing_session)

    def add(self, obj):
        pass

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        pass

    def rollback(self):
        self.rolled_back = True


class FakeChunk:
    """Stand-in for the `AIMessageChunk` a real `on_chat_model_stream` event carries."""

    def __init__(self, content):
        self.content = content


def _parse_sse(body: str):
    """Splits a raw SSE response body into its decoded JSON frames."""
    frames = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        _, _, payload = block.partition("data: ")
        frames.append(json.loads(payload))
    return frames


@pytest.fixture
def client():
    main.app.dependency_overrides[main.get_current_user_from_cookie] = lambda: "test-user"
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()


def test_stream_forwards_only_tagged_tokens_and_persists(client, monkeypatch):
    fake_session = ChatSession(id="session-1", user_id="test-user", chat_state=[])
    fake_db = FakeDB(existing_session=fake_session)

    def override_get_db():
        yield fake_db

    main.app.dependency_overrides[main.get_db] = override_get_db

    events = [
        # Internal LLM call (e.g. the rewriter/classifier) — must NOT reach the client.
        {"event": "on_chat_model_stream", "name": "ChatOpenAI", "tags": ["some_internal_tag"],
         "data": {"chunk": FakeChunk("SHOULD_NOT_APPEAR")}},
        # The two chunks of the actual user-facing answer.
        {"event": "on_chat_model_stream", "name": "ChatOpenAI", "tags": ["final_answer"],
         "data": {"chunk": FakeChunk("Hello ")}},
        {"event": "on_chat_model_stream", "name": "ChatOpenAI", "tags": ["final_answer"],
         "data": {"chunk": FakeChunk("world!")}},
        {"event": "on_chain_end", "name": "classifier", "tags": [],
         "data": {"output": {"needs_tools": True}}},
        {"event": "on_chain_end", "name": "grader_node", "tags": [],
         "data": {"output": {"status": "CORRECT"}}},
    ]

    async def fake_astream_events(*args, **kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(main.ChatBot, "astream_events", fake_astream_events)

    response = client.post("/chat", json={"raw_question": "hi", "session_id": "session-1"})

    assert response.status_code == 200
    frames = _parse_sse(response.text)

    token_frames = [f for f in frames if f["type"] == "token"]
    assert [f["content"] for f in token_frames] == ["Hello ", "world!"]

    done_frame = next(f for f in frames if f["type"] == "done")
    assert done_frame == {
        "type": "done",
        "session_id": "session-1",
        "status": "CORRECT",
        "used_tools": True,
    }

    assert fake_session.chat_state == [
        {"role": "user", "content": "hi"},
        {"role": "ai", "content": "Hello world!"},
    ]
    assert fake_db.committed is True


def test_stream_error_rolls_back_and_emits_error_frame(client, monkeypatch):
    fake_session = ChatSession(id="session-2", user_id="test-user", chat_state=[])
    fake_db = FakeDB(existing_session=fake_session)

    def override_get_db():
        yield fake_db

    main.app.dependency_overrides[main.get_db] = override_get_db

    async def flaky_astream_events(*args, **kwargs):
        yield {"event": "on_chat_model_stream", "name": "ChatOpenAI", "tags": ["final_answer"],
               "data": {"chunk": FakeChunk("partial")}}
        raise RuntimeError("boom")

    monkeypatch.setattr(main.ChatBot, "astream_events", flaky_astream_events)

    response = client.post("/chat", json={"raw_question": "hi", "session_id": "session-2"})

    assert response.status_code == 200
    frames = _parse_sse(response.text)

    assert frames[0] == {"type": "token", "content": "partial"}
    assert frames[-1]["type"] == "error"

    assert fake_db.rolled_back is True
    assert fake_db.committed is False
    # The exception happened before the DB-persist step, so nothing was appended.
    assert fake_session.chat_state == []


def test_stream_returns_404_for_unknown_session(client):
    fake_db = FakeDB(existing_session=None)

    def override_get_db():
        yield fake_db

    main.app.dependency_overrides[main.get_db] = override_get_db

    response = client.post("/chat", json={"raw_question": "hi", "session_id": "does-not-exist"})

    assert response.status_code == 404
