import httpx
import respx
from fastapi.testclient import TestClient
import json
from main import app, OLLAMA_BASE_URL

client = TestClient(app)
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_spaces_only_prompt():
    response = client.post("/chat", json={"prompt": "   "})

    assert response.status_code == 422


@respx.mock
def test_chat_success():
    route = respx.post(OLLAMA_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Hello from the model!",
                }
            },
        )
    )

    response = client.post("/chat", json={"prompt": "Hello"})

    assert route.called
    assert response.status_code == 200
    assert response.json() == {"answer": "Hello from the model!"}


@respx.mock
def test_ollama_offline():
    respx.post(OLLAMA_CHAT_URL).mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    response = client.post("/chat", json={"prompt": "Hello"})

    assert response.status_code == 503


@respx.mock
def test_ollama_timeout():
    respx.post(OLLAMA_CHAT_URL).mock(
        side_effect=httpx.ReadTimeout("Timed out")
    )

    response = client.post("/chat", json={"prompt": "Hello"})

    assert response.status_code == 504


@respx.mock
def test_chat_sends_history():
    route = respx.post(OLLAMA_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "You are learning Python.",
                }
            },
        )
    )

    response = client.post(
        "/chat",
        json={
            "prompt": "What am I learning?",
            "history": [
                {
                    "role": "user",
                    "content": "I am learning Python.",
                },
                {
                    "role": "assistant",
                    "content": "I can help with that.",
                },
            ],
        },
    )

    assert response.status_code == 200

    sent_body = json.loads(route.calls.last.request.content)

    assert sent_body["messages"] == [
        {"role": "user", "content": "I am learning Python."},
        {"role": "assistant", "content": "I can help with that."},
        {"role": "user", "content": "What am I learning?"},
    ]

def test_history_message_too_long():
    response = client.post(
        "/chat",
        json={
            "prompt": "Explain more.",
            "history": [
                {
                    "role": "assistant",
                    "content": "a" * 4001,
                }
            ],
        },
    )

    assert response.status_code == 422

@respx.mock
def test_missing_model():
    respx.post(OLLAMA_CHAT_URL).mock(
        return_value=httpx.Response(
            404,
            json={"error": "model not found"},
        )
    )

    response = client.post(
        "/chat",
        json={"prompt": "Hello"},
    )

    assert response.status_code == 503
    assert "ollama pull" in response.json()["detail"]

def read_events(response):
    return [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]


@respx.mock
def test_stream_success():
    chunks = [
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]

    body = "\n".join(json.dumps(chunk) for chunk in chunks) + "\n"

    respx.post(OLLAMA_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
    )

    response = client.post(
        "/chat/stream",
        json={"prompt": "Hello"},
    )

    assert response.status_code == 200
    assert read_events(response) == [
        {"content": "Hello"},
        {"content": " world"},
        {"done": True},
    ]


@respx.mock
def test_stream_offline():
    respx.post(OLLAMA_CHAT_URL).mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    response = client.post(
        "/chat/stream",
        json={"prompt": "Hello"},
    )

    events = read_events(response)

    assert response.status_code == 200
    assert "error" in events[-1]
    assert not any(event.get("done") for event in events)


@respx.mock
def test_stream_incomplete():
    respx.post(OLLAMA_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=json.dumps({
                "message": {"content": "Partial answer"},
                "done": False,
            }) + "\n",
        )
    )

    response = client.post(
        "/chat/stream",
        json={"prompt": "Hello"},
    )

    events = read_events(response)

    assert events[0] == {"content": "Partial answer"}
    assert "error" in events[-1]
    assert not any(event.get("done") for event in events)


def test_stream_spaces_only():
    response = client.post(
        "/chat/stream",
        json={"prompt": "   "},
    )

    assert response.status_code == 422