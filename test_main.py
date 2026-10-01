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