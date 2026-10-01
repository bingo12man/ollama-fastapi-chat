import httpx
import respx
from fastapi.testclient import TestClient

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