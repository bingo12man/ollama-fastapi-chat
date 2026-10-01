import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
import os
from dotenv import load_dotenv
from typing import Literal
from pathlib import Path
from fastapi.responses import FileResponse


load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:1b",
)

app=FastAPI()

BASE_DIR = Path(__file__).resolve().parent


@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(BASE_DIR / "index.html")

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=10,
    )

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/chat")
async def chat(request:ChatRequest):
    prompt=request.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=422,
            detail="Prompt cannot contain only spaces"
        )

    messages = []

    for message in request.history:
        content = message.content.strip()

        if not content:
            raise HTTPException(
                status_code=422,
                detail="History messages cannot contain only spaces.",
            )

        messages.append({
            "role": message.role,
            "content": content,
        })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream":False,
                },
            )

            response.raise_for_status()
            data=response.json()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Model took too long to respond. Try Again"
        )

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The configured model may be missing. "
                    f"Run: ollama pull {OLLAMA_MODEL}"
                ),
            )

        raise HTTPException(
            status_code=502,
            detail="Model returned an error"
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Cannot communicate with model. Check if its running"
        )

    try:
        answer = data["message"]["content"]

        if not isinstance(answer,str):
            raise TypeError("Expected a text answer")

    except (KeyError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unexpected response Structure"
        )

    return {"answer": answer}

