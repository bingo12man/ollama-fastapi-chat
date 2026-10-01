import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
import os
from dotenv import load_dotenv

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

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1,max_length=4000)

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

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages":[
                        {"role":"user","content":request.prompt}
                    ],
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
        if exc.resonse.status_code == 404:
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

