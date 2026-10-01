# Ollama + FastAPI Local Chat

A step-by-step project for running a local LLM through a FastAPI API.

## Stage 1
Implemented GET /health, which returns {"status": "ok"}.

## Run
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload

## Next
Connect the chat endpoint to Ollama using llama3.2:1b.
