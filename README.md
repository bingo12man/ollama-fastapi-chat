# Ollama + FastAPI Local Chat

A local chat application built with FastAPI, Ollama, and a simple browser interface.

## Features

- Local generation using llama3.2:1b
- Browser chat interface
- Conversation history sent with each request
- Input validation
- Error handling for unavailable Ollama, timeouts, and upstream errors
- Automated API tests with simulated Ollama responses

## Architecture

Browser → FastAPI → Ollama → local model

The browser sends a prompt and recent conversation history.
FastAPI validates the request and calls Ollama.
Ollama generates an answer, which FastAPI returns to the browser.

## Requirements

- Python 3.10 or newer
- Ollama installed and running
- Git

## Setup

Clone this repository and open its folder.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the model:

```bash
ollama pull llama3.2:1b
```

Copy the configuration template:

```bash
cp .env.example .env
```

Default configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
```

Start FastAPI:

```bash
python -m uvicorn main:app --reload
```

Open:

- Chat: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## API Example

```bash
curl http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain Python in two sentences.","history":[]}'
```

The response contains an `answer` field.

## Tests

```bash
python -m pytest -q
```

Tests simulate Ollama responses, so Ollama does not need to run during testing.

## Error Responses

| Status | Meaning |
|---|---|
| 422 | Invalid prompt or history |
| 503 | Ollama or configured model unavailable |
| 504 | Ollama request timed out |
| 502 | Upstream error or unexpected response structure |

## Current Limitations

- Browser history is cleared on refresh.
- The browser sends only the last 10 messages.
- Each history message is limited to 4,000 characters.
- Longer answers are displayed fully but truncated in stored browser history.
- Responses are returned after generation finishes; streaming is not implemented.
- No authentication, database, or production deployment is included.
- `/health` checks FastAPI, not Ollama availability.
- Small-model answers may be inaccurate.

## Privacy

With the default localhost configuration, prompts are sent to Ollama on
the same computer. Changing OLLAMA_BASE_URL changes where prompts are sent.

The `.env` file and virtual environment are excluded from Git.