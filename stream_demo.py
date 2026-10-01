import asyncio
import json

import httpx

from main import OLLAMA_BASE_URL, OLLAMA_MODEL


async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain APIs in five simple sentences.",
                    }
                ],
                "stream": True,
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                if "error" in chunk:
                    raise RuntimeError(chunk["error"])

                text = chunk.get("message", {}).get("content", "")
                print(text, end="", flush=True)

                if chunk.get("done"):
                    print()
                    break


if __name__ == "__main__":
    asyncio.run(main())