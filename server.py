"""
ChatGPT MCP Server
Wraps the OpenAI ChatGPT API as an MCP server with SSE/Streamable HTTP transport.
Tools: chatgpt_ask, chatgpt_list_models
"""

import os
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = "https://api.openai.com/v1"
PORT = int(os.environ.get("PORT", "8000"))

# --- MCP Server ---
mcp = FastMCP("ChatGPT MCP Server")


def _openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


@mcp.tool()
async def chatgpt_ask(prompt: str, model: str | None = None, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """
    Send a prompt to ChatGPT and get a response.

    Args:
        prompt: The text prompt to send to ChatGPT.
        model: The OpenAI model to use (defaults to OPENAI_MODEL env var, usually gpt-4o-mini).
        temperature: Sampling temperature (0-2). Higher = more creative. Default 0.7.
        max_tokens: Maximum tokens in the response. Default 2048.
    """
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY environment variable is not set."

    use_model = model or OPENAI_MODEL
    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=_openai_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"OpenAI API error {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"


@mcp.tool()
async def chatgpt_list_models() -> str:
    """
    List available OpenAI models. Returns model IDs as a newline-separated list.
    """
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY environment variable is not set."

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                f"{OPENAI_BASE_URL}/models",
                headers=_openai_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            model_ids = sorted([m["id"] for m in data["data"]])
            return "\n".join(model_ids)
        except httpx.HTTPStatusError as e:
            return f"OpenAI API error {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error listing models: {str(e)}"


async def health(request):
    return JSONResponse({"status": "ok", "server": "ChatGPT MCP Server"})


# Build the app: mount the MCP SSE app and add a health endpoint
sse_app = mcp.sse_app()

app = Starlette(
    routes=[
        Route("/health", health),
        Mount("/", app=sse_app),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
