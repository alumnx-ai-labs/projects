# Restaurant Assistant Backend

FastAPI backend with a single Gemini agent (LangChain `create_agent`). `POST /chat` takes `{message, history}` and returns `{source, reply, tool_calls}`, where each tool call is `{name, args, output}`.

Tools (`tools.py`): search restaurants by cuisine, get restaurant details, check reservation availability.

Endpoints: `POST /chat` and `GET /health`; `GET /` redirects to the Swagger UI. Run `uvicorn main:app --reload`, then open http://localhost:8000/docs to try the agent (there is no separate frontend).

Environment variables (set in `.env` locally or in the host's settings; never commit them): `GOOGLE_API_KEY`, `MODEL` (optional, default `google_genai:gemini-3.5-flash-lite`), `CORS_ORIGINS`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.

## main.py

```python
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from tools import tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """
You are a helpful restaurant assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability

Use the appropriate tool whenever necessary.

Do not invent restaurant information.

For reservations, never guess the date or party size. If either is
missing, ask the user for it before calling the availability tool.
Dates in the system use the format YYYY-MM-DD.

If the available tools cannot answer the question, clearly say that
the requested information is outside your available capabilities.

Be friendly, conversational, and concise.
"""

# Fail fast when Gemini is overloaded instead of retrying for many seconds.
model = init_chat_model(
    os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite"),
    max_retries=1,
    timeout=30,
)

agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)

app = FastAPI(
    title="Restaurant Assistant API",
    description=(
        "Chat with a Gemini restaurant agent. Open **POST /chat**, click "
        "*Try it out*, and send a message. To continue a conversation, copy the "
        "previous turns into `history`."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(description="`user` or `assistant`", examples=["user"])
    content: str


class ChatRequest(BaseModel):
    message: str = Field(examples=["Is Spice Palace free on 2024-11-15 for 4 people?"])
    history: list[Message] = Field(default_factory=list, description="Earlier turns, oldest first")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Suggest some Italian restaurants", "history": []},
                {
                    "message": "Tell me more about the first one",
                    "history": [
                        {"role": "user", "content": "Suggest some Italian restaurants"},
                        {"role": "assistant", "content": "Try Pizza Bella, Pasta Dreams or Roma Kitchen."},
                    ],
                },
            ]
        }
    }


class ToolCall(BaseModel):
    name: str
    args: dict
    output: str = Field(description="Tool result; lists and dicts are JSON text")


class ChatResponse(BaseModel):
    source: str = "Google Gemini"
    reply: str
    tool_calls: list[ToolCall]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content) if content is not None else ""


def _tool_calls(messages: list) -> list[dict]:
    """Pair each tool call the agent made with its output."""
    outputs = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    calls = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for c in m.tool_calls:
            out = outputs.get(c["id"])
            raw = out.content if out else ""
            calls.append(
                {
                    "name": c["name"],
                    "args": c["args"],
                    "output": raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False),
                }
            )
    return calls


@traceable(name="Gemini Restaurant Agent", run_type="chain")
async def run_turn(req: ChatRequest) -> ChatResponse:
    messages = [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})
    result = await agent.ainvoke({"messages": messages})

    new = result["messages"][len(messages):]  # only this turn's messages
    return ChatResponse(reply=_text(new[-1].content), tool_calls=_tool_calls(new))


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.post("/chat", response_model=ChatResponse, tags=["agent"], summary="Send a message to the agent")
async def chat(req: ChatRequest):
    try:
        return await run_turn(req)
    except Exception:
        logger.exception("Gemini agent failed")
        raise HTTPException(status_code=502, detail="The Gemini agent failed to respond.")


@app.get("/health", tags=["status"])
async def health():
    return {"status": "ok"}
```
