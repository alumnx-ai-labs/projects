import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

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

MODEL = os.getenv("MODEL", "gemini-3.8-flash").removeprefix("google_genai:")
MAX_TOOL_ROUNDS = 5

client = genai.Client()  # reads GOOGLE_API_KEY from the environment

config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
)

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


def _run_tool(name: str, args: dict):
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(**args)
    except Exception as e:  # report to the model instead of crashing the turn
        return {"error": f"{type(e).__name__}: {e}"}


async def run_turn(req: ChatRequest) -> ChatResponse:
    contents = [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part(text=m.content)],
        )
        for m in req.history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=req.message)]))

    tool_calls: list[dict] = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.aio.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        calls = response.function_calls or []
        if not calls:
            return ChatResponse(reply=response.text or "", tool_calls=tool_calls)

        # keep the model's turn verbatim, then answer every call it made
        contents.append(response.candidates[0].content)
        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            out = _run_tool(call.name, args)
            tool_calls.append(
                {
                    "name": call.name,
                    "args": args,
                    "output": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False),
                }
            )
            result_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": out})
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return ChatResponse(
        reply="Sorry, I couldn't complete that request. Please try rephrasing.",
        tool_calls=tool_calls,
    )


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
