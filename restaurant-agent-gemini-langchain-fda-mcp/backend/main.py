import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ToolCallRequest, dynamic_prompt, wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langsmith import traceable
from pydantic import BaseModel, Field

import flow
from evals.api import router as eval_router
from jev_decision import safe_interpret_answer
from tools import tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """
You are a helpful restaurant recommendation assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability
4. FDA menu rating tools (when available): rate a dish or a whole menu
   A-E against FDA Daily Values, estimate a dish's nutrition, and
   explain a dish's grade

Use the appropriate tool whenever necessary.

When the user asks how healthy a dish or restaurant is, use the FDA
rating tools; for a restaurant, rate the dishes from its details.
Say the grade is unofficial: the FDA does not rate restaurants.

When the user asks where they can get a food (e.g. "where can I get
biryani?"), work out which cuisine the food belongs to, search that
cuisine, then get the details of each restaurant found and recommend
the ones whose speciality includes that food. If none do, say so and
suggest the restaurants of that cuisine instead.

Do not invent restaurant information.

For reservations, never guess the date or party size. If either is
missing, ask the user for it before calling the availability tool;
do not assume a party size such as 2.
You can only check availability, not make bookings: never say a table
is booked, reserved or "all set".
Each restaurant has a few time slots per day; check without a time to
list the free slots, or with a time to check one slot.
When the user picks a time, check that slot with the tool before
answering, even if an earlier check listed it: seats change.

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

Be friendly, conversational, and concise.
"""

@dynamic_prompt
def prompt_with_today(request: ModelRequest) -> str:
    # Today's date, so Gemini can resolve "tomorrow" or "this Saturday".
    return f"{system_prompt}\nToday is {date.today():%A %Y-%m-%d}."


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty"]
BLOCKED = "Not checked:"
PARTY_PHRASES = {
    1: ["just me", "only me", "myself", "alone", "solo"],
    2: ["couple", "me and my", "both of us", "pair"],
}


def user_said_party_size(messages: list, size: int) -> bool:
    """Whether a user message states `size` people, e.g. "2", "two", "me and my wife"."""
    said = " ".join(_text(m.content) for m in messages if getattr(m, "type", None) == "human").lower()
    words = [rf"\b{size}\b"] + ([rf"\b{NUMBER_WORDS[size]}\b"] if size < len(NUMBER_WORDS) else [])
    words += [re.escape(p) for p in PARTY_PHRASES.get(size, [])]
    return any(re.search(w, said) for w in words)


@wrap_tool_call
async def require_party_size(request: ToolCallRequest, handler):
    # Gemini tends to assume a party of 2 despite the prompt; refuse the check
    # until the user has actually said how many people are coming.
    call = request.tool_call
    if call["name"] == "check_reservation_availability":
        size = call["args"].get("party_size")
        size = int(size) if isinstance(size, (int, float)) and size == int(size) else None
        if size is None or not user_said_party_size(request.state["messages"], size):
            return ToolMessage(
                content=f"{BLOCKED} the user has not said how many people are in the party. "
                        "Ask them for the number of people; do not assume one.",
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
    return await handler(request)


def load_fda_tools() -> list:
    """Tools from the FDA menu rating MCP server; none if it can't be reached."""
    url = os.getenv("FDA_MCP_URL", "https://fda-rating.onrender.com/mcp")
    if not url:
        return []

    connection = {"transport": "streamable_http", "url": url}
    if token := os.getenv("FDA_MCP_TOKEN"):
        connection["headers"] = {"Authorization": f"Bearer {token}"}
    client = MultiServerMCPClient({"fda-menu-rating": connection})
    try:
        # Run in a thread: this module is imported inside uvicorn's running loop.
        # The timeout allows for a Render free-tier cold start.
        with ThreadPoolExecutor(1) as pool:
            fda_tools = pool.submit(
                asyncio.run, asyncio.wait_for(client.get_tools(), timeout=90)
            ).result()
    except Exception:
        logger.warning("FDA MCP server at %s unavailable; continuing without it", url, exc_info=True)
        return []

    logger.info("FDA MCP tools from %s: %s", url, [t.name for t in fda_tools])
    return fda_tools


agent = create_agent(
    model=init_chat_model(os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")),
    tools=tools + load_fda_tools(),
    middleware=[prompt_with_today, require_party_size],
)

app = FastAPI(title="Restaurant Assistant API")

# Browsers send the origin without a trailing slash, so strip any; empty means allow all.
cors_origins = [
    o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
] or ["*"]
logger.info("CORS origins: %s", cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)  # Jev's conversation
    gemini_history: list[Message] | None = None  # Gemini's own conversation
    state: dict | None = None  # echo back the `state` from the last response


class ChatResponse(BaseModel):
    # Gemini's separate run on the same input: source, available, reply, tool_calls
    google: dict
    # Jev is the agent: source, available, understood, reply, tool_calls, next_question
    jev: dict
    state: dict  # send this back with the next request


def _text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content) if content is not None else ""


@traceable(name="Gemini Restaurant Agent", run_type="chain")
async def run_gemini(messages: list[dict]) -> dict:
    """Gemini answers with its own tool calls; failures are reported, not raised."""
    try:
        result = await agent.ainvoke({"messages": messages})
    except Exception:
        logger.exception("Gemini agent invocation failed")
        return {"available": False, "reply": "", "tool_calls": []}

    out = result["messages"]

    # Arguments the model passed to each tool, keyed by tool-call id.
    call_args = {
        c["id"]: c.get("args", {})
        for m in out
        for c in (getattr(m, "tool_calls", None) or [])
    }

    tool_calls = [
        {
            "name": m.name,
            "args": call_args.get(getattr(m, "tool_call_id", None), {}),
            "output": _text(m.content),
            # Refused by require_party_size: the tool did not run.
            "blocked": getattr(m, "status", None) == "error" and _text(m.content).startswith(BLOCKED),
        }
        for m in out
        if getattr(m, "type", None) == "tool"
    ]

    return {
        "available": True,
        "reply": _text(out[-1].content) if out else "",
        "tool_calls": tool_calls,
    }


def _question_line(question: dict) -> str:
    """The question as plain text, so Gemini sees what the user was asked."""
    options = [c["label"] for c in question["choices"] if c["key"] != "restart"]
    return f"{question['text']} Options: {', '.join(options)}."


@traceable(name="Jev Agent Turn", run_type="chain")
async def run_turn(req: ChatRequest) -> ChatResponse:
    history = [m.model_dump() for m in req.history]
    gemini_history = [m.model_dump() for m in (req.gemini_history or req.history)]
    state = req.state or dict(flow.START)
    asked = flow.question_for(state)

    # Two separate executions on the same input, run at the same time:
    # Jev reads the answer to the question it asked; Gemini answers on its own
    # with its own history and tools. Neither sees the other's output.
    gemini_messages = gemini_history + [
        {"role": "assistant", "content": _question_line(asked)},
        {"role": "user", "content": req.message},
    ]
    jev_result, gemini = await asyncio.gather(
        safe_interpret_answer(asked, req.message, history),
        run_gemini(gemini_messages),
    )

    if jev_result["available"]:
        method = "jev"
        choice = jev_result["choice"]
    else:
        method = "fallback"
        choice = flow.match_label(asked, req.message)

    # Jev's decision drives its tools, its reply and the next question.
    advance = flow.step(state, choice)

    new_state = advance["state"]
    next_question = flow.question_for(new_state)
    if advance["notice"]:
        next_question["text"] = f"{advance['notice']} {next_question['text']}"

    return ChatResponse(
        google={"source": "Google Gemini", **gemini},
        jev={
            "source": "Jev",
            "available": jev_result["available"],
            "understood": {
                "question_id": asked["id"],
                "choice": choice,
                "confidence": jev_result["confidence"],
                "method": method,
            },
            "reply": advance["reply"],
            "tool_calls": [advance["tool"]] if advance["tool"] else [],
            "next_question": next_question,
        },
        state=new_state,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await run_turn(req)


app.include_router(eval_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
