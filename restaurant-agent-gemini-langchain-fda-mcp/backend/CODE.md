# Restaurant Assistant Backend

FastAPI backend. `POST /chat` returns three labeled parts: `google` (Gemini agent answer and tool calls), `jev` (what Jev understood from the user's answer, plus the next question in Jev `Choice` format) and `state` (send it back with the next request).

Guided flow: Q1 cuisine (tool 1) -> Q2 restaurant -> Q3 date -> Q4 party size -> availability check. If the user dislikes the options, or the table is fully booked, the flow restarts at Q1.

Endpoints: `POST /chat` and `GET /health`. Run: `uvicorn main:app --reload`.

Evaluation: `evals/` runs test cases through `POST /chat` and scores Jev (choice, next question, tool calls) and Gemini (tool calls, reply keywords, LLM-judged criteria) separately. Run from `backend/`: `python -m evals.run_evals` (options: `--url`, `--only F1,C3`, `--category flow|jev|gemini`, `--no-judge`, `--concurrency`). Cases live in `evals/cases.py`; results are saved to `evals/results/`. The same evaluation is served over HTTP by `evals/api.py`: `GET /eval/cases`, `POST /eval/run` (body: `only`, `category`, `judge`, `concurrency`; starts a background run and returns its `run_id`), `GET /eval/runs` and `GET /eval/runs/{run_id}` (status, progress, summary, results).

LangChain/LangSmith evaluation of the agent itself: `python -m evals.langchain_eval` runs `main.agent` on the Gemini cases with `langsmith.aevaluate`, scored by rule checks (tool calls, forbidden tools, required words) and an LLM judge (LangChain chat model with structured output). Results go to a LangSmith experiment when the API key can use datasets; otherwise (or with `--local`) it runs in memory.

Environment variables (set in `.env` locally or in the host's settings; never commit them): `GOOGLE_API_KEY`, `OPENROUTER_API_KEY` (Jev via OpenRouter; or `TYPESAFE_API_KEY`), `CORS_ORIGINS`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.

## main.py

```python
﻿import asyncio
import logging
import os
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
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

Use the appropriate tool whenever necessary.

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

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

Be friendly, conversational, and concise.
"""

@dynamic_prompt
def prompt_with_today(request: ModelRequest) -> str:
    # Today's date, so Gemini can resolve "tomorrow" or "this Saturday".
    return f"{system_prompt}\nToday is {date.today():%A %Y-%m-%d}."


agent = create_agent(
    model=init_chat_model(os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")),
    tools=tools,
    middleware=[prompt_with_today],
)

app = FastAPI(title="Restaurant Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
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
```

## flow.py

```python
"""The guided booking conversation, asked as Jev-style Choice questions.

Q1 cuisine (tool 1) -> Q2 restaurant -> Q3 date -> Q4 party size
-> Q5 time slot -> booking. If the user dislikes the options, or the day
is fully booked, the flow restarts at Q1. The server is stateless: `state`
travels with each request.
"""

import json
from datetime import date

from tools import (
    booking_dates,
    check_reservation_availability,
    free_times,
    get_restaurant_details,
    is_closed,
    search_restaurants_by_cuisine,
)

CUISINES = ["Indian", "Italian", "Chinese", "Mexican"]

PARTY_SIZES = [2, 4, 6]


def date_label(day: str) -> str:
    """'2026-10-29' -> 'Thu 29 Oct'."""
    return date.fromisoformat(day).strftime("%a %d %b")


def time_label(t: str) -> str:
    """'19:30' -> '7:30 PM'."""
    hour, minute = map(int, t.split(":"))
    return f"{(hour - 1) % 12 + 1}:{minute:02d} {'PM' if hour >= 12 else 'AM'}"

START = {"asking": "cuisine"}

RESTART_LABEL = "None of these - start over"
RESTART_DESCRIPTION = (
    "The user dislikes these options, wants something different, "
    "or wants to start over"
)


def _choice(key: str, label: str, description: str) -> dict:
    # `message` is what the frontend sends when the user clicks the choice.
    return {"key": key, "label": label, "description": description, "message": label}


def _question(qid: str, text: str, choices: list[dict], restartable: bool = True) -> dict:
    if restartable:
        choices = choices + [
            _choice("restart", RESTART_LABEL, RESTART_DESCRIPTION)
        ]
    return {"id": qid, "type": "choice", "text": text, "choices": choices}


def restaurants_for(cuisine: str) -> list[dict]:
    return search_restaurants_by_cuisine.invoke({"cuisine_type": cuisine})


def question_for(state: dict) -> dict:
    """The Jev-format question the user is being asked right now."""
    asking = state.get("asking", "cuisine")

    if asking == "restaurant":
        cuisine = state["cuisine"]
        return _question(
            "restaurant",
            f"Which {cuisine} restaurant would you like?",
            # Rating and price in the description, so "the highest rated one"
            # or "the cheapest" can be matched.
            [
                _choice(r["name"], r["name"],
                        f"The restaurant {r['name']} (rated {r['rating']}, price {r['price']})")
                for r in restaurants_for(cuisine)
            ],
        )

    if asking == "date":
        restaurant = state["restaurant"]
        open_days = [d for d in booking_dates() if not is_closed(restaurant, d)]
        return _question(
            "date",
            f"When would you like to book {restaurant}?",
            [_choice(d, date_label(d), f"The date {date_label(d)} ({d})") for d in open_days],
        )

    if asking == "party":
        return _question(
            "party",
            "How many people?",
            [_choice(str(n), f"{n} people", f"A party of {n}") for n in PARTY_SIZES],
        )

    if asking == "time":
        times = free_times(state["restaurant"], state["date"], state["party"])
        return _question(
            "time",
            f"What time on {date_label(state['date'])}?",
            [_choice(t, time_label(t), f"The time slot {time_label(t)} ({t})") for t in times],
        )

    return _question(
        "cuisine",
        "What cuisine would you like?",
        [_choice(c.lower(), c, f"{c} cuisine") for c in CUISINES],
        restartable=False,
    )


def match_label(question: dict, message: str) -> str | None:
    """Fallback when Jev is unavailable: match the text against the choices."""
    text = message.strip().lower()
    for c in question["choices"]:
        if text in (c["label"].lower(), c["key"].lower()):
            return c["key"]
    return None


def _run(tool, **args) -> dict:
    """Run a tool directly and package it the way the frontend renders tool results."""
    out = tool.invoke(args)
    return {
        "name": tool.name,
        "args": args,
        "output": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False),
    }


OTHER_REPLY = (
    "I can only help with restaurant bookings here. "
    "Pick one of the options below, or tell me which one you want."
)


def step(state: dict, choice: str | None) -> dict:
    """Advance the flow by one answer.

    Returns {state, tool, reply, notice}. `tool` is the tool call Jev's
    decision triggers (or None), `reply` is the line shown with it, and
    `notice` is a line shown before the next question.
    """
    asking = state.get("asking", "cuisine")
    stay = {"state": state, "tool": None, "reply": OTHER_REPLY, "notice": None}

    if choice in (None, "other"):
        return stay

    if choice == "restart" and asking != "cuisine":
        return {
            "state": dict(START),
            "tool": None,
            "reply": "No problem, let's begin again!",
            "notice": "Okay, let's start over.",
        }

    if asking == "cuisine":
        cuisine = next((c for c in CUISINES if c.lower() == choice), None)
        if cuisine is None:
            return stay
        return {
            "state": {"asking": "restaurant", "cuisine": cuisine},
            "tool": _run(search_restaurants_by_cuisine, cuisine_type=cuisine),
            "reply": f"Here are some {cuisine} restaurants you might like.",
            "notice": None,
        }

    if asking == "restaurant":
        return {
            "state": {**state, "asking": "date", "restaurant": choice},
            "tool": _run(get_restaurant_details, restaurant_name=choice),
            "reply": f"Here are the details for {choice}.",
            "notice": None,
        }

    if asking == "date":
        return {
            "state": {**state, "asking": "party", "date": choice},
            "tool": None,
            "reply": f"Great, {date_label(choice)} it is.",
            "notice": None,
        }

    if asking == "party":
        party = int(choice)
        call = _run(
            check_reservation_availability,
            restaurant_name=state["restaurant"],
            date=state["date"],
            party_size=party,
        )
        day = date_label(state["date"])
        if "has availability" not in call["output"]:
            return {
                "state": dict(START),
                "tool": call,
                "reply": f"Sorry, {state['restaurant']} is fully booked for {party} on {day}.",
                "notice": "Sorry, there's no table that day. Let's start over.",
            }
        return {
            "state": {**state, "asking": "time", "party": party},
            "tool": call,
            "reply": f"{state['restaurant']} has tables for {party} on {day}.",
            "notice": None,
        }

    if asking == "time":
        call = _run(
            check_reservation_availability,
            restaurant_name=state["restaurant"],
            date=state["date"],
            party_size=state["party"],
            time=choice,
        )
        booked = "has availability" in call["output"]
        when = f"{date_label(state['date'])} at {time_label(choice)}"
        return {
            "state": dict(START),
            "tool": call,
            "reply": (
                f"Good news, {state['restaurant']} has a table for {state['party']} on {when}."
                if booked
                else f"Sorry, {state['restaurant']} is fully booked on {when}."
            ),
            "notice": (
                "Good news, that table is available! Want to look for another?"
                if booked
                else "Sorry, that table isn't available. Let's start over."
            ),
        }

    return {"state": dict(START), "tool": None, "reply": OTHER_REPLY, "notice": None}
```

## jev_decision.py

```python
import asyncio
import logging
import os
from datetime import date

from langsmith import traceable
from typesafe_sdk import AsyncTypeSafeClient, Choice

logger = logging.getLogger(__name__)


def _jev_client() -> AsyncTypeSafeClient:
    # Jev is also served through OpenRouter; use it when that key is set.
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return AsyncTypeSafeClient(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api",
            model=os.getenv("JEV_MODEL", "jev-latest"),
        )
    return AsyncTypeSafeClient()


@traceable(name="Jev Interpret Answer")
async def interpret_answer(
    question: dict, user_answer: str, history: list[dict] | None = None
) -> dict:
    """Ask Jev which choice of `question` the user's answer selects."""
    criteria = {c["key"]: c["description"] for c in question["choices"]}
    criteria["other"] = (
        "Anything else: a new question, or something that is not one of the choices"
    )
    conversation = "\n".join(
        f"{m['role']}: {m['content']}" for m in (history or [])
    )

    async with _jev_client() as client:
        response = await client.system_one(
            state={
                # So answers like "tomorrow" or "this Saturday" can be resolved.
                "today": f"{date.today():%A %Y-%m-%d}",
                "conversation_so_far": conversation,
                "question_asked": question["text"],
                "user_answer": user_answer,
            },
            questions={
                "answer": Choice(
                    instructions=(
                        "The assistant asked the user the question in state. "
                        "Decide which choice the user's answer selects."
                    ),
                    criteria=criteria,
                )
            },
        )

    answer = response.answers["answer"]
    return {
        "choice": answer.choice,
        "confidence": getattr(answer, "confidence", None),
        "available": True,
    }


async def safe_interpret_answer(
    question: dict, user_answer: str, history: list[dict] | None = None
) -> dict:
    try:
        return await asyncio.wait_for(
            interpret_answer(question, user_answer, history),
            timeout=10,
        )
    except Exception:
        logger.exception("Jev interpretation failed")
        return {"choice": None, "confidence": None, "available": False}
```

## tools.py

```python
from datetime import date, timedelta


from langchain_core.tools import tool

# ============================================================
# TOOL 1: Search Restaurants by Cuisine
# ============================================================

@tool
def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """
    Search for restaurants by cuisine type.
    Supported cuisines: Indian, Italian, Chinese, Mexican.
    """

    restaurants = {
        "indian": [
            {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
            {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
            {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"}
        ],
        "italian": [
            {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
            {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"}
        ],
        "chinese": [
            {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
            {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
            {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"}
        ],
        "mexican": [
            {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
            {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Burrito House", "rating": 4.2, "price": "₹₹"}
        ]
    }

    cuisine_lower = cuisine_type.lower().strip()

    return restaurants.get(cuisine_lower, [])


# ============================================================
# TOOL 2: Get Restaurant Details
# ============================================================

@tool
def get_restaurant_details(restaurant_name: str) -> dict:
    """
    Get detailed information about a restaurant,
    including address, phone, hours and specialties.
    """

    details = {
        "Spice Palace": {
            "cuisine": "Indian",
            "rating": 4.5,
            "address": "123 Food Street, City Center",
            "phone": "9876543210",
            "hours": "11 AM - 11 PM",
            "speciality": "Butter Chicken, Biryani"
        },

        "Pizza Bella": {
            "cuisine": "Italian",
            "rating": 4.4,
            "address": "456 Dine Avenue, Downtown",
            "phone": "9123456789",
            "hours": "12 PM - 10:30 PM",
            "speciality": "Margherita Pizza, Tiramisu"
        },

        "Dragon House": {
            "cuisine": "Chinese",
            "rating": 4.5,
            "address": "789 Noodle Lane, Market Area",
            "phone": "9111222333",
            "hours": "11:30 AM - 11 PM",
            "speciality": "Hakka Noodles, Fried Rice"
        },

        "Taco Fiesta": {
            "cuisine": "Mexican",
            "rating": 4.4,
            "address": "321 Spice Row, Hip District",
            "phone": "9222333444",
            "hours": "1 PM - 12 AM",
            "speciality": "Fish Tacos, Guacamole"
        },

        "Curry House": {
            "cuisine": "Indian",
            "rating": 4.3,
            "address": "555 Flavor Street, Old Town",
            "phone": "9333444555",
            "hours": "11 AM - 10 PM",
            "speciality": "Paneer Tikka, Dal Makhani"
        },

        "Taj Flavors": {
            "cuisine": "Indian",
            "rating": 4.7,
            "address": "999 Taste Lane, Premium Zone",
            "phone": "9444555666",
            "hours": "12 PM - 11 PM",
            "speciality": "Biryani, Tandoori Chicken"
        },

        "Golden Wok": {
            "cuisine": "Chinese",
            "rating": 4.8,
            "address": "777 Wok Street, Central Market",
            "phone": "9555666777",
            "hours": "11 AM - 11 PM",
            "speciality": "Szechuan Noodles, Dim Sum"
        },

        "Pasta Dreams": {
            "cuisine": "Italian",
            "rating": 4.6,
            "address": "12 Olive Road, Riverside",
            "phone": "9666777888",
            "hours": "12 PM - 11 PM",
            "speciality": "Truffle Pasta, Lasagna"
        },

        "Roma Kitchen": {
            "cuisine": "Italian",
            "rating": 4.2,
            "address": "34 Piazza Lane, Old Town",
            "phone": "9777888999",
            "hours": "11:30 AM - 10:30 PM",
            "speciality": "Risotto, Wood-fired Pizza"
        },

        "Ming's Kitchen": {
            "cuisine": "Chinese",
            "rating": 4.3,
            "address": "56 Lantern Street, Chinatown",
            "phone": "9888999000",
            "hours": "12 PM - 11 PM",
            "speciality": "Manchurian, Hot and Sour Soup"
        },

        "Salsa Nights": {
            "cuisine": "Mexican",
            "rating": 4.6,
            "address": "78 Fiesta Avenue, Hip District",
            "phone": "9999000111",
            "hours": "2 PM - 12:30 AM",
            "speciality": "Enchiladas, Churros"
        },

        "Burrito House": {
            "cuisine": "Mexican",
            "rating": 4.2,
            "address": "90 Cantina Road, Market Area",
            "phone": "9000111222",
            "hours": "11 AM - 10 PM",
            "speciality": "Loaded Burritos, Nachos"
        }
    }

    return details.get(
        restaurant_name.strip(),
        {"error": f"Restaurant '{restaurant_name}' not found"}
    )


# ============================================================
# TOOL 3: Check Reservation Availability
# ============================================================

# Bookings are taken for the next BOOKING_DAYS days, starting tomorrow.
BOOKING_DAYS = 7

# Each restaurant: the weekday it is closed (None = open every day) and
# its time slots (24h "HH:MM") with the seats still free in each slot.
# On Saturdays and Sundays, dinner slots (19:00 and later) have half the seats.
RESERVATION_SLOTS = {
    "Spice Palace": ("Monday", {"12:00": 8, "13:30": 4, "19:00": 2, "20:30": 6}),
    "Curry House": ("Tuesday", {"12:00": 6, "13:30": 0, "19:00": 4, "20:30": 2}),
    "Taj Flavors": ("Monday", {"12:30": 4, "14:00": 2, "19:30": 0, "21:00": 6}),
    "Pizza Bella": ("Wednesday", {"12:30": 6, "14:00": 6, "19:00": 0, "20:30": 4}),
    "Pasta Dreams": ("Monday", {"12:30": 2, "14:00": 4, "19:30": 6, "21:00": 0}),
    "Roma Kitchen": ("Thursday", {"12:00": 8, "13:30": 2, "19:00": 4, "20:30": 0}),
    "Dragon House": ("Tuesday", {"12:00": 4, "13:30": 6, "19:00": 2, "21:00": 8}),
    "Ming's Kitchen": (None, {"12:30": 6, "14:00": 4, "19:30": 4, "21:00": 2}),
    "Golden Wok": ("Wednesday", {"12:00": 0, "13:30": 2, "19:00": 6, "20:30": 4}),
    "Taco Fiesta": (None, {"13:00": 8, "15:00": 6, "19:00": 4, "22:00": 6}),
    "Salsa Nights": ("Monday", {"14:00": 2, "17:00": 4, "20:00": 0, "22:30": 0}),
    "Burrito House": ("Sunday", {"12:00": 4, "14:00": 0, "18:30": 2, "20:00": 2}),
}


def booking_dates() -> list[str]:
    """The dates (YYYY-MM-DD) open for booking."""
    today = date.today()
    return [(today + timedelta(days=i)).isoformat() for i in range(1, BOOKING_DAYS + 1)]


def is_closed(restaurant_name: str, day: str) -> bool:
    closed_on, _ = RESERVATION_SLOTS[restaurant_name]
    return date.fromisoformat(day).strftime("%A") == closed_on


def seats_left(restaurant_name: str, day: str) -> dict[str, int]:
    """Free seats per time slot on `day` (empty if the restaurant is closed)."""
    if is_closed(restaurant_name, day):
        return {}
    _, slots = RESERVATION_SLOTS[restaurant_name]
    weekend = date.fromisoformat(day).weekday() >= 5
    return {
        t: seats // 2 if weekend and t >= "19:00" else seats
        for t, seats in slots.items()
    }


def free_times(restaurant_name: str, day: str, party_size: int) -> list[str]:
    """The time slots on `day` that can still seat `party_size` people."""
    return [t for t, seats in seats_left(restaurant_name, day).items() if seats >= party_size]


@tool
def check_reservation_availability(
    restaurant_name: str,
    date: str,
    party_size: int,
    time: str = ""
) -> str:
    """
    Check reservation availability at a restaurant for a date (YYYY-MM-DD)
    and number of people. Give a time (24h HH:MM) to check one slot, or
    leave it empty to list every time slot that can seat the party that day.
    Bookings are open for the next 7 days, starting tomorrow.
    Only call this once the user has told you both the date and the number
    of people; if either is missing, ask the user instead of guessing.
    """

    if restaurant_name not in RESERVATION_SLOTS:
        return f"Restaurant '{restaurant_name}' not found."

    dates = booking_dates()
    if date not in dates:
        return (
            f"No bookings for {date}: reservations are open "
            f"from {dates[0]} to {dates[-1]}."
        )

    if is_closed(restaurant_name, date):
        return f"✗ {restaurant_name} is closed on {date}."

    if time:
        seats = seats_left(restaurant_name, date)
        if time not in seats:
            return (
                f"✗ {restaurant_name} has no {time} slot. "
                f"Times: {', '.join(seats)}"
            )
        if seats[time] >= party_size:
            return (
                f"✓ {restaurant_name} has availability "
                f"for {party_size} people on {date} at {time}"
            )
        return (
            f"✗ {restaurant_name} is fully booked "
            f"for {party_size} people on {date} at {time}"
        )

    times = free_times(restaurant_name, date, party_size)
    if times:
        return (
            f"✓ {restaurant_name} has availability "
            f"for {party_size} people on {date} at {', '.join(times)}"
        )

    return (
        f"✗ {restaurant_name} is fully booked "
        f"for {party_size} people on {date}"
    )


tools = [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]
```

## evals/cases.py

```python
"""Evaluation cases for POST /chat.

Each case is one or more turns sent in order; `state` and history carry over
between turns like the frontend does. A turn's `expect` has a `jev` part
and/or a `gemini` part (see run_evals.py for the checks). Dates are built
from tools.booking_dates(), so the cases stay valid as the window moves.

Jev checks:    choice, next_question, next_choices, next_choices_lack,
               tools, no_tools, tool_output_has, reply_has
Gemini checks: tools, no_tools, reply_has, reply_lacks, judge
"""

from datetime import date

from flow import date_label
from tools import booking_dates


def _first(pred) -> str:
    return next(d for d in booking_dates() if pred(date.fromisoformat(d)))


def _weekday(*closed: int) -> str:
    """First Mon-Fri booking date that is not one of the `closed` weekdays (0 = Monday)."""
    return _first(lambda d: d.weekday() < 5 and d.weekday() not in closed)


def _long(day: str) -> str:
    """'2026-11-02' -> 'Monday 02 November'."""
    return date.fromisoformat(day).strftime("%A %d %B")


def build_cases() -> list[dict]:
    tomorrow = booking_dates()[0]
    monday = _first(lambda d: d.weekday() == 0)
    saturday = _first(lambda d: d.weekday() == 5)
    spice_day = _weekday(0)  # Spice Palace is closed on Mondays
    salsa_day = _weekday(0)
    taco_day = _weekday()
    taj_day = _weekday(0)

    italian = {"asking": "restaurant", "cuisine": "Italian"}
    taco = {"cuisine": "Mexican", "restaurant": "Taco Fiesta"}

    return [
        # ---------------- Guided flow (clicking the buttons) ----------------
        {
            "id": "F1", "category": "flow",
            "name": "Full booking: Indian -> Spice Palace -> weekday -> 2 -> 7 PM",
            "turns": [
                {"message": "Indian", "expect": {"jev": {
                    "choice": "indian", "next_question": "restaurant",
                    "tools": [{"name": "search_restaurants_by_cuisine", "args": {"cuisine_type": "Indian"}}],
                }}},
                {"message": "Spice Palace", "expect": {"jev": {
                    "choice": "Spice Palace", "next_question": "date",
                    "next_choices_lack": [date_label(monday)],  # closed on Mondays
                }}},
                {"message": date_label(spice_day), "expect": {
                    "jev": {"choice": spice_day, "next_question": "party", "no_tools": True},
                    # Party size is not known yet: Gemini must not guess it.
                    "gemini": {"no_tools": ["check_reservation_availability"],
                               "judge": "Asks how many people are in the party."},
                }},
                {"message": "2 people", "expect": {
                    "jev": {
                        "choice": "2", "next_question": "time",
                        "next_choices": ["12:00 PM", "1:30 PM", "7:00 PM", "8:30 PM"],
                        "tools": [{"name": "check_reservation_availability",
                                   "args": {"restaurant_name": "Spice Palace", "date": spice_day, "party_size": 2}}],
                    },
                    "gemini": {"tools": [{"name": "check_reservation_availability",
                                          "args": {"restaurant_name": "Spice Palace", "date": spice_day, "party_size": 2}}]},
                }},
                {"message": "7:00 PM", "expect": {
                    "jev": {
                        "choice": "19:00", "next_question": "cuisine",
                        "tools": [{"name": "check_reservation_availability", "args": {"time": "19:00"}}],
                        "tool_output_has": "has availability",
                    },
                    "gemini": {
                        "tools": [{"name": "check_reservation_availability",
                                   "args": {"restaurant_name": "Spice Palace", "date": spice_day,
                                            "party_size": 2, "time": "19:00"}}],
                        "judge": "Says a table for 2 is available at 7 PM (19:00).",
                    },
                }},
            ],
        },
        {
            "id": "F2", "category": "flow",
            "name": "Party of 6 at Salsa Nights on a weekday: no slot fits, start over",
            "state": {"asking": "party", "cuisine": "Mexican", "restaurant": "Salsa Nights", "date": salsa_day},
            "turns": [{"message": "6 people", "expect": {"jev": {
                "choice": "6", "next_question": "cuisine",
                "tool_output_has": "fully booked", "reply_has": ["fully booked"],
            }}}],
        },
        {
            "id": "F3", "category": "flow",
            "name": "Start over from the restaurant question",
            "state": italian,
            "turns": [{"message": "None of these - start over", "expect": {"jev": {
                "choice": "restart", "next_question": "cuisine", "no_tools": True,
            }}}],
        },
        {
            "id": "F4", "category": "flow",
            "name": "Saturday dinner has half the seats: 7 PM (1 seat) not offered for 2",
            "state": {"asking": "party", "cuisine": "Indian", "restaurant": "Spice Palace", "date": saturday},
            "turns": [{"message": "2 people", "expect": {"jev": {
                "choice": "2", "next_question": "time",
                "next_choices": ["12:00 PM", "1:30 PM", "8:30 PM"],
            }}}],
        },

        # ---------------- Jev: understanding typed answers ----------------
        {"id": "B1", "category": "jev", "name": "Food -> cuisine",
         "turns": [{"message": "I'm in the mood for pasta",
                    "expect": {"jev": {"choice": "italian", "next_question": "restaurant"}}}]},
        {"id": "B2", "category": "jev", "name": "Dish -> cuisine",
         "turns": [{"message": "something with tacos",
                    "expect": {"jev": {"choice": "mexican", "next_question": "restaurant"}}}]},
        {"id": "B3", "category": "jev", "name": "Unsupported cuisine stays on the question",
         "turns": [{"message": "Japanese please",
                    "expect": {"jev": {"choice": "other", "next_question": "cuisine", "no_tools": True}}}]},
        {"id": "B4", "category": "jev", "name": "'Highest rated' picks Pasta Dreams (4.6)",
         "state": italian,
         "turns": [{"message": "the highest rated one",
                    "expect": {"jev": {"choice": "Pasta Dreams", "next_question": "date"}}}]},
        {"id": "B5", "category": "jev", "name": "Restaurant from another cuisine is not a choice",
         "state": italian,
         "turns": [{"message": "Dragon House", "expect": {"jev": {"choice": "other", "next_question": "restaurant"}}}]},
        {"id": "B6", "category": "jev", "name": "Dislikes options -> restart",
         "state": italian,
         "turns": [{"message": "meh, none of these", "expect": {"jev": {"choice": "restart", "next_question": "cuisine"}}}]},
        {"id": "B7", "category": "jev", "name": "'Tomorrow' resolves to the first booking date",
         "state": {**taco, "asking": "date"},
         "turns": [{"message": "tomorrow", "expect": {"jev": {"choice": tomorrow, "next_question": "party"}}}]},
        {"id": "B8", "category": "jev", "name": "Date outside the window is not a choice",
         "state": {**taco, "asking": "date"},
         "turns": [{"message": "Christmas day please", "expect": {"jev": {"choice": "other", "next_question": "date"}}}]},
        {"id": "B9", "category": "jev", "name": "'Me and my wife' -> 2",
         "state": {**taco, "asking": "party", "date": taco_day},
         "turns": [{"message": "me and my wife", "expect": {"jev": {"choice": "2", "next_question": "time"}}}]},
        {"id": "B10", "category": "jev", "name": "Party of 5 is not rounded to 4 or 6",
         "state": {**taco, "asking": "party", "date": taco_day},
         "turns": [{"message": "five of us", "expect": {"jev": {"choice": "other", "next_question": "party"}}}]},
        {"id": "B11", "category": "jev", "name": "'Dinner at 7' -> 19:00 slot",
         "state": {**taco, "asking": "time", "date": taco_day, "party": 4},
         "turns": [{"message": "dinner at 7", "expect": {"jev": {
             "choice": "19:00", "next_question": "cuisine", "tool_output_has": "has availability"}}}]},
        {"id": "B12", "category": "jev", "name": "Off-topic stays on the question",
         "state": {**taco, "asking": "party", "date": taco_day},
         "turns": [{"message": "what's the weather today?", "expect": {"jev": {
             "choice": "other", "next_question": "party", "no_tools": True}}}]},

        # ---------------- Gemini: tool use and grounded answers ----------------
        {"id": "C1", "category": "gemini", "name": "Where can I get biryani?",
         "turns": [{"message": "Where can I get biryani?", "expect": {"gemini": {
             "tools": [{"name": "search_restaurants_by_cuisine", "args": {"cuisine_type": "Indian"}}],
             "reply_has": ["Spice Palace", "Taj Flavors"],
             "judge": "Recommends Spice Palace and Taj Flavors for biryani, and does not claim Curry House serves biryani.",
         }}}]},
        {"id": "C2", "category": "gemini", "name": "Where can I get dim sum?",
         "turns": [{"message": "Where can I get dim sum?", "expect": {"gemini": {"reply_has": ["Golden Wok"]}}}]},
        {"id": "C3", "category": "gemini", "name": "Sushi is not available",
         "turns": [{"message": "Where can I get sushi?", "expect": {"gemini": {
             "no_tools": ["check_reservation_availability"],
             "judge": "Says there is no sushi or Japanese restaurant it can offer, and does not invent one.",
         }}}]},
        {"id": "C4", "category": "gemini", "name": "Phone number lookup",
         "turns": [{"message": "What's Dragon House's phone number?", "expect": {"gemini": {
             "tools": [{"name": "get_restaurant_details", "args": {"restaurant_name": "Dragon House"}}],
             "reply_has": ["9111222333"],
         }}}]},
        {"id": "C5", "category": "gemini", "name": "Opening hours reasoning",
         "turns": [{"message": "Is Pizza Bella open at 11 AM?", "expect": {"gemini": {
             "tools": [{"name": "get_restaurant_details", "args": {"restaurant_name": "Pizza Bella"}}],
             "judge": "Says Pizza Bella is not open at 11 AM because it opens at 12 PM.",
         }}}]},
        {"id": "C6", "category": "gemini", "name": "Booking without date or party size",
         "turns": [{"message": "Book me a table at Spice Palace", "expect": {"gemini": {
             "no_tools": ["check_reservation_availability"],
             "judge": "Asks for the date and the number of people instead of guessing them.",
         }}}]},
        {"id": "C7", "category": "gemini", "name": "No invented prices",
         "turns": [{"message": "How much does a meal at Curry House cost?", "expect": {"gemini": {
             "judge": "Does not state any specific price amount; at most describes the price level (₹₹) or says it doesn't know.",
         }}}]},
        {"id": "C8", "category": "gemini", "name": "'Tomorrow' resolved to a real date",
         "turns": [{"message": "Can I get a table for 4 at Pizza Bella tomorrow?", "expect": {"gemini": {
             "tools": [{"name": "check_reservation_availability",
                        "args": {"restaurant_name": "Pizza Bella", "date": tomorrow, "party_size": 4}}],
         }}}]},
        {"id": "C9", "category": "gemini", "name": "Closed day",
         "turns": [{"message": f"Is there a table for 2 at Spice Palace on {_long(monday)}?", "expect": {"gemini": {
             "tools": [{"name": "check_reservation_availability",
                        "args": {"restaurant_name": "Spice Palace", "date": monday, "party_size": 2}}],
             "judge": "Says Spice Palace is closed on that day.",
         }}}]},
        {"id": "C10", "category": "gemini", "name": "Date outside the booking window",
         "turns": [{"message": "Can I book Ming's Kitchen for 2 on 2024-11-15?", "expect": {"gemini": {
             "judge": "Says that date cannot be booked because reservations are only open for the next few days; does not say the restaurant is fully booked.",
         }}}]},
        {"id": "C11", "category": "gemini", "name": "Specific time slot",
         "turns": [{"message": f"Is there a table for 6 at Taj Flavors at 9 PM on {_long(taj_day)}?", "expect": {"gemini": {
             "tools": [{"name": "check_reservation_availability",
                        "args": {"restaurant_name": "Taj Flavors", "date": taj_day, "party_size": 6, "time": "21:00"}}],
             "judge": "Says a table for 6 is available at 9 PM.",
         }}}]},
    ]
```

## evals/run_evals.py

```python
"""Evaluate the /chat backend: Jev's understanding and flow, Gemini's tool use and replies.

Run from backend/:
    python -m evals.run_evals                      # in-process, no server needed
    python -m evals.run_evals --url http://localhost:8000
    python -m evals.run_evals --only F1,C3 --no-judge
    python -m evals.run_evals --category gemini

Rule-based checks score tool calls, choices and keywords. `judge` checks ask an
LLM (JUDGE_MODEL, default MODEL) whether Gemini's reply meets a criterion.
Results are printed and saved to evals/results/<timestamp>.json.
Exit code is 1 if any check failed.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from evals.cases import build_cases

RESULTS_DIR = Path(__file__).parent / "results"

# The Gemini free tier allows 15 requests a minute; on a failed Gemini call
# (main.run_gemini reports it as available=False) wait and retry the turn.
RETRIES = 2
RETRY_WAIT_S = 40


# ---------------- checks ----------------

def _norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def _has_call(calls: list[dict], want: dict) -> bool:
    return any(
        c["name"] == want["name"]
        and all(_norm(c.get("args", {}).get(k)) == _norm(v) for k, v in want.get("args", {}).items())
        for c in calls
    )


def _calls(calls: list[dict]) -> str:
    return "; ".join(f"{c['name']}({c.get('args', {})})" for c in calls) or "none"


def _labels(question: dict) -> list[str]:
    return [c["label"] for c in question["choices"] if c["key"] != "restart"]


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, check: str, passed: bool, detail: str = ""):
        self.items.append({"check": check, "passed": bool(passed), "detail": detail})


def check_jev(jev: dict, exp: dict) -> list[dict]:
    c = Checks()
    calls = jev.get("tool_calls", [])

    if "choice" in exp:
        got = jev["understood"]["choice"] or "other"
        want = exp["choice"] if isinstance(exp["choice"], list) else [exp["choice"]]
        c.add("choice", got in want, f"got {got!r}, want {want} ({jev['understood']['method']})")
    if "next_question" in exp:
        got = jev["next_question"]["id"]
        c.add("next_question", got == exp["next_question"], f"got {got!r}, want {exp['next_question']!r}")
    if "next_choices" in exp:
        got = _labels(jev["next_question"])
        c.add("next_choices", got == exp["next_choices"], f"got {got}, want {exp['next_choices']}")
    if "next_choices_lack" in exp:
        bad = [x for x in exp["next_choices_lack"] if x in _labels(jev["next_question"])]
        c.add("next_choices_lack", not bad, f"should not offer {bad}" if bad else "")
    for want in exp.get("tools", []):
        c.add(f"tool:{want['name']}", _has_call(calls, want), f"want {want.get('args', {})}, got {_calls(calls)}")
    if exp.get("no_tools"):
        c.add("no_tools", not calls, f"got {_calls(calls)}")
    if "tool_output_has" in exp:
        ok = any(exp["tool_output_has"] in t.get("output", "") for t in calls)
        c.add("tool_output_has", ok, f"want {exp['tool_output_has']!r} in {[t.get('output') for t in calls]}")
    for text in exp.get("reply_has", []):
        c.add(f"reply_has:{text}", _norm(text) in _norm(jev["reply"]), f"reply: {jev['reply']!r}")
    return c.items


class Verdict(BaseModel):
    passed: bool
    reason: str


async def check_gemini(google: dict, exp: dict, message: str, judge) -> list[dict]:
    c = Checks()
    if not google.get("available", True):
        c.add("available", False, "Gemini call failed after retries (often the free-tier rate limit; see the log)")
        return c.items

    calls = google.get("tool_calls", [])
    reply = google.get("reply", "")

    for want in exp.get("tools", []):
        c.add(f"tool:{want['name']}", _has_call(calls, want), f"want {want.get('args', {})}, got {_calls(calls)}")
    for name in exp.get("no_tools", []):
        used = [t for t in calls if t["name"] == name]
        c.add(f"no_tool:{name}", not used, f"called {_calls(used)}")
    for text in exp.get("reply_has", []):
        c.add(f"reply_has:{text}", _norm(text) in _norm(reply))
    for text in exp.get("reply_lacks", []):
        c.add(f"reply_lacks:{text}", _norm(text) not in _norm(reply))
    if "judge" in exp and judge is not None:
        prompt = (
            "You are grading a restaurant assistant's reply.\n\n"
            f"User message: {message}\n\n"
            f"Tool calls it made: {json.dumps(calls, ensure_ascii=False)}\n\n"
            f"Assistant reply: {reply}\n\n"
            f"Criterion: {exp['judge']}\n\n"
            "Pass only if the reply clearly meets the criterion."
        )
        for attempt in range(RETRIES + 1):
            try:
                v = await judge.ainvoke(prompt)
                c.add("judge", v.passed, f"{exp['judge']} -> {v.reason}")
                break
            except Exception as e:
                if attempt == RETRIES:
                    c.add("judge", False, f"judge error: {str(e)[:200]}")
                else:
                    await asyncio.sleep(RETRY_WAIT_S)
    return c.items


# ---------------- running ----------------

async def run_case(client: httpx.AsyncClient, case: dict, judge) -> dict:
    state = case.get("state")
    history: list[dict] = []
    gemini_history: list[dict] = []
    turns = []

    for turn in case["turns"]:
        message, exp = turn["message"], turn.get("expect", {})
        out = {"message": message, "jev": [], "gemini": []}
        started = time.perf_counter()
        try:
            for attempt in range(RETRIES + 1):
                r = await client.post("/chat", json={
                    "message": message, "history": history,
                    "gemini_history": gemini_history, "state": state,
                })
                r.raise_for_status()
                body = r.json()
                gemini_ok = (body.get("google") or {}).get("available", True)
                if gemini_ok or "gemini" not in exp or attempt == RETRIES:
                    break
                await asyncio.sleep(RETRY_WAIT_S)
        except Exception as e:
            out["latency_s"] = round(time.perf_counter() - started, 2)
            out["jev"] = out["gemini"] = [{"check": "request", "passed": False, "detail": repr(e)}]
            turns.append(out)
            break
        out["latency_s"] = round(time.perf_counter() - started, 2)

        jev, google = body["jev"], body.get("google") or {}
        if "jev" in exp:
            out["jev"] = check_jev(jev, exp["jev"])
        if "gemini" in exp:
            out["gemini"] = await check_gemini(google, exp["gemini"], message, judge)
        out["jev_reply"], out["gemini_reply"] = jev["reply"], google.get("reply", "")
        out["jev_available"] = jev["available"]
        turns.append(out)

        state = body["state"]
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": jev["reply"]}]
        gemini_history += [{"role": "user", "content": message},
                           {"role": "assistant", "content": google.get("reply", "")}]

    def side(key):
        items = [i for t in turns for i in t[key]]
        return None if not items else all(i["passed"] for i in items)

    return {
        "id": case["id"], "category": case["category"], "name": case["name"],
        "jev_passed": side("jev"), "gemini_passed": side("gemini"),
        "latency_s": round(sum(t["latency_s"] for t in turns), 2),
        "turns": turns,
    }


def select_cases(cases: list[dict], only: list[str] | None = None, category: str | None = None) -> list[dict]:
    if only:
        ids = {i.strip().upper() for i in only}
        cases = [c for c in cases if c["id"] in ids]
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


async def evaluate(client: httpx.AsyncClient, cases: list[dict], judge, concurrency: int = 2,
                   on_done=None) -> list[dict]:
    """Run `cases` through POST /chat on `client`; results come back in case order.
    `on_done(result)` is called as each case finishes."""
    limit = asyncio.Semaphore(concurrency)

    async def one(case):
        async with limit:
            r = await run_case(client, case, judge)
            if on_done:
                on_done(r)
            return r

    return list(await asyncio.gather(*(one(c) for c in cases)))


def summarize(results: list[dict]) -> dict:
    def score(who):
        scored = [r for r in results if r[f"{who}_passed"] is not None]
        passed = sum(r[f"{who}_passed"] for r in scored)
        return {"passed": passed, "total": len(scored)}

    categories = {}
    for cat in dict.fromkeys(r["category"] for r in results):
        rows = [r for r in results if r["category"] == cat]
        ok = sum(all(r[k] in (True, None) for k in ("jev_passed", "gemini_passed")) for r in rows)
        categories[cat] = {"passed": ok, "total": len(rows)}

    return {
        "jev": score("jev"),
        "gemini": score("gemini"),
        "categories": categories,
        "failed_cases": [r["id"] for r in results if False in (r["jev_passed"], r["gemini_passed"])],
        "jev_fallback_turns": sum(1 for r in results for t in r["turns"] if t.get("jev_available") is False),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 1) if results else 0,
    }


def save_results(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _mark(passed) -> str:
    return "  -  " if passed is None else ("PASS " if passed else "FAIL ")


def report(results: list[dict]) -> None:
    print(f"\n{'ID':<5}{'JEV':<6}{'GEMINI':<8}{'TIME':>6}  NAME")
    for r in results:
        print(f"{r['id']:<5}{_mark(r['jev_passed']):<6}{_mark(r['gemini_passed']):<8}{r['latency_s']:>5}s  {r['name']}")
        for i, t in enumerate(r["turns"], 1):
            for who in ("jev", "gemini"):
                for c in t[who]:
                    if not c["passed"]:
                        print(f"        turn {i} {who} {c['check']}: {c['detail']}")

    s = summarize(results)
    print("\nSUMMARY")
    for who in ("jev", "gemini"):
        if s[who]["total"]:
            print(f"  {who:<7} {s[who]['passed']}/{s[who]['total']} cases passed "
                  f"({100 * s[who]['passed'] / s[who]['total']:.0f}%)")
    for cat, c in s["categories"].items():
        print(f"  {cat:<7} {c['passed']}/{c['total']} cases fully passed")
    if s["jev_fallback_turns"]:
        print(f"  note: Jev was unavailable on {s['jev_fallback_turns']} turn(s); "
              "the text-match fallback was scored instead")
    print(f"  avg latency per case: {s['avg_latency_s']}s")


def make_judge():
    from langchain.chat_models import init_chat_model

    name = os.getenv("JUDGE_MODEL") or os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")
    return init_chat_model(name).with_structured_output(Verdict)


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", help="evaluate a running server instead of the app in-process")
    p.add_argument("--only", help="comma-separated case ids, e.g. F1,C3")
    p.add_argument("--category", help="flow, jev or gemini")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-judged checks")
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()

    cases = select_cases(build_cases(), args.only.split(",") if args.only else None, args.category)
    if not cases:
        print("No cases selected.")
        return 1

    if args.url:
        client = httpx.AsyncClient(base_url=args.url, timeout=120)
    else:
        from main import app
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://eval", timeout=120)

    judge = None if args.no_judge else make_judge()

    def progress(r):
        print(f"  done {r['id']:<4} jev {_mark(r['jev_passed'])} gemini {_mark(r['gemini_passed'])}", flush=True)

    print(f"Running {len(cases)} case(s) against {args.url or 'the app in-process'}...")
    async with client:
        results = await evaluate(client, cases, judge, args.concurrency, progress)

    report(results)
    print(f"\nSaved {save_results(results)}")

    failed = any(r[k] is False for r in results for k in ("jev_passed", "gemini_passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

## evals/api.py

```python
"""HTTP endpoints for the evaluation in run_evals.py.

GET  /eval/cases            the test cases
POST /eval/run              start a run in the background -> {run_id}
GET  /eval/runs             runs since the server started, newest first
GET  /eval/runs/{run_id}    status, progress, summary and per-case results

A run calls this same app's POST /chat in-process. Runs are kept in memory
(lost on restart); each finished run is also saved to evals/results/.
"""

import asyncio
import logging
import uuid
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from evals.cases import build_cases
from evals.run_evals import evaluate, make_judge, save_results, select_cases, summarize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["evaluation"])

RUNS: dict[str, dict] = {}
_tasks: set[asyncio.Task] = set()  # keep running tasks referenced


class RunRequest(BaseModel):
    only: list[str] | None = None  # case ids, e.g. ["F1", "C3"]
    category: str | None = None  # flow | jev | gemini
    judge: bool = True  # run the LLM-judged checks
    concurrency: int = Field(2, ge=1, le=5)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@router.get("/cases")
async def list_cases():
    return [
        {"id": c["id"], "category": c["category"], "name": c["name"], "turns": len(c["turns"])}
        for c in build_cases()
    ]


@router.post("/run", status_code=202)
async def start_run(req: RunRequest, request: Request):
    if any(r["status"] == "running" for r in RUNS.values()):
        # One at a time: parallel runs would share the Gemini rate limit.
        raise HTTPException(409, "An evaluation is already running.")

    cases = select_cases(build_cases(), req.only, req.category)
    if not cases:
        raise HTTPException(400, "No cases selected.")

    run_id = uuid.uuid4().hex[:8]
    run = RUNS[run_id] = {
        "run_id": run_id, "status": "running", "started_at": _now(), "finished_at": None,
        "options": req.model_dump(), "total": len(cases), "done": 0,
        "summary": None, "results": [], "saved_to": None, "error": None,
    }
    task = asyncio.create_task(_run(run, cases, req, request.app))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"run_id": run_id, "status": "running", "total": len(cases), "poll": f"/eval/runs/{run_id}"}


async def _run(run: dict, cases: list[dict], req: RunRequest, app) -> None:
    def on_done(result):
        run["done"] += 1
        run["results"].append(result)

    try:
        judge = make_judge() if req.judge else None
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=120) as client:
            results = await evaluate(client, cases, judge, req.concurrency, on_done)
        run["results"] = results  # in case order
        run["summary"] = summarize(results)
        run["saved_to"] = str(save_results(results))
        run["status"] = "finished"
    except Exception as e:
        logger.exception("Evaluation run %s failed", run["run_id"])
        run["status"], run["error"] = "failed", repr(e)
    finally:
        run["finished_at"] = _now()


@router.get("/runs")
async def list_runs():
    return [
        {k: v for k, v in r.items() if k != "results"}
        for r in sorted(RUNS.values(), key=lambda r: r["started_at"], reverse=True)
    ]


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Unknown run id.")
    return RUNS[run_id]
```

## evals/langchain_eval.py

```python
"""Evaluate the LangChain agent (main.agent) with LangSmith.

The Gemini cases in evals/cases.py become a LangSmith dataset. langsmith.aevaluate
runs the agent on every example and scores it with:
  - rule_checks: the expected tool calls (name + args), no forbidden tools,
    and the words the reply must contain
  - llm_judge: a LangChain chat model (JUDGE_MODEL, default MODEL) with
    structured output grades the reply against the case's criterion
Each run is a LangSmith experiment (LANGSMITH_API_KEY in .env) and is also
printed here. The dataset is per day because the case dates are relative.

Run from backend/:
    python -m evals.langchain_eval
    python -m evals.langchain_eval --only C1,C6 --no-judge
    python -m evals.langchain_eval --refresh      # re-create today's dataset after editing cases
    python -m evals.langchain_eval --no-upload    # don't save the experiment to LangSmith
    python -m evals.langchain_eval --local        # no LangSmith dataset at all (in-memory examples)

A tracing-only LangSmith key cannot use datasets; the run then falls back to --local.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, aevaluate
from langsmith.schemas import Example
from langsmith.utils import LangSmithError

from evals.cases import build_cases
from evals.run_evals import RETRIES, RETRY_WAIT_S, _has_call, _norm, make_judge
from main import _text, agent

DATASET_PREFIX = "restaurant-agent-gemini"


# ---------------- dataset ----------------

def dataset_examples() -> list[dict]:
    """One example per single-turn Gemini case: the question in, the expectations as reference."""
    return [
        {
            "inputs": {"question": case["turns"][0]["message"]},
            "outputs": case["turns"][0]["expect"]["gemini"],
            "metadata": {"case_id": case["id"], "name": case["name"]},
        }
        for case in build_cases()
        if case["category"] == "gemini" and len(case["turns"]) == 1
    ]


def local_examples() -> list[Example]:
    """The same examples, in memory: for --local runs (no LangSmith dataset needed)."""
    dataset_id = uuid.uuid4()
    return [Example(id=uuid.uuid4(), dataset_id=dataset_id, **e) for e in dataset_examples()]


def sync_dataset(client: Client, refresh: bool = False) -> str:
    name = f"{DATASET_PREFIX}-{date.today()}"
    if refresh and client.has_dataset(dataset_name=name):
        client.delete_dataset(dataset_name=name)
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="Gemini restaurant-agent cases from evals/cases.py")
        client.create_examples(dataset_id=ds.id, examples=dataset_examples())
    return name


# ---------------- target ----------------

async def run_agent(inputs: dict) -> dict:
    """Run the LangChain agent on one question; return its reply and tool calls."""
    for attempt in range(RETRIES + 1):
        try:
            result = await agent.ainvoke({"messages": [{"role": "user", "content": inputs["question"]}]})
            break
        except Exception:
            # Usually the Gemini free-tier rate limit (15 requests a minute).
            if attempt == RETRIES:
                raise
            await asyncio.sleep(RETRY_WAIT_S)

    out = result["messages"]
    return {
        "reply": _text(out[-1].content) if out else "",
        "tool_calls": [
            {"name": c["name"], "args": c.get("args", {})}
            for m in out
            for c in (getattr(m, "tool_calls", None) or [])
        ],
    }


# ---------------- evaluators ----------------

def rule_checks(outputs: dict, reference_outputs: dict) -> dict:
    calls, reply = outputs["tool_calls"], outputs["reply"]
    results = []

    if want := reference_outputs.get("tools"):
        missing = [w for w in want if not _has_call(calls, w)]
        results.append({"key": "tool_calls", "score": int(not missing),
                        "comment": f"missing {missing}; got {calls}" if missing else ""})
    if forbidden := reference_outputs.get("no_tools"):
        used = [c for c in calls if c["name"] in forbidden]
        results.append({"key": "no_forbidden_tools", "score": int(not used),
                        "comment": f"called {used}" if used else ""})
    if words := reference_outputs.get("reply_has"):
        missing = [w for w in words if _norm(w) not in _norm(reply)]
        results.append({"key": "mentions", "score": int(not missing),
                        "comment": f"reply lacks {missing}" if missing else ""})
    return {"results": results}


def make_llm_judge(judge):
    async def llm_judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        criterion = reference_outputs.get("judge")
        if not criterion:
            return {"results": []}
        prompt = (
            "You are grading a restaurant assistant's reply.\n\n"
            f"User message: {inputs['question']}\n\n"
            f"Tool calls it made: {json.dumps(outputs['tool_calls'], ensure_ascii=False)}\n\n"
            f"Assistant reply: {outputs['reply']}\n\n"
            f"Criterion: {criterion}\n\n"
            "Pass only if the reply clearly meets the criterion."
        )
        for attempt in range(RETRIES + 1):
            try:
                v = await judge.ainvoke(prompt)
                return {"key": "judge", "score": int(v.passed), "comment": v.reason}
            except Exception:
                if attempt == RETRIES:
                    raise
                await asyncio.sleep(RETRY_WAIT_S)

    return llm_judge


# ---------------- running ----------------

def _scores(row) -> list:
    res = row["evaluation_results"]["results"]
    return [r for r in res if r.score is not None]


def report(rows: list) -> bool:
    print(f"\n{'ID':<5}{'RESULT':<8}NAME")
    all_passed = True
    for row in sorted(rows, key=lambda r: int(r["example"].metadata["case_id"][1:])):
        meta, run, scores = row["example"].metadata, row["run"], _scores(row)
        passed = run.error is None and all(s.score == 1 for s in scores)
        all_passed &= passed
        print(f"{meta['case_id']:<5}{'PASS' if passed else 'FAIL':<8}{meta['name']}")
        if run.error:
            print(f"        error: {run.error[:200]}")
        for s in scores:
            if s.score != 1:
                print(f"        {s.key}: {s.comment}")

    print("\nSCORES")
    keys = dict.fromkeys(s.key for row in rows for s in _scores(row))
    for key in keys:
        vals = [s.score for row in rows for s in _scores(row) if s.key == key]
        print(f"  {key:<20} {sum(vals)}/{len(vals)}")
    ok = sum(1 for row in rows if row["run"].error is None and all(s.score == 1 for s in _scores(row)))
    print(f"  {'cases passed':<20} {ok}/{len(rows)}")
    return all_passed


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="comma-separated case ids, e.g. C1,C6")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument("--refresh", action="store_true", help="re-create today's dataset from evals/cases.py")
    p.add_argument("--no-upload", action="store_true", help="don't save the experiment to LangSmith")
    p.add_argument("--local", action="store_true",
                   help="use in-memory examples and don't upload (no LangSmith dataset access needed)")
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()

    client = Client()
    dataset = None
    if not args.local:
        try:
            dataset = sync_dataset(client, args.refresh)
        except LangSmithError as e:
            # e.g. a tracing-only API key: it cannot read or create datasets.
            print(f"LangSmith dataset unavailable ({str(e)[-120:]}); running locally instead.")
    if dataset is None:
        args.no_upload = True

    examples = list(client.list_examples(dataset_name=dataset)) if dataset else local_examples()
    if args.only:
        ids = {i.strip().upper() for i in args.only.split(",")}
        examples = [e for e in examples if e.metadata.get("case_id") in ids]
    if not examples:
        print("No cases selected.")
        return 1

    evaluators = [rule_checks]
    if not args.no_judge:
        evaluators.append(make_llm_judge(make_judge()))

    print(f"Evaluating the LangChain agent on {len(examples)} example(s) "
          f"{f'from dataset {dataset!r}' if dataset else '(local, not uploaded)'}...")
    results = await aevaluate(
        run_agent,
        data=examples,
        evaluators=evaluators,
        experiment_prefix="restaurant-agent",
        metadata={"model": os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")},
        max_concurrency=args.concurrency,
        upload_results=not args.no_upload,
        client=client,
    )
    rows = [row async for row in results]

    passed = report(rows)
    if not args.no_upload:
        print(f"\nLangSmith experiment: {results.experiment_name} (dataset '{dataset}')")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

## requirements.txt

```text
fastapi
uvicorn[standard]
langchain>=1.0
langchain-core>=1.0
langchain-google-genai
pydantic
langsmith
python-dotenv
typesafe-sdk
httpx
```

## .gitignore

```text
.env
__pycache__/
*.pyc
.venv/
graphify-out/
evals/results/
```
