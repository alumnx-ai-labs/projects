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
