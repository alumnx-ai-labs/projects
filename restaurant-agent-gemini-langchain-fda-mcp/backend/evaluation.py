"""Evaluation for the LangChain restaurant agent (main.agent).

For each test case it runs the agent, then scores the result with:
  1. Tool-call check: did the agent call the expected tools with the expected args?
  2. Forbidden-tool check: did it avoid tools it should not call?
  3. Keyword check: does the reply mention the required words?
  4. LLM judge: a LangChain chat model grades the reply against a criterion.

Run from backend/:
    python evaluation.py
    python evaluation.py --no-judge      # skip the LLM judge
"""

import argparse
import asyncio
import json
import os
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from main import _text, agent

# Gemini free tier: 15 requests a minute. Wait and retry when it is hit.
RETRIES = 2
RETRY_WAIT_S = 40


# ---------------- test cases ----------------

def next_weekday(weekday: int) -> str:
    """The next date (from tomorrow) that falls on `weekday` (0 = Monday)."""
    day = date.today() + timedelta(days=1)
    while day.weekday() != weekday:
        day += timedelta(days=1)
    return day.isoformat()


TOMORROW = (date.today() + timedelta(days=1)).isoformat()
MONDAY = next_weekday(0)
TUESDAY = next_weekday(1)

TEST_CASES = [
    {
        "id": "T1",
        "question": "Where can I get biryani?",
        "expected_tools": [{"name": "search_restaurants_by_cuisine", "args": {"cuisine_type": "Indian"}}],
        "must_mention": ["Spice Palace", "Taj Flavors"],
        "criterion": "Recommends Spice Palace and Taj Flavors for biryani and does not claim Curry House serves it.",
    },
    {
        "id": "T2",
        "question": "Where can I get sushi?",
        "forbidden_tools": ["check_reservation_availability"],
        "criterion": "Says there is no sushi or Japanese restaurant available and does not invent one.",
    },
    {
        "id": "T3",
        "question": "What's Dragon House's phone number?",
        "expected_tools": [{"name": "get_restaurant_details", "args": {"restaurant_name": "Dragon House"}}],
        "must_mention": ["9111222333"],
    },
    {
        "id": "T4",
        "question": "Is Pizza Bella open at 11 AM?",
        "expected_tools": [{"name": "get_restaurant_details", "args": {"restaurant_name": "Pizza Bella"}}],
        "criterion": "Says Pizza Bella is not open at 11 AM because it opens at 12 PM.",
    },
    {
        "id": "T5",
        "question": "Book me a table at Spice Palace",
        "forbidden_tools": ["check_reservation_availability"],
        "criterion": "Asks for the date and number of people instead of guessing them.",
    },
    {
        "id": "T6",
        "question": "How much does a meal at Curry House cost?",
        "criterion": "Does not state a specific price amount; at most gives the price level or says it doesn't know.",
    },
    {
        "id": "T7",
        "question": "Can I get a table for 4 at Pizza Bella tomorrow?",
        "expected_tools": [{"name": "check_reservation_availability",
                            "args": {"restaurant_name": "Pizza Bella", "date": TOMORROW, "party_size": 4}}],
    },
    {
        "id": "T8",
        "question": f"Is there a table for 2 at Spice Palace on {MONDAY}?",
        "expected_tools": [{"name": "check_reservation_availability",
                            "args": {"restaurant_name": "Spice Palace", "date": MONDAY, "party_size": 2}}],
        "criterion": "Says Spice Palace is closed on that day.",
    },
    {
        "id": "T9",
        "question": f"Is there a table for 6 at Taj Flavors at 9 PM on {TUESDAY}?",
        "expected_tools": [{"name": "check_reservation_availability",
                            "args": {"restaurant_name": "Taj Flavors", "date": TUESDAY,
                                     "party_size": 6, "time": "21:00"}}],
        "criterion": "Says a table for 6 is available at 9 PM.",
    },
    {
        "id": "T10",
        "question": "Can I book Ming's Kitchen for 2 on 2024-11-15?",
        "criterion": "Says that date can't be booked because bookings are only open for the next few days; "
                     "does not say the restaurant is fully booked.",
    },
]


# ---------------- run the agent ----------------

async def run_agent(question: str) -> dict:
    """Run the agent on one question; return its reply and tool calls."""
    for attempt in range(RETRIES + 1):
        try:
            result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
            break
        except Exception:
            if attempt == RETRIES:
                raise
            await asyncio.sleep(RETRY_WAIT_S)

    messages = result["messages"]
    tool_calls = [
        {"name": call["name"], "args": call.get("args", {})}
        for m in messages
        for call in (getattr(m, "tool_calls", None) or [])
    ]
    return {"reply": _text(messages[-1].content), "tool_calls": tool_calls}


# ---------------- evaluators ----------------

def _same(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def check_tool_calls(output: dict, case: dict) -> tuple[bool, str]:
    missing = [
        want for want in case.get("expected_tools", [])
        if not any(
            call["name"] == want["name"]
            and all(_same(call["args"].get(k), v) for k, v in want["args"].items())
            for call in output["tool_calls"]
        )
    ]
    return not missing, f"missing {missing}; got {output['tool_calls']}" if missing else ""


def check_forbidden_tools(output: dict, case: dict) -> tuple[bool, str]:
    used = [c for c in output["tool_calls"] if c["name"] in case.get("forbidden_tools", [])]
    return not used, f"called {used}" if used else ""


def check_mentions(output: dict, case: dict) -> tuple[bool, str]:
    missing = [w for w in case.get("must_mention", []) if w.lower() not in output["reply"].lower()]
    return not missing, f"reply lacks {missing}" if missing else ""


class Verdict(BaseModel):
    passed: bool
    reason: str


def make_judge():
    model = os.getenv("JUDGE_MODEL") or os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")
    return init_chat_model(model).with_structured_output(Verdict)


async def llm_judge(judge, output: dict, case: dict) -> tuple[bool, str]:
    prompt = (
        "You are grading a restaurant assistant's reply.\n\n"
        f"User question: {case['question']}\n\n"
        f"Tool calls it made: {json.dumps(output['tool_calls'], ensure_ascii=False)}\n\n"
        f"Assistant reply: {output['reply']}\n\n"
        f"Criterion: {case['criterion']}\n\n"
        "Pass only if the reply clearly meets the criterion."
    )
    for attempt in range(RETRIES + 1):
        try:
            verdict = await judge.ainvoke(prompt)
            return verdict.passed, verdict.reason
        except Exception:
            if attempt == RETRIES:
                raise
            await asyncio.sleep(RETRY_WAIT_S)


# ---------------- main ----------------

async def evaluate_case(case: dict, judge) -> dict:
    try:
        output = await run_agent(case["question"])
    except Exception as e:
        return {"id": case["id"], "question": case["question"], "passed": False,
                "checks": {"agent": (False, f"agent error: {e}")}}

    checks = {}
    if "expected_tools" in case:
        checks["tool_calls"] = check_tool_calls(output, case)
    if "forbidden_tools" in case:
        checks["forbidden_tools"] = check_forbidden_tools(output, case)
    if "must_mention" in case:
        checks["mentions"] = check_mentions(output, case)
    if "criterion" in case and judge is not None:
        try:
            checks["llm_judge"] = await llm_judge(judge, output, case)
        except Exception as e:
            checks["llm_judge"] = (False, f"judge error: {e}")

    return {
        "id": case["id"],
        "question": case["question"],
        "reply": output["reply"],
        "tool_calls": output["tool_calls"],
        "checks": checks,
        "passed": all(ok for ok, _ in checks.values()),
    }


def print_report(results: list[dict]) -> None:
    for r in results:
        print(f"{r['id']:<4} {'PASS' if r['passed'] else 'FAIL'}  {r['question']}")
        for name, (ok, detail) in r["checks"].items():
            if not ok:
                print(f"       {name}: {detail}")

    print("\nSCORES")
    names = dict.fromkeys(n for r in results for n in r["checks"])
    for name in names:
        scored = [r["checks"][name][0] for r in results if name in r["checks"]]
        print(f"  {name:<16} {sum(scored)}/{len(scored)}")
    print(f"  {'cases passed':<16} {sum(r['passed'] for r in results)}/{len(results)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LangChain restaurant agent.")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    args = parser.parse_args()

    judge = None if args.no_judge else make_judge()
    results = []
    for case in TEST_CASES:  # one at a time, to stay under the Gemini rate limit
        results.append(await evaluate_case(case, judge))
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
