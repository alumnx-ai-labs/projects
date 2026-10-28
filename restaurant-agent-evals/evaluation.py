"""Evaluate the restaurant agent end to end with LangChain.

Each scenario is a scripted conversation run through `main.run_turn` (the same
code path as /chat). Every turn is scored by:

  * choice accuracy  - did the agent understand the answer? (deterministic)
  * tool accuracy    - did it call the expected tools? (deterministic)
  * outcome          - did the conversation end in the expected state? (deterministic)
  * LLM judge        - is the reply grounded in the tool output and helpful?
                       (LangChain chat model with structured output, optional)

Run it from the CLI:  python evaluation.py [--no-judge] [--scenario NAME]
or through the API:   POST /evaluate
"""

import argparse
import asyncio
import json
import logging
import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Each turn: what the user says, the choice the agent should understand, and the
# tools it should call (in order). `final` is a subset of the end state.
SCENARIOS = [
    {
        "name": "happy_path_available",
        "turns": [
            {"say": "I feel like having some Indian food", "choice": "indian", "tools": ["search_restaurants_by_cuisine"]},
            {"say": "Spice Palace", "choice": "Spice Palace", "tools": ["get_restaurant_details"]},
            {"say": "15 Nov 2024", "choice": "2024-11-15", "tools": []},
            {"say": "2 people", "choice": "2", "tools": ["check_reservation_availability"]},
        ],
        "final": {"asking": "cuisine"},
        "final_output_contains": "has availability",
    },
    {
        "name": "fully_booked_restarts",
        "turns": [
            {"say": "Mexican", "choice": "mexican", "tools": ["search_restaurants_by_cuisine"]},
            {"say": "let us go with salsa nights", "choice": "Salsa Nights", "tools": ["get_restaurant_details"]},
            {"say": "15 Nov 2024", "choice": "2024-11-15", "tools": []},
            {"say": "there will be four of us", "choice": "4", "tools": ["check_reservation_availability"]},
        ],
        "final": {"asking": "cuisine"},
        "final_output_contains": "fully booked",
    },
    {
        "name": "start_over_midway",
        "turns": [
            {"say": "Italian", "choice": "italian", "tools": ["search_restaurants_by_cuisine"]},
            {"say": "None of these - start over", "choice": "restart", "tools": []},
        ],
        "final": {"asking": "cuisine"},
    },
    {
        "name": "browse_specialities_then_resume",
        "turns": [
            {"say": "Chinese", "choice": "chinese", "tools": ["search_restaurants_by_cuisine"]},
            {"say": "Show Chinese special dishes", "choice": "speciality_cuisine", "tools": ["get_specialities"]},
            {"say": "Continue booking", "choice": "resume", "tools": []},
        ],
        "final": {"asking": "restaurant", "cuisine": "Chinese"},
    },
    {
        "name": "off_topic_stays_put",
        "turns": [
            {"say": "What's the weather like?", "choice": "other", "tools": []},
        ],
        "final": {"asking": "cuisine"},
    }, 
    {
        "name": "happy_path_available",
        "turns": [
            {"say": "I feel like having some Chinese food", "choice": "Iran", "tools": ["search_restaurants_by_cuisine"]},
            {"say": "Spice Palace", "choice": "Spice Palace", "tools": ["get_restaurant_details"]},
            {"say": "15 Nov 2024", "choice": "2024-11-15", "tools": []},
            {"say": "2 people", "choice": "2", "tools": ["check_reservation_availability"]},
        ],
        "final": {"asking": "cuisine"},
        "final_output_contains": "has availability",
    },
]


class Verdict(BaseModel):
    grounded: int = Field(ge=1, le=5, description="5 = every claim is supported by the tool output")
    helpful: int = Field(ge=1, le=5, description="5 = the reply moves the booking forward clearly")
    reason: str = Field(description="One short sentence justifying the scores")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade a restaurant booking assistant. Score the assistant's reply "
            "from 1 to 5 for being grounded (no invented facts beyond the tool output) "
            "and helpful (clear next step for the user). Be strict.",
        ),
        (
            "human",
            "User said: {user_said}\nAssistant reply: {reply}\n"
            "Tool output: {tools}\nNext question shown: {next_question}",
        ),
    ]
)

_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        model = init_chat_model(os.getenv("JUDGE_MODEL", os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")))
        _judge = JUDGE_PROMPT | model.with_structured_output(Verdict)
    return _judge


@traceable(name="LLM Judge", run_type="chain")
async def judge_turn(say: str, jev: dict) -> dict:
    verdict = await _get_judge().ainvoke(
        {
            "user_said": say,
            "reply": jev["reply"],
            "tools": json.dumps([t["output"] for t in jev["tool_calls"]], ensure_ascii=False) or "none",
            "next_question": jev["next_question"]["text"],
        }
    )
    return verdict.model_dump()


def _subset(actual: dict, expected: dict) -> bool:
    return all(actual.get(k) == v for k, v in expected.items())


@traceable(name="Eval Scenario", run_type="chain")
async def run_scenario(scenario: dict, use_judge: bool = True) -> dict:
    from main import ChatRequest, run_turn  # imported lazily: main imports this module

    state, history, turns = None, [], []
    last_jev = None
    for turn in scenario["turns"]:
        req = ChatRequest(message=turn["say"], history=history, state=state)
        result = await run_turn(req)
        jev = result.jev
        last_jev = jev
        called = [t["name"] for t in jev["tool_calls"]]
        record = {
            "say": turn["say"],
            "expected_choice": turn["choice"],
            "actual_choice": jev["understood"]["choice"],
            "method": jev["understood"]["method"],
            "choice_ok": jev["understood"]["choice"] == turn["choice"],
            "expected_tools": turn["tools"],
            "actual_tools": called,
            "tools_ok": called == turn["tools"],
        }
        if use_judge:
            try:
                record["judge"] = await judge_turn(turn["say"], jev)
            except Exception:
                logger.exception("LLM judge failed")
                record["judge"] = None
        turns.append(record)
        state = result.state
        history += [
            {"role": "user", "content": turn["say"]},
            {"role": "assistant", "content": jev["reply"]},
        ]

    outcome_ok = _subset(state, scenario["final"])
    needle = scenario.get("final_output_contains")
    if needle:
        outputs = " ".join(t["output"] for t in last_jev["tool_calls"])
        outcome_ok = outcome_ok and needle in outputs

    return {
        "name": scenario["name"],
        "turns": turns,
        "outcome_ok": outcome_ok,
        "passed": outcome_ok and all(t["choice_ok"] and t["tools_ok"] for t in turns),
    }


def _summarize(results: list[dict]) -> dict:
    turns = [t for r in results for t in r["turns"]]
    judged = [t["judge"] for t in turns if t.get("judge")]

    def rate(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "scenarios": len(results),
        "scenarios_passed": sum(r["passed"] for r in results),
        "choice_accuracy": rate([t["choice_ok"] for t in turns]),
        "tool_accuracy": rate([t["tools_ok"] for t in turns]),
        "outcome_accuracy": rate([r["outcome_ok"] for r in results]),
        "avg_grounded": rate([j["grounded"] for j in judged]),
        "avg_helpful": rate([j["helpful"] for j in judged]),
    }


@traceable(name="Evaluate Agent", run_type="chain")
async def evaluate_agent(use_judge: bool = True, names: list[str] | None = None) -> dict:
    chosen = [s for s in SCENARIOS if not names or s["name"] in names]
    results = []
    for scenario in chosen:  # sequential: scenarios share the Jev rate limit
        results.append(await run_scenario(scenario, use_judge))
    return {"summary": _summarize(results), "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the restaurant agent")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--scenario", action="append", help="run only this scenario (repeatable)")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    report = asyncio.run(evaluate_agent(not args.no_judge, args.scenario))
    for r in report["results"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']}")
        for t in r["turns"]:
            if not (t["choice_ok"] and t["tools_ok"]):
                print(f"    '{t['say']}': choice {t['actual_choice']!r} (want {t['expected_choice']!r}), "
                      f"tools {t['actual_tools']} (want {t['expected_tools']})")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
