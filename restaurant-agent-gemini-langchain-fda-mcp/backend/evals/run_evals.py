"""Evaluate the /chat backend: Jev's understanding and flow, Gemini's tool use and replies.

Run from backend/:
    python -m evals.run_evals                      # in-process, no server needed
    python -m evals.run_evals --url http://localhost:8000
    python -m evals.run_evals --only F1,C3 --no-judge
    python -m evals.run_evals --category gemini
    python -m evals.run_evals --category fda       # Gemini with the FDA MCP tools

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
from datetime import date, datetime
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

    # Calls the backend refused (e.g. availability before the party size is known) never ran.
    calls = [t for t in google.get("tool_calls", []) if not t.get("blocked")]
    reply = google.get("reply", "")

    for want in exp.get("tools", []):
        c.add(f"tool:{want['name']}", _has_call(calls, want), f"want {want.get('args', {})}, got {_calls(calls)}")
    if "tools_any" in exp:
        ok = any(t["name"] in exp["tools_any"] for t in calls)
        c.add("tools_any", ok, f"want one of {exp['tools_any']}, got {_calls(calls)}")
    for name in exp.get("no_tools", []):
        used = [t for t in calls if t["name"] == name]
        c.add(f"no_tool:{name}", not used, f"called {_calls(used)}")
    for name, text in exp.get("tool_output_has", {}).items():
        outputs = [t.get("output", "") for t in calls if t["name"] == name]
        c.add(f"tool_output_has:{name}", any(text in o for o in outputs),
              f"want {text!r} in {name} output, got {[o[:200] for o in outputs] or 'no call'}")
    for text in exp.get("reply_has", []):
        c.add(f"reply_has:{text}", _norm(text) in _norm(reply))
    for text in exp.get("reply_lacks", []):
        c.add(f"reply_lacks:{text}", _norm(text) not in _norm(reply))
    if "judge" in exp and judge is not None:
        prompt = (
            "You are grading a restaurant assistant's reply.\n"
            f"Today is {date.today():%A %Y-%m-%d}; 'tomorrow' means the next day.\n\n"
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


def agent_tool_names(agent) -> list[str]:
    """Tool names the in-process Gemini agent was built with."""
    return list(agent.get_graph().nodes["tools"].data.tools_by_name)


def make_judge():
    from langchain.chat_models import init_chat_model

    name = os.getenv("JUDGE_MODEL") or os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")
    return init_chat_model(name).with_structured_output(Verdict)


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", help="evaluate a running server instead of the app in-process")
    p.add_argument("--only", help="comma-separated case ids, e.g. F1,C3")
    p.add_argument("--category", help="flow, jev, gemini or fda")
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
        from main import agent, app
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://eval", timeout=120)
        if any(c["category"] == "fda" for c in cases) and "rate_dish" not in agent_tool_names(agent):
            print("WARNING: the FDA MCP tools are not loaded (check FDA_MCP_URL / FDA_MCP_TOKEN); "
                  "the fda cases will fail.")

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
