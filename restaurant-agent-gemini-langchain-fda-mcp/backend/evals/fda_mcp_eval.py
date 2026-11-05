"""Evaluate the FDA menu rating MCP server directly, without the Gemini agent.

Run from backend/:
    python -m evals.fda_mcp_eval                         # FDA_MCP_URL / FDA_MCP_TOKEN from .env
    python -m evals.fda_mcp_eval --url http://localhost:3333/mcp --token <token>
    python -m evals.fda_mcp_eval --no-ai                 # skip the tools that call Gemini

Checks the server's contract, the part the backend's Gemini agent relies on:
- the four tools are listed, with the input fields the agent fills in
- rate_dish scores match the FDA Daily Value formula (ported below) exactly
- rate_restaurant_menu averages, sorts and labels dishes correctly
- bad input is rejected, and requests without the token get 401
- AI tools (estimate_dish_nutrition, explain_dish_rating) return plausible output

Results are printed and saved to evals/results/fda-<timestamp>.json.
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
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
TOOLS = {
    "rate_restaurant_menu": {"restaurant", "dishes"},
    "rate_dish": {"name", "calories", "satFat", "sodium", "sugar", "fiber", "protein"},
    "estimate_dish_nutrition": {"restaurant", "dish"},
    "explain_dish_rating": {"restaurant", "dish", "calories", "satFat", "sodium", "sugar", "fiber", "protein"},
}
NUTRIENTS = ["calories", "satFat", "sodium", "sugar", "fiber", "protein"]


# ---------------- reference scoring (mirrors core.js in the FDA server) ----------------

DV = {"satFat": 20, "sodium": 2300, "sugar": 50, "fiber": 28, "protein": 50, "calories": 2000}


def _pct(d: dict, k: str) -> float:
    return d[k] / DV[k] * 100


def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def _round(x: float) -> int:
    """JavaScript Math.round: halves round up."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def expected_score(d: dict) -> int:
    cal_extra = max(0.0, _pct(d, "calories") - 25)
    s = (100
         - _clamp(_pct(d, "satFat"), 0, 100) * 0.5
         - _clamp(_pct(d, "sodium"), 0, 100) * 0.4
         - _clamp(_pct(d, "sugar"), 0, 100) * 0.4
         - _clamp(cal_extra, 0, 75) * 0.5
         + _clamp(_pct(d, "fiber"), 0, 40) * 0.5
         + _clamp(_pct(d, "protein"), 0, 40) * 0.25)
    return _round(_clamp(s, 0, 100))


def expected_grade(score: int) -> str:
    return "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "E"


def expected_flags(d: dict) -> list[str]:
    out = []
    for k, label in [("satFat", "saturated fat"), ("sodium", "sodium"), ("sugar", "added sugar")]:
        p = _pct(d, k)
        if p >= 20:
            out.append(f"High {label} ({_round(p)}% DV)")
        elif p <= 5:
            out.append(f"Low {label}")
    if (f := _pct(d, "fiber")) >= 20:
        out.append(f"High fiber ({_round(f)}% DV)")
    return out


def dish(name: str, calories, satFat, sodium, sugar, fiber, protein) -> dict:
    return {"name": name, "calories": calories, "satFat": satFat, "sodium": sodium,
            "sugar": sugar, "fiber": fiber, "protein": protein}


# Dishes with known nutrition: no Gemini call, so every score is exact.
DISHES = [
    dish("Garden Salad", 180, 2, 300, 3, 6, 5),
    dish("Cheeseburger", 850, 22, 1650, 9, 2, 48),
    dish("Water", 0, 0, 0, 0, 0, 0),
    dish("Lentil Soup", 230, 0.5, 480, 2, 15, 18),          # high fiber and protein
    dish("Triple Fudge Sundae", 1250, 30, 400, 110, 3, 14),  # every penalty capped
    dish("Grilled Chicken Bowl", 520, 3, 690, 4, 7, 42),
]


# ---------------- checks ----------------

class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, check: str, passed: bool, detail: str = ""):
        self.items.append({"check": check, "passed": bool(passed), "detail": detail})


def _text(result) -> str:
    return "".join(getattr(b, "text", "") for b in result.content)


async def call(session: ClientSession, tool: str, args: dict):
    """Call a tool; returns (is_error, text, seconds)."""
    started = time.perf_counter()
    result = await session.call_tool(tool, args)
    return bool(result.isError), _text(result), round(time.perf_counter() - started, 2)


async def case_tools_listed(session) -> list[dict]:
    c = Checks()
    listed = {t.name: t for t in (await session.list_tools()).tools}
    for name, fields in TOOLS.items():
        tool = listed.get(name)
        c.add(f"listed:{name}", tool is not None, f"listed: {sorted(listed)}")
        if tool:
            got = set(tool.inputSchema.get("properties", {}))
            c.add(f"inputs:{name}", fields <= got, f"missing {sorted(fields - got)}")
            c.add(f"description:{name}", len(tool.description or "") > 20, "the agent picks tools by description")
    return c.items


async def case_rate_dish(session, d: dict) -> list[dict]:
    c = Checks()
    is_error, text, secs = await call(session, "rate_dish", d)
    if is_error:
        c.add("call", False, text)
        return c.items
    r = json.loads(text)
    want = expected_score(d)
    c.add("score", r["score"] == want, f"got {r['score']}, want {want}")
    c.add("grade", r["grade"] == expected_grade(want), f"got {r['grade']}, want {expected_grade(want)}")
    c.add("flags", r["flags"] == expected_flags(d), f"got {r['flags']}, want {expected_flags(d)}")
    c.add("nutrition echoed", all(r["nutrition"][k] == d[k] for k in NUTRIENTS), str(r["nutrition"]))
    c.add("fast (no AI call)", secs < 5, f"{secs}s")
    return c.items


async def case_rate_menu(session) -> list[dict]:
    c = Checks()
    is_error, text, _ = await call(session, "rate_restaurant_menu", {"restaurant": "Eval Diner", "dishes": DISHES})
    if is_error:
        c.add("call", False, text)
        return c.items
    r = json.loads(text)
    scores = {d["name"]: expected_score(d) for d in DISHES}
    avg = _round(sum(scores.values()) / len(scores))
    ranked = sorted(scores, key=scores.get, reverse=True)

    c.add("restaurant", r["restaurant"] == "Eval Diner", r["restaurant"])
    c.add("overall score", r["overall"]["score"] == avg, f"got {r['overall']['score']}, want {avg}")
    c.add("overall grade", r["overall"]["grade"] == expected_grade(avg), r["overall"]["grade"])
    c.add("dish count", r["overall"]["dishCount"] == len(DISHES), str(r["overall"]["dishCount"]))
    c.add("dish scores", {x["name"]: x["score"] for x in r["dishes"]} == scores,
          f"got {[(x['name'], x['score']) for x in r['dishes']]}")
    got_scores = [x["score"] for x in r["dishes"]]
    c.add("sorted best first", got_scores == sorted(got_scores, reverse=True), str(got_scores))
    c.add("best", r["best"] == ranked[0], f"got {r['best']}, want {ranked[0]}")
    c.add("lowest", r["lowest"] == ranked[-1], f"got {r['lowest']}, want {ranked[-1]}")
    c.add("source provided", all(x["source"] == "provided" for x in r["dishes"]),
          "dishes with full nutrition must not be estimated")
    c.add("disclaimer", "FDA does not rate restaurants" in r.get("disclaimer", ""), r.get("disclaimer", ""))
    return c.items


async def case_bad_input(session) -> list[dict]:
    c = Checks()
    bad = [
        ("rate_dish", {**DISHES[0], "calories": -100}, "negative calories"),
        ("rate_dish", {"name": "Half a dish", "calories": 300}, "missing nutrients"),
        ("rate_restaurant_menu", {"restaurant": "Empty", "dishes": []}, "empty menu"),
    ]
    for tool, args, label in bad:
        try:
            is_error, text, _ = await call(session, tool, args)
            c.add(f"rejects {label}", is_error, f"accepted: {text[:200]}")
        except Exception as e:  # a protocol-level error is also a rejection
            c.add(f"rejects {label}", True, repr(e)[:200])
    return c.items


async def case_estimate(session) -> list[dict]:
    c = Checks()
    is_error, text, secs = await call(session, "estimate_dish_nutrition",
                                      {"restaurant": "Pizza Bella", "dish": "Margherita Pizza"})
    if is_error:
        c.add("call", False, f"{text} (Gemini quota or GEMINI_API_KEY on the server?)")
        return c.items
    r = json.loads(text)
    c.add("all nutrients", set(NUTRIENTS) <= set(r), f"got {sorted(r)}")
    c.add("non-negative numbers", all(isinstance(r.get(k), (int, float)) and r[k] >= 0 for k in NUTRIENTS), str(r))
    c.add("plausible calories (100-3000)", 100 <= r.get("calories", -1) <= 3000, str(r.get("calories")))
    c.add("plausible sodium (< 6000 mg)", r.get("sodium", 1e9) < 6000, str(r.get("sodium")))
    c.add("latency < 30s", secs < 30, f"{secs}s")
    return c.items


async def case_explain(session) -> list[dict]:
    c = Checks()
    d = DISHES[1]
    args = {"restaurant": "Eval Diner", "dish": d["name"], **{k: d[k] for k in NUTRIENTS}}
    is_error, text, secs = await call(session, "explain_dish_rating", args)
    if is_error:
        c.add("call", False, f"{text} (Gemini quota or GEMINI_API_KEY on the server?)")
        return c.items
    low = text.lower()
    c.add("not empty", len(text.strip()) > 40, text)
    c.add("short (< 1000 chars)", len(text) < 1000, f"{len(text)} chars")
    c.add("talks about the grade", "grade" in low or "score" in low, text[:300])
    # The burger is flagged high in saturated fat and sodium; the explanation should say why.
    c.add("names a flagged nutrient", "saturated fat" in low or "sodium" in low, text[:300])
    claims = [p for p in ("fda approved", "fda-approved", "rated by the fda", "fda has rated", "fda rates")
              if p in low]
    c.add("no FDA approval claim", not claims, f"says {claims}")
    c.add("latency < 30s", secs < 30, f"{secs}s")
    return c.items


async def case_auth(url: str, token: str | None) -> list[dict]:
    c = Checks()
    if not token:
        c.add("token configured", False, "no FDA_MCP_TOKEN: a public server without auth lets anyone spend its Gemini quota")
        return c.items
    async with httpx.AsyncClient(timeout=90) as http:
        for label, headers in [("no token", {}), ("wrong token", {"Authorization": "Bearer wrong"})]:
            r = await http.post(url, json={}, headers={**headers, "Accept": "application/json, text/event-stream"})
            c.add(f"401 with {label}", r.status_code == 401, f"got HTTP {r.status_code}")
    return c.items


# ---------------- running ----------------

def build_cases(ai: bool) -> list[tuple[str, str, object]]:
    cases = [("T1", "Tools listed with the inputs the agent uses", case_tools_listed)]
    cases += [(f"R{i}", f"rate_dish: {d['name']}", lambda s, d=d: case_rate_dish(s, d))
              for i, d in enumerate(DISHES, 1)]
    cases += [("M1", "rate_restaurant_menu: averages, ranks and labels dishes", case_rate_menu),
              ("V1", "Bad input is rejected", case_bad_input)]
    if ai:
        cases += [("A1", "estimate_dish_nutrition: plausible numbers (Gemini)", case_estimate),
                  ("A2", "explain_dish_rating: short, explains the grade, no FDA claim (Gemini)", case_explain)]
    return cases


async def evaluate(url: str, token: str | None, ai: bool) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    results = []

    started = time.perf_counter()
    results.append({"id": "S1", "name": "Requests without the right token are refused",
                    "checks": await case_auth(url, token), "latency_s": round(time.perf_counter() - started, 2)})

    # Generous timeouts: a sleeping Render free-tier service takes about a minute to wake up.
    async with streamablehttp_client(url, headers=headers, timeout=90) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for case_id, name, run in build_cases(ai):
                started = time.perf_counter()
                try:
                    checks = await run(session)
                except Exception as e:
                    checks = [{"check": "run", "passed": False, "detail": repr(e)[:300]}]
                results.append({"id": case_id, "name": name, "checks": checks,
                                "latency_s": round(time.perf_counter() - started, 2)})
                print(f"  done {case_id:<3} {_mark(results[-1])}", flush=True)
    for r in results:
        r["passed"] = all(c["passed"] for c in r["checks"])
    return results


def _mark(result: dict) -> str:
    return "PASS" if all(c["passed"] for c in result["checks"]) else "FAIL"


def report(results: list[dict]) -> None:
    print(f"\n{'ID':<5}{'':<6}{'TIME':>6}  NAME")
    for r in results:
        print(f"{r['id']:<5}{_mark(r):<6}{r['latency_s']:>5}s  {r['name']}")
        for c in r["checks"]:
            if not c["passed"]:
                print(f"        {c['check']}: {c['detail']}")
    passed = sum(r["passed"] for r in results)
    checks = [c for r in results for c in r["checks"]]
    print(f"\nSUMMARY\n  cases  {passed}/{len(results)} passed"
          f"\n  checks {sum(c['passed'] for c in checks)}/{len(checks)} passed")


def save_results(results: list[dict], url: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"fda-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"url": url, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.getenv("FDA_MCP_URL", "https://fda-rating.onrender.com/mcp"))
    p.add_argument("--token", default=os.getenv("FDA_MCP_TOKEN") or None)
    p.add_argument("--no-ai", action="store_true", help="skip the tools that call Gemini")
    args = p.parse_args()

    print(f"Evaluating the FDA MCP server at {args.url}...")
    try:
        results = await evaluate(args.url, args.token, not args.no_ai)
    except Exception as e:
        print(f"Could not connect to {args.url}: {e!r}\n"
              "Is the server running `npm run mcp:http`, and is the token right?")
        return 1

    report(results)
    print(f"\nSaved {save_results(results, args.url)}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
