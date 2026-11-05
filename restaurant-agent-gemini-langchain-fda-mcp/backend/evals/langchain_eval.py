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
