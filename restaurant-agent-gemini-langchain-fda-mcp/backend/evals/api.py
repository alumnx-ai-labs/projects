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
    category: str | None = None  # flow | jev | gemini | fda
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
