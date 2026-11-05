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
