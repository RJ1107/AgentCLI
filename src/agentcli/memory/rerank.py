"""Tier-2 recall with a Jev relevance check (evals/jev_recall.py, experiment 5).

BM25 alone has to guess relevance from shared words, so its gate is strict: in the recall
evaluation it let in only 56% of the relevant memories. Here BM25 only makes a loose
shortlist, and Jev reads the question with the whole shortlist in one request and says, per
memory, how likely it helps answer the question. In the evaluation this recalled 97% of the
relevant memories at 98% precision, and nothing on unrelated questions, for about a quarter
of a second and $0.00007 a request. On any failure the caller keeps the BM25 gate.
"""

from __future__ import annotations

import asyncio

import httpx

from agentcli import jev
from agentcli.memory.retrieval import Hit

_INSTRUCTIONS = "Does `memories.{name}` contain information that helps answer `question`?"
_CRITERIA = {
    "true": "The memory states a fact, rule, or preference that answers the question "
    "or that someone answering it would need",
    "false": "The memory is about something else, even if it shares some words",
}


async def jev_recall(
    query: str,
    shortlist: list[Hit],
    *,
    threshold: float,
    limit: int,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Hit]:
    """The shortlisted memories Jev finds relevant, most likely first. Raises on failure."""

    if not shortlist:
        return []
    by_name = {hit.record.name: hit for hit in shortlist}
    state = {
        "question": query[:4000],
        "memories": {
            name: f"{hit.record.title}: {hit.record.content}"[:1500] for name, hit in by_name.items()
        },
    }
    questions = {
        name: {"type": "noul", "instructions": _INSTRUCTIONS.format(name=name), "criteria": _CRITERIA}
        for name in by_name
    }
    answers = await asyncio.wait_for(jev.ask(state, questions, transport=transport), timeout)
    scored = sorted(
        ((float(answers[name]["noul"]), name) for name in by_name if name in answers),
        key=lambda pair: -pair[0],
    )
    return [by_name[name] for score, name in scored if score >= threshold][: max(0, limit)]


__all__ = ["jev_recall"]
