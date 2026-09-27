"""Ranking memories for a query: BM25 first, then a small nudge for importance and recency.

Relevance decides the order; importance and recency only break near-ties, because a
memory is useful when it is about the question, not because it is important in general.
Their weights, and the relevance gate for automatic recall, are fixed by the recall
evaluation in evals/memory_recall.py rather than by hand.

Tokens: lowercase words for Latin text; for Chinese, every character and every pair of
adjacent characters, so "数据库连接" matches "连接数据库" and "数据库" without a segmenter.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")

# Weights found by the recall evaluation (evals/memory_recall.py); see evals/README.md.
IMPORTANCE_WEIGHT = 0.05
RECENCY_WEIGHT = 0.02
# BM25 parameters: the usual defaults.
K1 = 1.2
B = 0.75

# Words that carry the shape of a question, not its subject. They count for nothing in
# coverage, so "我们的测试端口是多少" is judged by 测试 and 端口, not by 的/是/多少.
_STOP_WORDS = frozenset(
    [
        "的",
        "了",
        "是",
        "吗",
        "呢",
        "吧",
        "啊",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
        "我们",
        "你们",
        "这",
        "那",
        "这个",
        "那个",
        "怎么",
        "怎样",
        "什么",
        "多少",
        "哪",
        "哪个",
        "哪些",
        "如何",
        "为什么",
        "为何",
        "一下",
        "请",
        "帮",
        "帮我",
        "用",
        "有",
        "在",
        "和",
        "与",
        "及",
        "或",
        "也",
        "都",
        "就",
        "要",
        "会",
        "能",
        "可以",
        "吗",
        "么",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "what",
        "how",
        "which",
        "who",
        "why",
        "when",
        "where",
        "do",
        "does",
        "did",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "my",
        "our",
        "we",
        "i",
        "you",
        "it",
        "this",
        "that",
        "with",
        "by",
        "at",
        "from",
    ]
)
_STOP_CHARS = frozenset("的了是吗呢吧啊我你他她它们这那么什哪怎用有在和与及或也都就要会能请帮")


class Searchable(Protocol):
    title: str
    keywords: list[str]
    content: str
    importance: float
    updated: str


@dataclass(slots=True)
class Hit:
    record: Searchable
    score: float
    # Share of the query's subject words the memory contains (see term_weight): 0..1 and
    # comparable across queries, so it can gate automatic recall where BM25 cannot.
    coverage: float


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [_stem(word) for word in _WORD.findall(lowered)]
    for run in _CJK.findall(lowered):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def term_weight(term: str) -> float:
    """How much a query term says about the subject: for coverage, independent of the
    memories stored, so the same gate means the same thing in a small or a large store."""

    if term in _STOP_WORDS:
        return 0.0
    if _CJK.fullmatch(term):
        if len(term) == 1:
            return 0.0 if term in _STOP_CHARS else 0.3
        return 0.5 if any(char in _STOP_CHARS for char in term) else 1.0
    return 1.0


def _stem(word: str) -> str:
    """tests -> test, running -> run, fixed -> fix: enough to match simple inflections."""

    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
        return word[:-1] if len(word) > 3 and word[-1] == word[-2] else word
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) >= 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def document_tokens(record: Searchable) -> list[str]:
    # The title and keywords say what the memory is about; count them twice.
    head = tokenize(record.title) + tokenize(" ".join(record.keywords))
    return head + head + tokenize(record.content)


def rank(
    query: str,
    records: list[Searchable],
    *,
    importance_weight: float = IMPORTANCE_WEIGHT,
    recency_weight: float = RECENCY_WEIGHT,
    now: datetime | None = None,
) -> list[Hit]:
    """Memories that share at least one token with the query, best first."""

    query_terms = list(dict.fromkeys(tokenize(query)))
    if not query_terms or not records:
        return []
    docs = [Counter(document_tokens(record)) for record in records]
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths) / len(lengths) or 1.0
    total = len(records)
    idf = {
        term: math.log(1 + (total - df + 0.5) / (df + 0.5))
        for term in query_terms
        for df in [sum(1 for doc in docs if term in doc)]
    }
    weights = {term: term_weight(term) for term in query_terms}
    query_weight = sum(weights.values()) or 1.0
    now = now or datetime.now(UTC)

    raw: list[tuple[float, float, int]] = []
    for index, doc in enumerate(docs):
        score = 0.0
        matched = 0.0
        for term in query_terms:
            frequency = doc.get(term, 0)
            if not frequency:
                continue
            matched += weights[term]
            norm = K1 * (1 - B + B * lengths[index] / average)
            score += idf[term] * frequency * (K1 + 1) / (frequency + norm)
        if score > 0:
            raw.append((score, matched / query_weight, index))
    if not raw:
        return []

    best = max(score for score, _, _ in raw)
    hits = []
    for score, coverage, index in raw:
        record = records[index]
        final = (
            score / best
            + importance_weight * (record.importance - 0.5)
            + recency_weight * _recency(record.updated, now)
        )
        hits.append(Hit(record, final, coverage))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits


def _recency(updated: str, now: datetime) -> float:
    """1 for today, 0.5 after a month, towards 0 after that."""

    try:
        when = datetime.fromisoformat(updated)
    except (TypeError, ValueError):
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    days = max((now - when).total_seconds(), 0) / 86_400
    return 1 / (1 + days / 30)
