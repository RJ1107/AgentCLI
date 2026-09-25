"""Write the harder "hardkit" task: six modules with tricky specs.

Expected values come from Python itself (re.fullmatch, the Python evaluator) or from small
reference implementations here, so the tests are right by construction. The agent must
implement each spec without those shortcuts, which the tests check.

    python evals/make_hard_fixture.py
"""

from __future__ import annotations

import heapq
import re
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "fixtures" / "tasks" / "hardkit"
STUB = '    raise NotImplementedError("TODO")\n'

# --------------------------------------------------------------------------- regex

REGEX_DOC = '''def match(pattern: str, text: str) -> bool:
    """True when the whole text matches pattern. Supported: literal characters; "." (any one
    character); classes like "[abc]", ranges "[a-z0-9]", and negated classes "[^...]"; the
    quantifiers "*", "+", "?" applied to the preceding atom (greedy, with backtracking); and
    "\\\\" escaping the next character, so "\\\\." is a literal dot. No groups, alternation, or
    anchors: the match is always of the whole text. Do not use the re module."""
'''
REGEX_CASES = [
    ("a*b", ["b", "aaab", "ab", "a", "abb"]),
    ("a.c", ["abc", "a-c", "ac", "abbc"]),
    ("colou?r", ["color", "colour", "colouur"]),
    ("[a-c]+x", ["abcx", "x", "cax", "adx"]),
    ("[^0-9]+", ["abc", "a1c", "", "--"]),
    ("a\\.b", ["a.b", "axb"]),
    ("x*y*z*", ["", "xxz", "zy", "xyz"]),
    ("a*a", ["a", "aaaa", ""]),
    ("[abc]*c", ["abcc", "c", "ab"]),
    (".*end", ["the end", "end", "endless"]),
    ("ab*c?d+", ["ad", "abbbcdd", "abc", "acd"]),
    ("[0-9]+\\.[0-9]*", ["3.14", "3.", ".5", "10.0"]),
    ("h.?llo", ["hllo", "hello", "heello"]),
]

# --------------------------------------------------------------------------- expr

EXPR_DOC = '''def evaluate(expression: str) -> float:
    """Evaluate arithmetic: numbers (ints and decimals), + - * /, and ** (right-associative,
    binding tighter than unary minus, so -2**2 == -4 and 2**-1 == 0.5), unary minus and plus,
    parentheses, and spaces. Raise ValueError for malformed input and ZeroDivisionError for
    division by zero. Do not use eval, exec, or compile."""
'''
EXPR_OK = [
    "1 + 2 * 3",
    "(1+2)*3",
    "2**3**2",
    "-2**2",
    "2**-1",
    "-(3-5)*2",
    "10/4",
    "3 - -2",
    "+4",
    "2*(3+(4-1))**2",
    "1.5*4-0.5",
    "100 - 2*3**2 / 6",
]
EXPR_BAD = ["1 +", "(1", "1 2", "", "2**", "3 * / 4", ")("]

# --------------------------------------------------------------------------- cron

CRON_DOC = '''def next_run(expression: str, after: datetime) -> datetime:
    """The first minute strictly after `after` (seconds ignored) matching a five-field cron
    expression "minute hour day-of-month month day-of-week". Each field is "*", a number, a
    range "a-b", a step "*/n" or "a-b/n", or a comma list of those. Minutes 0-59, hours 0-23,
    days 1-31, months 1-12, weekdays 0-7 where 0 and 7 are Sunday. When both day-of-month and
    day-of-week are restricted (neither is "*"), a day matches if either matches; otherwise
    both must. Raise ValueError for a malformed expression or out-of-range value."""
'''
CRON_CASES = [
    ("*/15 * * * *", datetime(2026, 9, 25, 10, 7)),
    ("0 9 * * 1-5", datetime(2026, 9, 25, 18, 0)),
    ("30 2 1 * *", datetime(2026, 9, 25, 0, 0)),
    ("0 0 29 2 *", datetime(2026, 3, 1, 0, 0)),
    ("0 12 13 * 5", datetime(2026, 9, 25, 12, 0)),
    ("5,35 */6 * * 0", datetime(2026, 9, 26, 23, 59)),
    ("0 0 * * 7", datetime(2026, 9, 25, 0, 0)),
    ("15 10-12/2 * 1,6 *", datetime(2026, 9, 25, 0, 0)),
    ("59 23 31 12 *", datetime(2026, 12, 31, 23, 59)),
    ("0 8 1-7 * 1", datetime(2026, 9, 25, 9, 0)),
]
CRON_BAD = ["61 * * * *", "* * *", "*/0 * * * *", "* 24 * * *", "a * * * *", "5-2 * * * *"]
RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


def _cron_field(text: str, low: int, high: int) -> set[int]:
    values: set[int] = set()
    for part in text.split(","):
        base, _, step_text = part.partition("/")
        step = int(step_text) if step_text else 1
        if step < 1:
            raise ValueError(part)
        if base == "*":
            start, end = low, high
        elif "-" in base:
            a, b = base.split("-")
            start, end = int(a), int(b)
        else:
            start = end = int(base)
        if not (low <= start <= end <= high):
            raise ValueError(part)
        values.update(range(start, end + 1, step))
    return values


def _cron_next(expression: str, after: datetime) -> datetime:
    fields = expression.split()
    minutes, hours, days, months, weekdays = (
        _cron_field(f, lo, hi) for f, (lo, hi) in zip(fields, RANGES, strict=True)
    )
    weekdays = {0 if d == 7 else d for d in weekdays}
    dom_any, dow_any = fields[2] == "*", fields[4] == "*"
    moment = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60 * 5):
        dow = (moment.weekday() + 1) % 7
        if dom_any or dow_any:
            day_ok = moment.day in days and dow in weekdays
        else:
            day_ok = moment.day in days or dow in weekdays
        if moment.month in months and day_ok and moment.hour in hours and moment.minute in minutes:
            return moment
        moment += timedelta(minutes=1)
    raise AssertionError(expression)


# --------------------------------------------------------------------------- toposort

TOPO_DOC = '''def order(graph: dict[str, list[str]]) -> list[str]:
    """Order nodes so each comes after all of its dependencies. graph maps a node to the
    nodes it depends on; a node that appears only as a dependency is included too. Among
    nodes that are ready at the same time, take the alphabetically smallest first. Raise
    ValueError if there is a cycle."""
'''
TOPO_CASES = [
    {"b": ["a"], "c": ["a"], "a": []},
    {"deploy": ["build", "test"], "test": ["build"], "build": ["fetch"]},
    {"z": [], "y": [], "x": []},
    {"app": ["lib", "utils"], "lib": ["utils"], "docs": []},
    {},
]
TOPO_BAD = [{"a": ["b"], "b": ["a"]}, {"a": ["a"]}, {"a": ["b"], "b": ["c"], "c": ["a"], "d": []}]


def _topo(graph: dict[str, list[str]]) -> list[str]:
    nodes = set(graph) | {d for deps in graph.values() for d in deps}
    waiting = {n: set(graph.get(n, [])) for n in nodes}
    ready = [n for n in nodes if not waiting[n]]
    heapq.heapify(ready)
    out = []
    while ready:
        node = heapq.heappop(ready)
        out.append(node)
        for other in nodes:
            if node in waiting[other]:
                waiting[other].discard(node)
                if not waiting[other]:
                    heapq.heappush(ready, other)
    if len(out) != len(nodes):
        raise ValueError("cycle")
    return out


# --------------------------------------------------------------------------- jsonpointer

POINTER_DOC = '''def resolve(document, pointer: str):
    """Resolve an RFC 6901 JSON pointer. "" is the whole document; otherwise the pointer must
    start with "/", and each token has "~1" decoded to "/" and then "~0" to "~". A token
    indexes a dict by key or a list by a non-negative decimal index without leading zeros.
    Raise ValueError for a pointer not starting with "/" or a malformed list index ("01",
    "-", "x"), KeyError for a missing key, and IndexError for an index past the end."""
'''
POINTER_DOC_DATA = {
    "a": {"b": [10, 20, {"c": "deep"}]},
    "m~n": 1,
    "x/y": 2,
    "": "empty-key",
    "list": [],
}
POINTER_OK = ["", "/a/b/0", "/a/b/2/c", "/m~0n", "/x~1y", "/", "/a/b"]
POINTER_BAD = [
    ("a", "ValueError"),
    ("/a/b/01", "ValueError"),
    ("/a/b/-", "ValueError"),
    ("/nope", "KeyError"),
    ("/a/b/3", "IndexError"),
    ("/list/0", "IndexError"),
]


def _resolve(doc, pointer):
    if pointer == "":
        return doc
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        doc = doc[int(token)] if isinstance(doc, list) else doc[token]
    return doc


# --------------------------------------------------------------------------- allocate

ALLOC_DOC = '''def allocate(total: int, weights: list[int]) -> list[int]:
    """Split a non-negative integer total in proportion to non-negative weights, using the
    largest remainder method: each share is first rounded down, then the leftover units go
    one at a time to the shares with the largest fractional remainders, ties going to the
    earlier item. The result sums to total. Raise ValueError when total is negative, a
    weight is negative, or no weight is positive."""
'''
ALLOC_CASES = [
    (100, [1, 1, 1]),
    (10, [3, 3, 3]),
    (7, [1, 2, 4]),
    (0, [5, 5]),
    (101, [50, 50, 1]),
    (5, [0, 1, 0]),
    (1000, [7, 13, 29, 51]),
    (3, [1, 1, 1, 1, 1]),
]
ALLOC_BAD = [(-1, [1]), (5, [0, 0]), (5, [1, -1])]


def _allocate(total, weights):
    whole = sum(weights)
    exact = [Fraction(total * w, whole) for w in weights]
    shares = [int(x) for x in exact]
    left = total - sum(shares)
    by_remainder = sorted(range(len(weights)), key=lambda i: (-(exact[i] - shares[i]), i))
    for i in by_remainder[:left]:
        shares[i] += 1
    return shares


# --------------------------------------------------------------------------- write


def main() -> None:
    package, tests = ROOT / "hardkit", ROOT / "tests"
    package.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    for path in ROOT.rglob("*"):
        if path.is_file():
            path.unlink()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (ROOT / "conftest.py").write_text("", encoding="utf-8")

    no_shortcut = (
        "\n\ndef test_no_shortcut():\n"
        "    source = Path({path!r}).read_text(encoding='utf-8')\n"
        "    for banned in {banned!r}:\n"
        "        assert banned not in source, f'{{banned}} is not allowed'\n"
    )

    modules = {}

    lines = ["from pathlib import Path\n\nfrom hardkit.regex import match\n\n\ndef test_regex():"]
    for pattern, texts in REGEX_CASES:
        for text in texts:
            expected = re.fullmatch(pattern, text) is not None
            lines.append(f"    assert match({pattern!r}, {text!r}) is {expected}")
    modules["regex"] = (
        REGEX_DOC,
        "\n".join(lines)
        + no_shortcut.format(path="hardkit/regex.py", banned=["import re", "from re "]),
    )

    lines = [
        "import pytest\nfrom pathlib import Path\n\nfrom hardkit.expr import evaluate\n\n\ndef test_expr():"
    ]
    for text in EXPR_OK:
        lines.append(f"    assert evaluate({text!r}) == pytest.approx({eval(text)!r})")  # noqa: S307
    lines.append("    with pytest.raises(ZeroDivisionError):\n        evaluate('1/0')")
    for text in EXPR_BAD:
        lines.append(f"    with pytest.raises(ValueError):\n        evaluate({text!r})")
    modules["expr"] = (
        EXPR_DOC,
        "\n".join(lines)
        + no_shortcut.format(path="hardkit/expr.py", banned=["eval(", "exec(", "compile("]),
    )

    lines = [
        "import datetime\n\nimport pytest\n\nfrom hardkit.cron import next_run\n\n\ndef test_cron():"
    ]
    for expression, after in CRON_CASES:
        lines.append(
            f"    assert next_run({expression!r}, {after!r}) == {_cron_next(expression, after)!r}"
        )
    for expression in CRON_BAD:
        lines.append(
            f"    with pytest.raises(ValueError):\n        next_run({expression!r}, datetime.datetime(2026, 1, 1))"
        )
    modules["cron"] = ("from datetime import datetime\n\n\n" + CRON_DOC, "\n".join(lines) + "\n")

    lines = ["import pytest\n\nfrom hardkit.toposort import order\n\n\ndef test_order():"]
    for graph in TOPO_CASES:
        lines.append(f"    assert order({graph!r}) == {_topo(graph)!r}")
    for graph in TOPO_BAD:
        lines.append(f"    with pytest.raises(ValueError):\n        order({graph!r})")
    modules["toposort"] = (TOPO_DOC, "\n".join(lines) + "\n")

    lines = [
        "import pytest\n\nfrom hardkit.jsonpointer import resolve\n",
        f"DOC = {POINTER_DOC_DATA!r}\n\n\ndef test_resolve():",
    ]
    for pointer in POINTER_OK:
        lines.append(
            f"    assert resolve(DOC, {pointer!r}) == {_resolve(POINTER_DOC_DATA, pointer)!r}"
        )
    for pointer, error in POINTER_BAD:
        lines.append(f"    with pytest.raises({error}):\n        resolve(DOC, {pointer!r})")
    modules["jsonpointer"] = (POINTER_DOC, "\n".join(lines) + "\n")

    lines = ["import pytest\n\nfrom hardkit.allocate import allocate\n\n\ndef test_allocate():"]
    for total, weights in ALLOC_CASES:
        lines.append(f"    assert allocate({total}, {weights!r}) == {_allocate(total, weights)!r}")
    for total, weights in ALLOC_BAD:
        lines.append(f"    with pytest.raises(ValueError):\n        allocate({total}, {weights!r})")
    modules["allocate"] = (ALLOC_DOC, "\n".join(lines) + "\n")

    for name, (source, test) in modules.items():
        (package / f"{name}.py").write_text(source + STUB, encoding="utf-8")
        (tests / f"test_{name}.py").write_text(test + "\n", encoding="utf-8")
    names = ", ".join(f"hardkit/{name}.py" for name in modules)
    (ROOT / "TASK.md").write_text(
        f"""Implement every function that raises NotImplementedError("TODO") in
{names}. Follow each docstring exactly. The six modules are independent of each other.
Tests are in tests/; run them with `python -m pytest -q`. Do not change the tests.
""",
        encoding="utf-8",
    )
    print(f"wrote hardkit to {ROOT}")


if __name__ == "__main__":
    main()
