"""Write the coding-task fixtures used by the team and model experiments.

Each task is a small package with six independent modules. Every module has one function
(or class) left as a stub with a docstring, and its own test file. A task passes when all
of its tests pass; the six modules share nothing, so a planner can hand them to six workers.

    python evals/make_task_fixtures.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "fixtures" / "tasks"

STUB = '    raise NotImplementedError("TODO")\n'
METHOD_STUB = '        raise NotImplementedError("TODO")\n'

TASKS: dict[str, dict[str, tuple[str, str]]] = {
    "textkit": {
        "slug": (
            "def slugify(text: str) -> str:\n"
            '    """Lowercase, turn every run of non-alphanumeric characters into one "-",\n'
            '    and strip "-" from both ends. slugify("  Hello, World!  ") == "hello-world"."""\n',
            "from textkit.slug import slugify\n\n"
            "def test_slug():\n"
            '    assert slugify("  Hello, World!  ") == "hello-world"\n'
            '    assert slugify("a--b__c") == "a-b-c"\n'
            '    assert slugify("Already-Slug") == "already-slug"\n'
            '    assert slugify("!!!") == ""\n'
            '    assert slugify("Python 3.12 Release") == "python-3-12-release"\n',
        ),
        "wrap": (
            "def wrap(text: str, width: int) -> list[str]:\n"
            '    """Greedy word wrap: split on whitespace, put as many words on each line as fit\n'
            "    in width (words joined by single spaces). A word longer than width gets a line\n"
            '    to itself, unbroken. Empty or blank text gives []."""\n',
            "from textkit.wrap import wrap\n\n"
            "def test_wrap():\n"
            '    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]\n'
            '    assert wrap("a bb ccc", 3) == ["a", "bb", "ccc"]\n'
            '    assert wrap("  ", 5) == []\n'
            '    assert wrap("supercalifragilistic is long", 8) == ["supercalifragilistic", "is long"]\n'
            '    assert wrap("one two three", 100) == ["one two three"]\n',
        ),
        "roman": (
            "def to_roman(number: int) -> str:\n"
            '    """Integer 1..3999 to a Roman numeral with subtractive forms (4 -> "IV").\n'
            '    Raise ValueError outside 1..3999."""\n',
            "import pytest\nfrom textkit.roman import to_roman\n\n"
            "def test_roman():\n"
            '    assert to_roman(4) == "IV"\n'
            '    assert to_roman(1994) == "MCMXCIV"\n'
            '    assert to_roman(3999) == "MMMCMXCIX"\n'
            '    assert to_roman(58) == "LVIII"\n'
            "    with pytest.raises(ValueError):\n"
            "        to_roman(0)\n",
        ),
        "palindrome": (
            "def is_palindrome(text: str) -> bool:\n"
            '    """True when text reads the same backwards, ignoring case and every character\n'
            '    that is not a letter or digit. The empty string is a palindrome."""\n',
            "from textkit.palindrome import is_palindrome\n\n"
            "def test_palindrome():\n"
            '    assert is_palindrome("A man, a plan, a canal: Panama")\n'
            '    assert not is_palindrome("race a car")\n'
            '    assert is_palindrome("")\n'
            "    assert is_palindrome(\"No 'x' in Nixon\")\n",
        ),
        "caesar": (
            "def caesar(text: str, shift: int) -> str:\n"
            '    """Shift ASCII letters by shift places, wrapping within a-z and A-Z and keeping\n'
            '    case; everything else is unchanged. Negative shifts go backwards."""\n',
            "from textkit.caesar import caesar\n\n"
            "def test_caesar():\n"
            '    assert caesar("abc", 1) == "bcd"\n'
            '    assert caesar("xyz", 3) == "abc"\n'
            '    assert caesar("Hello, World!", 13) == "Uryyb, Jbeyq!"\n'
            '    assert caesar("bcd", -1) == "abc"\n'
            '    assert caesar("abc", 52) == "abc"\n',
        ),
        "vowels": (
            "def count_vowels(text: str) -> dict[str, int]:\n"
            '    """Count a, e, i, o, u case-insensitively. Return a dict with exactly those five\n'
            '    keys, in that order, zero when absent."""\n',
            "from textkit.vowels import count_vowels\n\n"
            "def test_vowels():\n"
            '    assert count_vowels("Education") == {"a": 1, "e": 1, "i": 1, "o": 1, "u": 1}\n'
            '    assert count_vowels("") == {"a": 0, "e": 0, "i": 0, "o": 0, "u": 0}\n'
            '    assert list(count_vowels("xyz")) == ["a", "e", "i", "o", "u"]\n'
            '    assert count_vowels("AAAee")["a"] == 3\n',
        ),
    },
    "numkit": {
        "duration": (
            "def parse_duration(text: str) -> int:\n"
            '    """Parse durations like "1h30m", "45s", "2d4h", "1h 5m 10s" into seconds. Units:\n'
            "    d, h, m, s, each at most once, in that order; spaces allowed between parts.\n"
            '    Raise ValueError for anything else, including an empty string."""\n',
            "import pytest\nfrom numkit.duration import parse_duration\n\n"
            "def test_duration():\n"
            '    assert parse_duration("1h30m") == 5400\n'
            '    assert parse_duration("45s") == 45\n'
            '    assert parse_duration("2d4h") == 187200\n'
            '    assert parse_duration("1h 5m 10s") == 3910\n'
            '    for bad in ["", "10", "5m1h", "1x"]:\n'
            "        with pytest.raises(ValueError):\n"
            "            parse_duration(bad)\n",
        ),
        "bytesize": (
            "def format_bytes(size: int) -> str:\n"
            '    """Human-readable size with 1024 steps and units B, KB, MB, GB, TB. Bytes are\n'
            '    shown as an integer ("512 B"); larger units with one decimal ("1.5 KB")."""\n',
            "from numkit.bytesize import format_bytes\n\n"
            "def test_bytes():\n"
            '    assert format_bytes(512) == "512 B"\n'
            '    assert format_bytes(1536) == "1.5 KB"\n'
            '    assert format_bytes(1048576) == "1.0 MB"\n'
            '    assert format_bytes(5 * 1024 ** 4) == "5.0 TB"\n'
            '    assert format_bytes(0) == "0 B"\n',
        ),
        "luhn": (
            "def luhn_valid(number: str) -> bool:\n"
            '    """Luhn checksum. Spaces are ignored; any other non-digit makes it invalid, as\n'
            '    does a string with fewer than two digits."""\n',
            "from numkit.luhn import luhn_valid\n\n"
            "def test_luhn():\n"
            '    assert luhn_valid("4539 3195 0343 6467")\n'
            '    assert not luhn_valid("8273 1232 7352 0569")\n'
            '    assert not luhn_valid("0")\n'
            '    assert luhn_valid("059")\n'
            '    assert not luhn_valid("055a 444 285")\n',
        ),
        "intervals": (
            "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
            '    """Merge overlapping or touching closed intervals ((1, 3) and (3, 5) merge).\n'
            '    Input may be unsorted; output is sorted by start."""\n',
            "from numkit.intervals import merge_intervals\n\n"
            "def test_intervals():\n"
            "    assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]\n"
            "    assert merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]\n"
            "    assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]\n"
            "    assert merge_intervals([]) == []\n",
        ),
        "median": (
            "def running_median(values: list[float]) -> list[float]:\n"
            '    """After each value, the median of all values so far (the mean of the middle\n'
            '    two when the count is even). Should be O(n log n)."""\n',
            "from numkit.median import running_median\n\n"
            "def test_median():\n"
            "    assert running_median([5, 15, 1, 3]) == [5, 10, 5, 4]\n"
            "    assert running_median([]) == []\n"
            "    assert running_median([2, 2, 2]) == [2, 2, 2]\n"
            "    assert running_median(list(range(1, 10001)))[-1] == 5000.5\n",
        ),
        "primes": (
            "def primes_up_to(limit: int) -> list[int]:\n"
            '    """All primes <= limit in ascending order (sieve). [] when limit < 2."""\n',
            "from numkit.primes import primes_up_to\n\n"
            "def test_primes():\n"
            "    assert primes_up_to(10) == [2, 3, 5, 7]\n"
            "    assert primes_up_to(1) == []\n"
            "    assert primes_up_to(2) == [2]\n"
            "    assert len(primes_up_to(100000)) == 9592\n",
        ),
    },
    "collkit": {
        "lru": (
            "class LRUCache:\n"
            '    """Fixed-capacity cache. get(key) returns the value or None and marks the key\n'
            "    as recently used; put(key, value) inserts or updates and marks it recently used,\n"
            '    evicting the least recently used key when over capacity. len() gives the size."""\n\n'
            "    def __init__(self, capacity: int):\n"
            + METHOD_STUB
            + "\n    def get(self, key):\n"
            + METHOD_STUB
            + "\n    def put(self, key, value) -> None:\n"
            + METHOD_STUB
            + "\n    def __len__(self) -> int:\n",
            "from collkit.lru import LRUCache\n\n"
            "def test_lru():\n"
            "    cache = LRUCache(2)\n"
            '    cache.put("a", 1)\n'
            '    cache.put("b", 2)\n'
            '    assert cache.get("a") == 1\n'
            '    cache.put("c", 3)\n'
            '    assert cache.get("b") is None\n'
            '    assert cache.get("c") == 3\n'
            '    cache.put("a", 10)\n'
            '    assert cache.get("a") == 10\n'
            "    assert len(cache) == 2\n",
        ),
        "flatten": (
            "def flatten(items) -> list:\n"
            '    """Flatten arbitrarily nested lists and tuples into one list, depth-first.\n'
            '    Strings, bytes, and dicts are kept as single items."""\n',
            "from collkit.flatten import flatten\n\n"
            "def test_flatten():\n"
            "    assert flatten([1, [2, [3, (4, 5)]], 6]) == [1, 2, 3, 4, 5, 6]\n"
            '    assert flatten(["ab", ["cd"]]) == ["ab", "cd"]\n'
            "    assert flatten([]) == []\n"
            '    assert flatten([{"a": 1}, [[]]]) == [{"a": 1}]\n',
        ),
        "chunks": (
            "def chunked(items: list, size: int) -> list[list]:\n"
            '    """Split into consecutive chunks of size; the last may be shorter.\n'
            '    Raise ValueError when size < 1."""\n',
            "import pytest\nfrom collkit.chunks import chunked\n\n"
            "def test_chunks():\n"
            "    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            "    assert chunked([], 3) == []\n"
            "    assert chunked([1, 2], 5) == [[1, 2]]\n"
            "    with pytest.raises(ValueError):\n"
            "        chunked([1], 0)\n",
        ),
        "groupby": (
            "def group_by(items: list, key) -> dict:\n"
            '    """Group items by key(item) into a dict of lists, keeping the order in which\n'
            '    keys first appear and the order of items within each group."""\n',
            "from collkit.groupby import group_by\n\n"
            "def test_group_by():\n"
            '    words = ["apple", "bob", "avocado", "cat", "banana"]\n'
            "    grouped = group_by(words, lambda w: w[0])\n"
            '    assert grouped == {"a": ["apple", "avocado"], "b": ["bob", "banana"], "c": ["cat"]}\n'
            '    assert list(grouped) == ["a", "b", "c"]\n'
            "    assert group_by([], len) == {}\n",
        ),
        "dedupe": (
            "def dedupe(items: list, key=None) -> list:\n"
            '    """Drop repeats, keeping the first occurrence and the original order. With key,\n'
            "    two items are repeats when key(item) is equal. Items may be unhashable when a\n"
            '    hashable key is given."""\n',
            "from collkit.dedupe import dedupe\n\n"
            "def test_dedupe():\n"
            "    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
            '    assert dedupe(["a", "A", "b"], key=str.lower) == ["a", "b"]\n'
            '    assert dedupe([{"id": 1}, {"id": 1}, {"id": 2}], key=lambda d: d["id"]) == [{"id": 1}, {"id": 2}]\n'
            "    assert dedupe([]) == []\n",
        ),
        "topk": (
            "def top_k_frequent(items: list, k: int) -> list:\n"
            '    """The k most frequent items, most frequent first; ties broken by which item\n'
            '    appeared first in the input."""\n',
            "from collkit.topk import top_k_frequent\n\n"
            "def test_topk():\n"
            "    assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]\n"
            '    assert top_k_frequent(["b", "a", "b", "a", "c"], 2) == ["b", "a"]\n'
            "    assert top_k_frequent([4], 3) == [4]\n"
            "    assert top_k_frequent([], 1) == []\n",
        ),
    },
    "parsekit": {
        "csvline": (
            "def parse_csv_line(line: str) -> list[str]:\n"
            '    """Split one CSV line on commas. Fields may be wrapped in double quotes, which\n'
            '    may contain commas and doubled quotes ("" means one quote). Do not use the csv\n'
            '    module."""\n',
            "from parsekit.csvline import parse_csv_line\n\n"
            "def test_csv():\n"
            '    assert parse_csv_line("a,b,c") == ["a", "b", "c"]\n'
            '    assert parse_csv_line(\'"a,b",c\') == ["a,b", "c"]\n'
            '    assert parse_csv_line(\'"say ""hi""",x\') == [\'say "hi"\', "x"]\n'
            '    assert parse_csv_line("a,,b") == ["a", "", "b"]\n'
            '    assert parse_csv_line("") == [""]\n',
        ),
        "query": (
            "def parse_query(query: str) -> dict[str, list[str]]:\n"
            '    """Parse a URL query string ("a=1&b=2&a=3", optional leading "?") into a dict of\n'
            '    lists in order of appearance. "+" is a space and %XX escapes are decoded; a key\n'
            '    without "=" gets "". Do not use urllib."""\n',
            "from parsekit.query import parse_query\n\n"
            "def test_query():\n"
            '    assert parse_query("?a=1&b=2&a=3") == {"a": ["1", "3"], "b": ["2"]}\n'
            '    assert parse_query("q=hello+world%21") == {"q": ["hello world!"]}\n'
            '    assert parse_query("flag") == {"flag": [""]}\n'
            '    assert parse_query("") == {}\n',
        ),
        "semver": (
            "def compare_versions(a: str, b: str) -> int:\n"
            '    """Compare semantic versions "MAJOR.MINOR.PATCH" with an optional "-prerelease"\n'
            "    of dot-separated identifiers. Return -1, 0, or 1. A prerelease sorts before the\n"
            '    release; numeric identifiers compare as numbers and sort before text ones."""\n',
            "from parsekit.semver import compare_versions\n\n"
            "def test_semver():\n"
            '    assert compare_versions("1.2.3", "1.2.3") == 0\n'
            '    assert compare_versions("1.10.0", "1.9.9") == 1\n'
            '    assert compare_versions("1.0.0-alpha", "1.0.0") == -1\n'
            '    assert compare_versions("1.0.0-alpha.2", "1.0.0-alpha.10") == -1\n'
            '    assert compare_versions("1.0.0-beta", "1.0.0-alpha.1") == 1\n',
        ),
        "brackets": (
            "def balanced(text: str) -> bool:\n"
            '    """True when (), [], {} are balanced and properly nested; other characters are\n'
            '    ignored."""\n',
            "from parsekit.brackets import balanced\n\n"
            "def test_brackets():\n"
            '    assert balanced("([]{})")\n'
            '    assert not balanced("([)]")\n'
            '    assert balanced("f(x) = [1, {2}]")\n'
            '    assert not balanced("((")\n'
            '    assert balanced("")\n',
        ),
        "rpn": (
            "def eval_rpn(expression: str) -> float:\n"
            '    """Evaluate a space-separated Reverse Polish expression with + - * / on numbers\n'
            '    (ints or decimals, possibly negative like "-3"). Raise ValueError on a bad\n'
            '    token, too few operands, or leftover operands; ZeroDivisionError on / 0."""\n',
            "import pytest\nfrom parsekit.rpn import eval_rpn\n\n"
            "def test_rpn():\n"
            '    assert eval_rpn("3 4 + 2 *") == 14\n'
            '    assert eval_rpn("5 1 2 + 4 * + 3 -") == 14\n'
            '    assert eval_rpn("-3 2 *") == -6\n'
            '    assert eval_rpn("1.5 2 /") == 0.75\n'
            '    for bad in ["1 +", "1 2", "1 x +"]:\n'
            "        with pytest.raises(ValueError):\n"
            "            eval_rpn(bad)\n"
            "    with pytest.raises(ZeroDivisionError):\n"
            '        eval_rpn("1 0 /")\n',
        ),
        "ini": (
            "def parse_ini(text: str) -> dict[str, dict[str, str]]:\n"
            '    """Parse INI text into {section: {key: value}}. Keys before any [section] go in\n'
            '    "default". Lines starting with ";" or "#" and blank lines are skipped; keys and\n'
            '    values are stripped; a repeated key keeps the last value. Do not use configparser."""\n',
            "from parsekit.ini import parse_ini\n\n"
            "def test_ini():\n"
            '    text = "name = top\\n; comment\\n[db]\\nhost = localhost\\nport=5432\\n\\n[db]\\nport = 6543\\n"\n'
            '    assert parse_ini(text) == {"default": {"name": "top"}, "db": {"host": "localhost", "port": "6543"}}\n'
            '    assert parse_ini("") == {}\n',
        ),
    },
}


def main() -> None:
    # Clear files rather than the folders, which a shell may be sitting in.
    if ROOT.exists():
        for path in ROOT.rglob("*"):
            if path.is_file():
                path.unlink()
    for task, modules in TASKS.items():
        base = ROOT / task
        package = base / task
        tests = base / "tests"
        package.mkdir(parents=True, exist_ok=True)
        tests.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        for module, (source, test) in modules.items():
            stub = METHOD_STUB if source.startswith("class ") else STUB
            (package / f"{module}.py").write_text(source + stub, encoding="utf-8")
            (tests / f"test_{module}.py").write_text(test, encoding="utf-8")
        names = ", ".join(f"{task}/{module}.py" for module in modules)
        (base / "TASK.md").write_text(
            textwrap.dedent(
                f"""\
                Implement every function or method that raises NotImplementedError("TODO") in
                {names}. Follow each docstring. The six modules are independent of each other.
                Tests are in tests/; run them with `python -m pytest -q`. Do not change the tests.
                """
            ),
            encoding="utf-8",
        )
        (base / "conftest.py").write_text("", encoding="utf-8")
    print(f"wrote {len(TASKS)} tasks to {ROOT}")


if __name__ == "__main__":
    main()
