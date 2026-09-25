from textkit.wrap import wrap


def test_wrap():
    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]
    assert wrap("a bb ccc", 3) == ["a", "bb", "ccc"]
    assert wrap("  ", 5) == []
    assert wrap("supercalifragilistic is long", 8) == ["supercalifragilistic", "is long"]
    assert wrap("one two three", 100) == ["one two three"]
