from textkit.vowels import count_vowels


def test_vowels():
    assert count_vowels("Education") == {"a": 1, "e": 1, "i": 1, "o": 1, "u": 1}
    assert count_vowels("") == {"a": 0, "e": 0, "i": 0, "o": 0, "u": 0}
    assert list(count_vowels("xyz")) == ["a", "e", "i", "o", "u"]
    assert count_vowels("AAAee")["a"] == 3
