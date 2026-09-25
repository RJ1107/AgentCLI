from textkit.caesar import caesar


def test_caesar():
    assert caesar("abc", 1) == "bcd"
    assert caesar("xyz", 3) == "abc"
    assert caesar("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert caesar("bcd", -1) == "abc"
    assert caesar("abc", 52) == "abc"
