from numkit.luhn import luhn_valid


def test_luhn():
    assert luhn_valid("4539 3195 0343 6467")
    assert not luhn_valid("8273 1232 7352 0569")
    assert not luhn_valid("0")
    assert luhn_valid("059")
    assert not luhn_valid("055a 444 285")
